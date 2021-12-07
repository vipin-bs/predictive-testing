#!/usr/bin/env python3

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import json
import os
import pandas as pd  # type: ignore[import]
import pickle
from pathlib import Path
from pyspark.sql import DataFrame, SparkSession, functions as funcs
from typing import Any, Dict, List, Optional, Tuple

import features
from auto_tracking import auto_tracking, auto_tracking_with, save_data_lineage
from ptesting import github_utils, train


def _setup_logger() -> Any:
    import logging
    logfmt = '%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=logfmt, datefmt=datefmt)
    return logging


_logger = _setup_logger()


def _to_pandas(name: str) -> Any:
    @auto_tracking_with(name)
    def _func(df: DataFrame) -> pd.DataFrame:
        return df.toPandas()

    return _func


def _to_spark(name: str) -> Any:
    @auto_tracking_with(name)
    def _func(spark: SparkSession, pdf: pd.DataFrame) -> DataFrame:
        return spark.createDataFrame(pdf)

    return _func


# Our predictive model uses LightGBM, an implementation of gradient-boosted decision trees.
# This is because the algorithm has desirable properties for this use-case
# (the reason is the same with the Facebook one):
#  - normalizing feature values are less required
#  - fast model training on commodity hardware
#  - robustness in imbalanced datasets
@auto_tracking
def _build_predictive_model(df: DataFrame, to_features: Any) -> Any:
    pdf = _to_pandas('_to_pandas_for_building_predictive_model')(to_features(df))
    X = pdf[pdf.columns[pdf.columns != 'failed']]  # type: ignore
    y = pdf['failed']
    X, y = train.rebalance_training_data(X, y, coeff=1.0)
    clf, score = train.build_model(X, y, opts={'hp.timeout': '3600', 'hp.no_progress_loss': '1'})
    _logger.info(f"model score: {score}")
    return clf


@auto_tracking
def _train_test_split(df: DataFrame, test_ratio: float) -> Tuple[DataFrame, DataFrame]:
    test_nrows = int(df.count() * test_ratio)
    test_df = df.orderBy(funcs.expr('to_timestamp(commit_date, "yyy/MM/dd HH:mm:ss")').desc()).limit(test_nrows)
    train_df = df.subtract(test_df)
    return train_df, test_df


@auto_tracking
def _predict_failed_probs_for_tests(test_df: DataFrame, clf: Any, to_features: Any) -> DataFrame:
    test_feature_df = to_features(test_df)
    pdf = _to_pandas('_to_pandas_for_failed_probs')(test_feature_df.selectExpr('sha', 'test'))
    X = _to_pandas('_to_pandas_for_evaluating_model')(test_feature_df.drop('sha', 'failed', 'test'))
    predicted = clf.predict_proba(X)
    pmf = map(lambda p: {"classes": clf.classes_.tolist(), "probs": p.tolist()}, predicted)
    pmf = map(lambda p: json.dumps(p), pmf)  # type: ignore
    pdf['predicted'] = pd.Series(list(pmf))
    to_map_expr = funcs.expr('from_json(predicted, "classes array<string>, probs array<double>")')
    to_failed_prob = 'map_from_arrays(pmf.classes, pmf.probs)["1"] failed_prob'
    df_with_failed_probs = _to_spark('predicted_failed_probs')(test_df.sql_ctx.sparkSession, pdf) \
        .withColumn('pmf', to_map_expr) \
        .selectExpr('sha', 'test', to_failed_prob)

    # Applied an emprical rule: set 1.0 to the failed probs of udpated tests
    # TODO: Removes package-depenent stuffs
    import spark_utils
    df_with_failed_probs = features.set_highest_failed_probs_for_updated_tests(
        test_df, df_with_failed_probs,
        spark_utils.RE_PARSE_PATH_PATTERN)

    compare = lambda x, y: \
        f"case when {x}.failed_prob < {y}.failed_prob then 1 " \
        f"when {x}.failed_prob > {y}.failed_prob then -1 " \
        "else 0 end"
    df_with_failed_probs = df_with_failed_probs.groupBy('sha') \
        .agg(funcs.expr('collect_set(named_struct("test", test, "failed_prob", failed_prob))').alias('tests')) \
        .selectExpr('sha', f'filter(tests, t -> t.test IS NOT NULL) tests') \
        .selectExpr('sha', f'array_sort(tests, (l, r) -> {compare("l", "r")}) tests')

    return df_with_failed_probs


@auto_tracking
def _predict_failed_probs(df: DataFrame, clf: Any) -> DataFrame:
    pdf = _to_pandas('to_pandas_for_predicting_failed_probs')(df)
    predicted = clf.predict_proba(pdf.drop('test', axis=1))
    pmf = map(lambda p: {"classes": clf.classes_.tolist(), "probs": p.tolist()}, predicted)
    pmf = map(lambda p: json.dumps(p), pmf)  # type: ignore
    pdf['predicted'] = pd.Series(list(pmf))
    to_map_expr = funcs.expr('from_json(predicted, "classes array<string>, probs array<double>")')
    to_failed_prob = 'map_from_arrays(pmf.classes, pmf.probs)["1"] failed_prob'
    df_with_failed_probs = _to_spark('predicted_failed_probs')(df.sql_ctx.sparkSession, pdf[['test', 'predicted']]) \
        .withColumn('pmf', to_map_expr) \
        .selectExpr('test', to_failed_prob)

    compare = lambda x, y: \
        f"case when {x}.failed_prob < {y}.failed_prob then 1 " \
        f"when {x}.failed_prob > {y}.failed_prob then -1 " \
        "else 0 end"
    df_with_failed_probs = df_with_failed_probs \
        .selectExpr('collect_set(named_struct("test", test, "failed_prob", failed_prob)) tests') \
        .selectExpr(f'filter(tests, t -> t.test IS NOT NULL) tests') \
        .selectExpr(f'array_sort(tests, (l, r) -> {compare("l", "r")}) tests')

    return df_with_failed_probs


@auto_tracking
def _compute_eval_stats(df: DataFrame) -> DataFrame:
    rank = 'array_position(transform(tests, x -> x.test), failed_test) rank'
    to_test_map = 'map_from_arrays(transform(tests, x -> x.test), transform(tests, x -> x.failed_prob)) tests'
    return df.selectExpr('sha', 'explode(failed_tests) failed_test', 'tests') \
        .selectExpr('sha', 'failed_test', rank, 'tests[0].failed_prob max_score', to_test_map) \
        .selectExpr('sha', 'failed_test', 'rank', 'tests[failed_test] score', 'max_score')


@auto_tracking
def _compute_eval_metrics(predicted: DataFrame, total_num_tests: int, eval_num_tests: List[int]) -> Any:
    # This method computes metrics to measure the quality of a selected test set; "test recall", which is computed
    # in the method, represents the emprical probability of a particular test selection strategy catching
    # an individual failure (See "Section 3.B. Measuring Quality of Test Selection" in the Facebook paper [1]).
    # The equation to compute the metric is defined as follows:
    #  - TestRecall(D) = \frac{\sum_{d \in D} |SelectedTests(d) \bigcamp F_{d}|}{\sum_{d \in D} |F_{d}|}
    # , where D is a set of code changes and F_{d} is a set of failed tests.
    #
    # TODO: Computes a "chnage recall" metric.
    def _metric(num_tests: int, score_thres: float = 0.0) -> Tuple[float, float]:
        # TODO: Needs to make 'num_dependent_tests' more precise
        filtered_test_expr = funcs.expr(f'filter(tests, x -> x.failed_prob >= {score_thres})')
        eval_df = predicted.withColumn('filtered_tests', filtered_test_expr) \
            .withColumn('tests_', funcs.expr(f'case when size(filtered_tests) <= {num_tests} then filtered_tests '
                                             f'else slice(tests, 1, {num_tests}) end')) \
            .selectExpr('sha', 'failed_tests', 'transform(tests_, x -> x.test) tests') \
            .selectExpr(
                'sha',
                f'{total_num_tests} num_dependent_tests',
                'size(failed_tests) num_failed_tests',
                'size(tests) num_tests',
                'size(array_intersect(failed_tests, tests)) covered') \
            .selectExpr(
                'SUM(covered) / SUM(num_failed_tests) recall',
                'SUM(num_tests) / SUM(num_dependent_tests) ratio')
        row = eval_df.collect()[0]
        return row.recall, row.ratio

    to_metric_values = lambda m: {'num_tests': m[0], 'test_recall': m[1], 'selection_ratio': m[2]}
    metrics = [(num_tests, *_metric(num_tests)) for num_tests in eval_num_tests]
    metrics = list(map(to_metric_values, metrics))  # type: ignore
    return metrics


def _format_eval_metrics(metrics: List[Dict[str, Any]]) -> str:
    strbuf: List[str] = []
    strbuf.append('|  #tests  |  test recall  |  selection ratio  |')
    strbuf.append('| ---- | ---- | ---- |')
    for m in metrics:  # type: ignore
        strbuf.append(f'|  {m["num_tests"]}  |  {m["test_recall"]}  |  {m["selection_ratio"]}  |')  # type: ignore

    return '\n'.join(strbuf)


def _save_metrics_as_chart(output_path: str, metrics: List[Dict[str, Any]], max_test_num: int) -> None:
    import altair as alt
    x_opts = {
        'scale': alt.Scale(domain=[0, max_test_num]),
        'axis': alt.Axis(title='#tests', titleFontSize=16, labelFontSize=14)
    }
    y_opts = {
        'scale': alt.Scale(domain=[0.0, 1.0]),
        'axis': alt.Axis(title='test recall', titleFontSize=16, labelFontSize=14)
    }
    df = pd.DataFrame(metrics)
    plot = alt.Chart(df).mark_point().encode(x=alt.X("num_tests", **x_opts), y=alt.Y("test_recall", **y_opts)) \
        .properties(width=600, height=400) \
        .interactive()

    plot.save(output_path)


@auto_tracking
def _train_and_eval_ptest_model(output_path: str, spark: SparkSession, df: DataFrame,
                                test_files: Dict[str, str],
                                commits: List[Tuple[str, str, List[str]]],
                                correlated_files: Dict[str, List[str]],
                                dep_graph: Dict[str, List[str]],
                                included_tests: List[str],
                                updated_file_stats: Dict[str, List[Tuple[str, str, str, str]]],
                                contributor_stats: Optional[List[Tuple[str, str]]],
                                test_ratio: float = 0.20) -> None:
    @auto_tracking
    def num_failed_tests(df: DataFrame) -> int:
        return df.selectExpr('explode(failed_tests)').count()

    train_df, test_df = _train_test_split(df, test_ratio=test_ratio)
    _logger.info('Split data: #total={}(#failed={}), #train={}(#failed={}), #test={}(#failed={})'.format(
        df.count(), num_failed_tests(df), train_df.count(), num_failed_tests(train_df),
        test_df.count(), num_failed_tests(test_df)))

    correlated_files_from_failed_tests = features.extract_correlated_files_from_failed_tests(train_df)
    for key, value in correlated_files_from_failed_tests.items():
        correlated_files[key] = list(set(correlated_files[key] + value)) if key in correlated_files else value

    repo_commits = list(map(lambda c: github_utils.from_github_datetime(c[0]), commits))
    failed_tests = features.build_failed_tests(train_df)
    to_train_features, to_test_features = features.create_train_test_pipeline(
        spark, test_files, repo_commits, dep_graph, correlated_files, included_tests, updated_file_stats,
        contributor_stats, failed_tests)

    clf = _build_predictive_model(train_df, to_train_features)

    with open(f"{output_path}/correlated-files-delta.json", 'w') as f:
        f.write(json.dumps(correlated_files_from_failed_tests, indent=2))
    with open(f"{output_path}/failed-tests.json", 'w') as f:
        f.write(json.dumps(failed_tests, indent=2))
    with open(f"{output_path}/model.pkl", 'wb') as f:  # type: ignore
        pickle.dump(clf, f)  # type: ignore

    predicted = _predict_failed_probs_for_tests(test_df.drop('failed_tests'), clf, to_test_features)
    predicted = test_df.selectExpr('sha', 'failed_tests').join(predicted, 'sha', 'LEFT_OUTER') \
        .selectExpr('sha', 'failed_tests', 'coalesce(tests, array()) tests')

    num_test_files = len(test_files)
    metrics = _compute_eval_metrics(predicted, total_num_tests=num_test_files,
                                    eval_num_tests=list(range(4, num_test_files + 1, 4)))
    stats = _compute_eval_stats(predicted)

    with open(f"{output_path}/model-eval-stats.json", 'w') as f:  # type: ignore
        f.write(json.dumps(stats.toPandas().to_dict(orient='records'), indent=2))  # type: ignore
    with open(f"{output_path}/model-eval-metric-summary.md", 'w') as f:  # type: ignore
        f.write(_format_eval_metrics([metrics[i] for i in [0, 29, 59, 89, 119]]))  # type: ignore
    with open(f"{output_path}/model-eval-metrics.json", 'w') as f:  # type: ignore
        f.write(json.dumps(metrics, indent=2))  # type: ignore

    _save_metrics_as_chart(f"{output_path}/model-eval-metrics.svg", metrics, len(test_files))


@auto_tracking
def _exclude_tests_from(df: DataFrame, excluded_tests: List[str]) -> DataFrame:
    spark = df.sql_ctx.sparkSession
    excluded_test_df = spark.createDataFrame(pd.DataFrame(excluded_tests, columns=['excluded_test'])) \
        .selectExpr('collect_set(excluded_test) excluded_tests')
    array_except_expr = 'array_except(failed_tests, excluded_tests) failed_tests'
    return df.join(excluded_test_df) \
        .selectExpr('author', 'sha', 'commit_date', array_except_expr, 'files')


def train_main(argv: Any) -> None:
    # Parses command-line arguments for a training mode
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--train-log-data', type=str, required=True)
    parser.add_argument('--test-files', type=str, required=True)
    parser.add_argument('--commits', type=str, required=True)
    parser.add_argument('--correlated-files', type=str, required=True)
    parser.add_argument('--updated-file-stats', type=str, required=True)
    parser.add_argument('--contributor-stats', type=str, required=False)
    parser.add_argument('--build-dep', type=str, required=True)
    parser.add_argument('--excluded-tests', type=str, required=False)
    parser.add_argument('--included-tests', type=str, required=False)
    parser.add_argument('--data-lineage', action='store_true')
    parser.add_argument('--spark-jars', type=str, required=False, default='')
    args = parser.parse_args(argv)

    if not os.path.isdir(args.output):
        raise ValueError(f"Output directory not found in {os.path.abspath(args.output)}")
    if not os.path.isfile(args.train_log_data):
        raise ValueError(f"Training data not found in {os.path.abspath(args.train_log_data)}")
    if not os.path.isfile(args.test_files):
        raise ValueError(f"Test list file not found in {os.path.abspath(args.test_files)}")
    if not os.path.isfile(args.commits):
        raise ValueError(f"Commit history file not found in {os.path.abspath(args.commits)}")
    if not os.path.isfile(args.correlated_files):
        raise ValueError(f"File for file correlation not found in {os.path.abspath(args.correlated_files)}")
    if not os.path.isfile(args.updated_file_stats):
        raise ValueError(f"Updated file stats not found in {os.path.abspath(args.updated_file_stats)}")
    if args.contributor_stats and not os.path.isfile(args.contributor_stats):
        raise ValueError(f"Contributor stats not found in {os.path.abspath(args.contributor_stats)}")
    if args.build_dep and not os.path.isfile(args.build_dep):
        raise ValueError(f"Dependency graph file not found in {os.path.abspath(args.build_dep)}")
    if args.excluded_tests and not os.path.isfile(args.excluded_tests):
        raise ValueError(f"Excluded test list file not found in {os.path.abspath(args.excluded_tests)}")
    if args.included_tests and not os.path.isfile(args.included_tests):
        raise ValueError(f"Included test list file not found in {os.path.abspath(args.included_tests)}")

    test_files = json.loads(Path(args.test_files).read_text())
    commits = json.loads(Path(args.commits).read_text())
    correlated_files = json.loads(Path(args.correlated_files).read_text())
    updated_file_stats = json.loads(Path(args.updated_file_stats).read_text())
    contributor_stats = json.loads(Path(args.contributor_stats).read_text()) \
        if args.contributor_stats else None
    dep_graph = json.loads(Path(args.build_dep).read_text()) \
        if args.build_dep else None
    excluded_tests = json.loads(Path(args.excluded_tests).read_text()) \
        if args.excluded_tests else []
    included_tests = json.loads(Path(args.included_tests).read_text()) \
        if args.included_tests else []

    # Removes comment entries from `excluded_tests`/`included_tests`
    excluded_tests = list(filter(lambda t: not t.startswith('$comment'), excluded_tests))
    included_tests = list(filter(lambda t: not t.startswith('$comment'), included_tests))
    intersected_tests = set(excluded_tests) & set(included_tests)
    if intersected_tests:
        _logger.warning('Some tests exist in both `excluded_tests` and `included_tests`: '
                        f'{",".join(intersected_tests)}')

    # Removes the excluded tests from `test_files`
    test_files = {k: test_files[k] for k in test_files if k not in excluded_tests} \
        if excluded_tests else test_files

    # Initializes a Spark session
    spark = SparkSession.builder \
        .config("spark.jars", args.spark_jars) \
        .enableHiveSupport() \
        .getOrCreate()

    # Suppresses user warinig messages in Python
    import warnings
    warnings.simplefilter("ignore", UserWarning)

    # Suppresses `WARN` messages in JVM
    spark.sparkContext.setLogLevel("ERROR")

    try:
        # Assigns a random string if 'sha' is an empty string
        # TODO: Needs to validate input log data
        expected_input_cols = [
            'author',
            'case when length(sha) > 0 then sha else sha(string(random())) end sha',
            'commit_date',
            'array_distinct(failed_tests) failed_tests',
            'files'
        ]
        log_data_df = spark.read.format('json').load(args.train_log_data) \
            .selectExpr(expected_input_cols)

        # Creates a temp view for making gen'd data lineage easy-to-see
        if args.data_lineage:
            log_data_df.createOrReplaceTempView('train_log_raw_data')

        too_many_failed_tests_df = log_data_df.where('size(failed_tests) > 32')
        if too_many_failed_tests_df.count() > 0:
            rows = too_many_failed_tests_df.selectExpr('sha', 'size(failed_tests) num_failed_tests').collect()
            too_many_tests = map(lambda r: f'{r.sha}({r.num_failed_tests})', rows)
            _logger.warning(f'Too many failed tests found: {",".join(too_many_tests)}')

        # Excludes some tests (e.g., flaky ones) if necessary
        if excluded_tests:
            log_data_df = _exclude_tests_from(log_data_df, excluded_tests)

        # Checks if all the failed tests exist in `test_files`
        failed_tests = log_data_df \
            .selectExpr('explode(failed_tests) failed_test') \
            .selectExpr('collect_set(failed_test) failed_tests') \
            .collect()[0].failed_tests

        unknown_failed_tests = set(failed_tests).difference(set(test_files.keys()))
        if len(unknown_failed_tests) > 0:
            _logger.warning(f'Unknown failed tests found: {",".join(unknown_failed_tests)}')

        _train_and_eval_ptest_model(args.output, spark, log_data_df, test_files, commits,
                                    correlated_files, dep_graph,
                                    included_tests,
                                    updated_file_stats, contributor_stats,
                                    test_ratio=0.10)

        if args.data_lineage:
            save_data_lineage(f'{args.output}/data_lineage', format='svg',
                              contracted=True, overwrite=True)
    finally:
        spark.stop()


def _format_for_selective_tests_in_scalatest(tests: List[str]) -> str:
    selected_tests = []
    for t in tests:
        selected_tests.append(f'TestFailed Some({t}) {t} None')
    return '\n'.join(selected_tests)


def predict_main(argv: Any) -> None:
    # Parses command-line arguments for a prediction mode
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--username', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--num-commits', type=int, required=True)
    parser.add_argument('--num-selected-tests', type=int, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--test-files', type=str, required=True)
    parser.add_argument('--commits', type=str, required=True)
    parser.add_argument('--correlated-files', type=str, required=True)
    parser.add_argument('--correlated-files-delta', type=str, required=True)
    parser.add_argument('--failed-tests', type=str, required=True)
    parser.add_argument('--updated-file-stats', type=str, required=True)
    parser.add_argument('--contributor-stats', type=str, required=False)
    parser.add_argument('--build-dep', type=str, required=True)
    parser.add_argument('--excluded-tests', type=str, required=False)
    parser.add_argument('--included-tests', type=str, required=False)
    parser.add_argument('--format', action='store_true')
    args = parser.parse_args(argv)

    if not os.path.isdir(f'{args.target}/.git'):
        raise ValueError(f"Git-managed directory not found in {os.path.abspath(args.target)}")
    if args.num_commits <= 0:
        raise ValueError(f"Target #commits must be positive, but {args.num_commits}")
    if args.num_selected_tests <= 0:
        raise ValueError(f"Predicted #tests must be positive, but {args.num_selected_tests}")
    if not os.path.isfile(args.model):
        raise ValueError(f"Predictive model not found in {os.path.abspath(args.model)}")
    if not os.path.isfile(args.test_files):
        raise ValueError(f"Test list file not found in {os.path.abspath(args.test_files)}")
    if not os.path.isfile(args.commits):
        raise ValueError(f"Commit history file not found in {os.path.abspath(args.commits)}")
    if not os.path.isfile(args.correlated_files):
        raise ValueError(f"File for file correlation not found in {os.path.abspath(args.correlated_files)}")
    if not os.path.isfile(args.correlated_files_delta):
        raise ValueError("File for extra file correlation not found in {os.path.abspath(args.correlated_files_delta)}")
    if not os.path.isfile(args.failed_tests):
        raise ValueError(f"Failed test list file not found in {os.path.abspath(args.failed_tests)}")
    if not os.path.isfile(args.updated_file_stats):
        raise ValueError(f"Updated file stats not found in {os.path.abspath(args.updated_file_stats)}")
    if args.contributor_stats and not os.path.isfile(args.contributor_stats):
        raise ValueError(f"Contributor stats not found in {os.path.abspath(args.contributor_stats)}")
    if args.build_dep and not os.path.isfile(args.build_dep):
        raise ValueError(f"Dependency graph file not found in {os.path.abspath(args.build_dep)}")
    if args.excluded_tests and not os.path.isfile(args.excluded_tests):
        raise ValueError(f"Excluded test list file not found in {os.path.abspath(args.excluded_tests)}")
    if args.included_tests and not os.path.isfile(args.included_tests):
        raise ValueError(f"Included test list file not found in {os.path.abspath(args.included_tests)}")

    clf = pickle.loads(Path(args.model).read_bytes())
    test_files = json.loads(Path(args.test_files).read_text())
    commits = json.loads(Path(args.commits).read_text())
    repo_commits = list(map(lambda c: github_utils.from_github_datetime(c[0]), commits))
    correlated_files = json.loads(Path(args.correlated_files).read_text())
    correlated_files_delta = json.loads(Path(args.correlated_files_delta).read_text())
    updated_file_stats = json.loads(Path(args.updated_file_stats).read_text())
    failed_tests = json.loads(Path(args.failed_tests).read_text())
    contributor_stats = json.loads(Path(args.contributor_stats).read_text()) \
        if args.contributor_stats else None
    dep_graph = json.loads(Path(args.build_dep).read_text()) \
        if args.build_dep else None
    excluded_tests = json.loads(Path(args.excluded_tests).read_text()) \
        if args.excluded_tests else []
    included_tests = json.loads(Path(args.included_tests).read_text()) \
        if args.included_tests else []

    # Merges `correlated_files` and `correlated_files_delta`
    for key, value in correlated_files_delta.items():
        correlated_files[key] = list(set(correlated_files[key] + value)) if key in correlated_files else value

    # Removes comment entries from `excluded_tests`/`included_tests`
    excluded_tests = list(filter(lambda t: not t.startswith('$comment'), excluded_tests))
    included_tests = list(filter(lambda t: not t.startswith('$comment'), included_tests))
    intersected_tests = set(excluded_tests) & set(included_tests)
    if intersected_tests:
        _logger.warning('Some tests exist in both `excluded_tests` and `included_tests`: '
                        f'{",".join(intersected_tests)}')

    # Removes the excluded tests from `test_files`
    test_files = {k: test_files[k] for k in test_files if k not in excluded_tests} \
        if excluded_tests else test_files

    # Initializes a Spark session
    spark = SparkSession.builder \
        .enableHiveSupport() \
        .getOrCreate()

    # Suppresses user warinig messages in Python
    import warnings
    warnings.simplefilter("ignore", UserWarning)

    # Suppresses `WARN` messages in JVM
    spark.sparkContext.setLogLevel("ERROR")

    try:
        import git_utils
        commit_date = git_utils.get_latest_commit_date(args.target)
        updated_files = git_utils.get_updated_files(args.target, args.num_commits)
        updated_files = ",".join(list(map(lambda f: f'"{f}"', updated_files)))  # type: ignore
        num_adds, num_dels, num_chgs = git_utils.get_updated_file_stats(args.target, args.num_commits)

        df = spark.range(1).selectExpr([
            '0 sha',
            f'"{args.username}" author',
            f'"{commit_date}" commit_date',
            f'array({updated_files}) filenames',
            f'{num_adds} num_adds',
            f'{num_dels} num_dels',
            f'{num_chgs} num_chgs'
        ])

        to_features = features.create_predict_pipeline(
            spark, test_files, repo_commits, dep_graph, correlated_files, included_tests,
            updated_file_stats, contributor_stats, failed_tests)

        predicted = _predict_failed_probs(to_features(df), clf)
        selected_test_df = predicted \
            .selectExpr(f'slice(tests, 1, {args.num_selected_tests}) selected_tests') \
            .selectExpr('selected_tests.test selected_tests')

        selected_tests = selected_test_df.collect()[0].selected_tests
        if args.format:
            print(_format_for_selective_tests_in_scalatest(selected_tests))
        else:
            print(json.dumps(selected_tests, indent=2))
    finally:
        spark.stop()


def main() -> None:
    # Checks if a training mode enabled or not first
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--train', dest='train_mode_enabled', action='store_true')
    args, rest_argv = parser.parse_known_args()

    if args.train_mode_enabled:
        train_main(rest_argv)
    else:
        predict_main(rest_argv)


if __name__ == '__main__':
    main()
