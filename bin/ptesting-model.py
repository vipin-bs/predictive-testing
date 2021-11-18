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
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from pyspark.sql import DataFrame, SparkSession, functions as funcs
from typing import Any, Dict, List, Optional, Tuple

from ptesting import github_utils, train


def _setup_logger() -> Any:
    import logging
    logfmt = '%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=logging.INFO, format=logfmt, datefmt=datefmt)
    return logging


_logger = _setup_logger()


def _create_func_for_path_diff() -> Any:
    # TODO: Removes package-depenent stuffs
    excluded_paths = set(['src', 'main', 'scala', 'target', 'scala-2.12', 'test-classes', 'test'])
    path_excluded = lambda p: p.difference(excluded_paths)

    def _func(x: str, y: str) -> int:
        return len(path_excluded(set(x.split('/')[:-1])) ^ path_excluded(set(y.split('/')[:-1])))

    return _func


def _create_func_to_transform_path_to_qualified_name() -> Any:
    # TODO: Removes package-depenent stuffs
    parse_path = re.compile(f"[a-zA-Z0-9/\-]+/(org\/apache\/spark\/[a-zA-Z0-9/\-]+)\.scala")

    def _func(path: str) -> Optional[Any]:
        return parse_path.search(path)

    return _func


def _create_func_to_enrich_authors(spark: SparkSession,
                                   contributor_stats: Optional[List[Tuple[str, str]]],
                                   input_col: str) -> Tuple[Any, List[str]]:
    if not contributor_stats:
        return lambda df: df.withColumn('num_commits', funcs.expr('0')), []

    contributor_stat_df = spark.createDataFrame(contributor_stats, schema=f'author: string, num_commits: int')

    def enrich_authors(df: DataFrame) -> DataFrame:
        return df.join(contributor_stat_df, df[input_col] == contributor_stat_df.author, 'LEFT_OUTER') \
            .na.fill({'num_commits': 0})

    return enrich_authors, [input_col]


def _to_datetime(d: str, fmt: str) -> datetime:
    return datetime.strptime(d, fmt).replace(tzinfo=timezone.utc)  # type: ignore


def _create_func_to_enrich_files(spark: SparkSession,
                                 commits: List[datetime],
                                 updated_file_stats: Dict[str, List[Tuple[str, str, str, str]]],
                                 input_commit_date: str,
                                 input_filenames: str) -> Tuple[Any, List[str]]:
    broadcasted_updated_file_stats = spark.sparkContext.broadcast(updated_file_stats)
    broadcasted_commits = spark.sparkContext.broadcast(commits)

    def enrich_files(df: DataFrame) -> DataFrame:
        @funcs.pandas_udf("string")  # type: ignore
        def _enrich_files(dates: pd.Series, filenames: pd.Series) -> pd.Series:
            updated_file_stats = broadcasted_updated_file_stats.value
            commits = broadcasted_commits.value
            ret = []
            for commit_date, files in zip(dates, filenames):
                base_date = _to_datetime(commit_date, '%Y/%m/%d %H:%M:%S')
                is_updated_in_days = lambda interval, date: \
                    base_date - timedelta(interval) <= date and base_date >= date

                def is_updated_in_commits(num_commits: int, date: Any) -> bool:
                    cur_pos = 0
                    while cur_pos < len(commits):
                        if commits[cur_pos] <= base_date:
                            target_pos = cur_pos + min([num_commits, len(commits) - cur_pos - 1])
                            return commits[target_pos] <= date and commits[cur_pos] >= date

                        cur_pos += 1

                    return False

                # Time-dependent features
                updated_num_3d = 0
                updated_num_14d = 0
                updated_num_56d = 0

                # Commit-dependent features
                updated_num_3c = 0
                updated_num_14c = 0
                updated_num_56c = 0

                for file in json.loads(files):
                    if file in updated_file_stats:
                        for update_date, _, _, _ in updated_file_stats[file]:
                            update_date = github_utils.from_github_datetime(update_date)
                            if is_updated_in_days(3, update_date):
                                updated_num_3d += 1
                            if is_updated_in_days(14, update_date):
                                updated_num_14d += 1
                            if is_updated_in_days(56, update_date):
                                updated_num_56d += 1
                            if is_updated_in_commits(3, update_date):
                                updated_num_3c += 1
                            if is_updated_in_commits(14, update_date):
                                updated_num_14c += 1
                            if is_updated_in_commits(56, update_date):
                                updated_num_56c += 1

                ret.append(json.dumps({
                    'n3d': updated_num_3d,
                    'n14d': updated_num_14d,
                    'n56d': updated_num_56d,
                    'n3c': updated_num_3c,
                    'n14c': updated_num_14c,
                    'n56c': updated_num_56c
                }))

            return pd.Series(ret)

        enrich_files_expr = _enrich_files(funcs.expr(input_commit_date), funcs.expr(f'to_json({input_filenames})'))
        enrich_files_schema = 'struct<n3d: int, n14d: int, n56d: int, n3c: int, n14c: int, n56c: int>'
        return df.withColumn('updated_file_stats', enrich_files_expr)  \
            .withColumn('ufs', funcs.expr(f'from_json(updated_file_stats, "{enrich_files_schema}")')) \
            .withColumn('updated_num_3d', funcs.expr('ufs.n3d')) \
            .withColumn('updated_num_14d', funcs.expr('ufs.n14d')) \
            .withColumn('updated_num_56d', funcs.expr('ufs.n56d')) \
            .withColumn('updated_num_3c', funcs.expr('ufs.n3c')) \
            .withColumn('updated_num_14c', funcs.expr('ufs.n14c')) \
            .withColumn('updated_num_56c', funcs.expr('ufs.n56c'))

    return enrich_files, [input_commit_date, input_filenames]


def _create_func_to_enumerate_related_tests(spark: SparkSession,
                                            dep_graph: Dict[str, List[str]],
                                            corr_map: Dict[str, List[str]],
                                            test_files: Dict[str, str],
                                            input_files: str,
                                            depth: int,
                                            max_num_tests: Optional[int] = None) -> Tuple[Any, List[str]]:
    broadcasted_dep_graph = spark.sparkContext.broadcast(dep_graph)
    broadcasted_corr_map = spark.sparkContext.broadcast(corr_map)
    broadcasted_test_files = spark.sparkContext.broadcast(test_files)

    def enumerate_related_tests(df: DataFrame) -> DataFrame:
        @funcs.pandas_udf("string")  # type: ignore
        def _enumerate_tests(file_paths: pd.Series) -> pd.Series:
            parse_path = _create_func_to_transform_path_to_qualified_name()
            path_diff = _create_func_for_path_diff()

            dep_graph = broadcasted_dep_graph.value
            corr_map = broadcasted_corr_map.value
            test_files = broadcasted_test_files.value

            def _enumerate_tests_from_dep_graph(target):  # type: ignore
                subgraph = {}
                visited_nodes = set()
                keys = list([target])
                for i in range(0, depth):
                    if len(keys) == 0:
                        break

                    next_keys = set()
                    for key in keys:
                        if key in dep_graph and key not in visited_nodes:
                            nodes = dep_graph[key]
                            next_keys.update(nodes)
                    visited_nodes.update(keys)
                    keys = list(next_keys)
                if keys is not None:
                    visited_nodes.update(keys)
                tests = list(filter(lambda n: n.endswith('Suite'), visited_nodes))
                return tests

            ret = []
            for file_path in file_paths:
                correlated_files = corr_map[file_path] if file_path in corr_map else []
                correlated_tests = list(filter(lambda f: f in test_files, correlated_files))
                if not file_path:
                    ret.append(json.dumps({'tests': correlated_tests}))
                else:
                    result = parse_path(file_path)
                    if not result:
                        ret.append(json.dumps({'tests': correlated_tests}))
                    else:
                        dependant_tests = _enumerate_tests_from_dep_graph(result.group(1).replace('/', '.'))
                        related_tests = dependant_tests + correlated_tests
                        ret.append(json.dumps({'tests': related_tests}))

            return pd.Series(ret)

        related_test_df = df.selectExpr('sha', f'explode_outer({input_files}) filename') \
            .withColumn('tests', _enumerate_tests(funcs.expr('filename'))) \
            .selectExpr('sha', 'from_json(tests, "tests ARRAY<STRING>").tests tests') \
            .selectExpr('sha', 'explode_outer(tests) test') \
            .groupBy('sha') \
            .agg(funcs.expr(f'collect_set(test) tests')) \
            .selectExpr('sha', 'size(tests) target_card', f'tests related_tests') \
            .where(f'related_tests IS NOT NULL')

        return df.join(related_test_df, 'sha', 'LEFT_OUTER')

    return enumerate_related_tests, ['sha', 'files']


def _create_func_to_enumerate_all_tests(spark: SparkSession, test_files: Dict[str, str]) -> Tuple[Any, List[str]]:
    all_test_df = spark.createDataFrame(list(test_files.items()), ['test', 'path'])

    def enumerate_all_tests(df: DataFrame) -> DataFrame:
        return df.join(all_test_df.selectExpr(f'collect_set(test) all_tests')) \
            .withColumn('target_card', funcs.expr(f'size(all_tests)'))

    return enumerate_all_tests, []


def _build_failed_tests(train_df: DataFrame) -> Dict[str, List[str]]:
    df = train_df.where('size(failed_tests) > 0') \
        .selectExpr('commit_date', 'explode(failed_tests) failed_test') \
        .groupBy('failed_test') \
        .agg(funcs.expr('collect_set(commit_date)').alias('commit_dates'))

    failed_tests: List[Tuple[str, List[str]]] = []
    for row in df.collect():
        commit_dates = sorted(row.commit_dates, key=lambda d: _to_datetime(d, '%Y/%m/%d %H:%M:%S'), reverse=True)
        failed_tests.append((row.failed_test, commit_dates))

    return dict(failed_tests)


def _create_func_to_enrich_tests(spark: SparkSession,
                                 commits: List[datetime],
                                 failed_tests: Dict[str, List[str]],
                                 input_commit_date: str,
                                 input_test: str) -> Tuple[Any, List[str]]:
    broadcasted_failed_tests = spark.sparkContext.broadcast(failed_tests)
    broadcasted_commits = spark.sparkContext.broadcast(commits)

    def enrich_tests(df: DataFrame) -> DataFrame:
        @funcs.pandas_udf("string")  # type: ignore
        def _enrich_tests(dates: pd.Series, tests: pd.Series) -> pd.Series:
            failed_tests = broadcasted_failed_tests.value
            commits = broadcasted_commits.value
            ret = []
            for commit_date, test in zip(dates, tests):
                base_date = _to_datetime(commit_date, '%Y/%m/%d %H:%M:%S')
                failed_in_days = lambda interval, date: \
                    base_date - timedelta(interval) <= date and base_date >= date

                def failed_in_commits(num_commits: int, date: Any) -> bool:
                    cur_pos = 0
                    while cur_pos < len(commits):
                        if commits[cur_pos] <= base_date:
                            target_pos = cur_pos + min([num_commits, len(commits) - cur_pos - 1])
                            return commits[target_pos] <= date and commits[cur_pos] >= date

                        cur_pos += 1

                    return False

                # Time-dependent features
                failed_num_7d = 0
                failed_num_14d = 0
                failed_num_28d = 0

                # Commit-dependent features
                failed_num_7c = 0
                failed_num_14c = 0
                failed_num_28c = 0

                total_failed_num = 0

                if test in failed_tests:
                    for failed_date in failed_tests[test]:
                        failed_date = _to_datetime(failed_date, '%Y/%m/%d %H:%M:%S')  # type: ignore
                        if failed_in_days(7, failed_date):
                            failed_num_7d += 1
                        if failed_in_days(14, failed_date):
                            failed_num_14d += 1
                        if failed_in_days(28, failed_date):
                            failed_num_28d += 1
                        if failed_in_commits(7, failed_date):
                            failed_num_7c += 1
                        if failed_in_commits(14, failed_date):
                            failed_num_14c += 1
                        if failed_in_commits(28, failed_date):
                            failed_num_28c += 1

                        total_failed_num += 1

                ret.append(json.dumps({
                    'n7d': failed_num_7d,
                    'n14d': failed_num_14d,
                    'n28d': failed_num_28d,
                    'n7c': failed_num_7c,
                    'n14c': failed_num_14c,
                    'n28c': failed_num_28c,
                    'total': total_failed_num
                }))

            return pd.Series(ret)

        enrich_tests_expr = _enrich_tests(funcs.expr(input_commit_date), funcs.expr(input_test))
        failed_test_stats_schema = 'struct<n7d: int, n14d: int, n28d: int, n7c: int, n14c: int, n28c: int, total: int>'
        return df.withColumn('failed_test_stats', enrich_tests_expr)  \
            .withColumn('fts', funcs.expr(f'from_json(failed_test_stats, "{failed_test_stats_schema}")')) \
            .withColumn('failed_num_7d', funcs.expr('fts.n7d')) \
            .withColumn('failed_num_14d', funcs.expr('fts.n14d')) \
            .withColumn('failed_num_28d', funcs.expr('fts.n28d')) \
            .withColumn('failed_num_7c', funcs.expr('fts.n7c')) \
            .withColumn('failed_num_14c', funcs.expr('fts.n14c')) \
            .withColumn('failed_num_28c', funcs.expr('fts.n28c')) \
            .withColumn('total_failed_num', funcs.expr('fts.total'))

    return enrich_tests, [input_commit_date, input_test]


def _create_func_to_compute_distances(spark: SparkSession,
                                      dep_graph: Dict[str, List[str]], test_files: Dict[str, str],
                                      input_files: str,
                                      input_test: str) -> Tuple[Any, List[str]]:
    broadcasted_dep_graph = spark.sparkContext.broadcast(dep_graph)
    broadcasted_test_files = spark.sparkContext.broadcast(test_files)

    @funcs.pandas_udf("int")  # type: ignore
    def _path_diff(filenames: pd.Series, test: pd.Series) -> pd.Series:
        path_diff = _create_func_for_path_diff()

        test_files = broadcasted_test_files.value

        ret = []
        for names, t in zip(filenames, test):
            if t in test_files:
                distances = []
                for n in json.loads(names):
                    distances.append(path_diff(n, test_files[t]))
                ret.append(min(distances))
            else:
                ret.append(128)

        return pd.Series(ret)

    @funcs.pandas_udf("int")  # type: ignore
    def _distance(filenames: pd.Series, test: pd.Series) -> pd.Series:
        parse_path = _create_func_to_transform_path_to_qualified_name()

        dep_graph = broadcasted_dep_graph.value

        ret = []
        for names, t in zip(filenames, test):
            distances = [128]
            for n in json.loads(names):
                result = parse_path(n)
                if result:
                    ident = result.group(1).replace('/', '.')
                    if ident == t:
                        distances.append(0)
                        break

                    visited_nodes = set()
                    keys = list([ident])
                    for i in range(0, 16):
                        if len(keys) == 0:
                            break

                        next_keys = set()
                        for key in keys:
                            if key in dep_graph and key not in visited_nodes:
                                nodes = dep_graph[key]
                                next_keys.update(nodes)
                        if t in next_keys:
                            distances.append(i + 1)
                            break

                        visited_nodes.update(keys)
                        keys = list(next_keys)

            ret.append(min(distances))

        return pd.Series(ret)

    def compute_distances(df: DataFrame) -> DataFrame:
        path_diff_udf = _path_diff(funcs.expr(f'to_json({input_files})'), funcs.expr(input_test))
        return df.withColumn('path_difference', path_diff_udf) \
            .withColumn('distance', _distance(funcs.expr(f'to_json({input_files})'), funcs.expr(input_test)))

    return compute_distances, [input_files, input_test]


def _create_func_to_compute_file_cardinality(input_col: str) -> Tuple[Any, List[str]]:
    def compute_file_cardinality(df: DataFrame) -> DataFrame:
        return df.withColumn('file_card', funcs.expr(f'size({input_col})'))

    return compute_file_cardinality, [input_col]


def _create_func_to_compute_interaction_features(input_cols: List[Tuple[str, str]]) -> Tuple[Any, List[str]]:
    interact_fts_exprs = list(map(lambda p: f'`{p[0]}` * `{p[1]}` AS `{p[0]}__x__{p[1]}`', input_cols))

    def compute_interaction_features(df: DataFrame) -> DataFrame:
        return df.selectExpr('*', *interact_fts_exprs)

    import itertools
    _input_cols = list(set((itertools.chain.from_iterable(input_cols))))
    return compute_interaction_features, _input_cols


def _create_func_to_expand_updated_stats() -> Tuple[Any, List[str]]:
    def expand_updated_stats(df: DataFrame) -> DataFrame:
        sum_expr = lambda c: funcs.expr(f'aggregate({c}, 0, (x, y) -> int(x) + int(y))')
        return df.withColumn('num_adds', sum_expr('files.file.additions')) \
            .withColumn('num_dels', sum_expr("files.file.deletions")) \
            .withColumn('num_chgs', sum_expr("files.file.changes"))

    return expand_updated_stats, ['files']


def _create_func_to_add_failed_column() -> Tuple[Any, List[str]]:
    def add_failed_column(df: DataFrame) -> DataFrame:
        passed_test_df = df \
            .selectExpr('*', 'array_except(related_tests, failed_tests) tests', '0 failed') \
            .where('size(tests) > 0')
        failed_test_df = df \
            .where('size(failed_tests) > 0') \
            .selectExpr('*', 'failed_tests tests', '1 failed')
        return passed_test_df.union(failed_test_df) \
            .selectExpr('*', 'explode(tests) test')

    return add_failed_column, ['related_tests', 'failed_tests']


# Our predictive model uses LightGBM, an implementation of gradient-boosted decision trees.
# This is because the algorithm has desirable properties for this use-case
# (the reason is the same with the Facebook one):
#  - normalizing feature values are less required
#  - fast model training on commodity hardware
#  - robustness in imbalanced datasets
def _build_predictive_model(df: DataFrame, to_features: Any) -> Any:
    pdf = to_features(df).toPandas()
    X = pdf[pdf.columns[pdf.columns != 'failed']]  # type: ignore
    y = pdf['failed']
    X, y = train.rebalance_training_data(X, y, coeff=1.0)
    clf, score = train.build_model(X, y, opts={'hp.timeout': '3600', 'hp.no_progress_loss': '1'})
    _logger.info(f"model score: {score}")
    return clf


def _train_test_split(df: DataFrame, test_ratio: float) -> Tuple[DataFrame, DataFrame]:
    test_nrows = int(df.count() * test_ratio)
    test_df = df.orderBy(funcs.expr('to_timestamp(commit_date, "yyy/MM/dd HH:mm:ss")').desc()).limit(test_nrows)
    train_df = df.subtract(test_df)
    return train_df, test_df


def _predict_failed_probs_for_tests(test_df: DataFrame, clf: Any, to_features: Any) -> DataFrame:
    test_feature_df = to_features(test_df)
    pdf = test_feature_df.selectExpr('sha', 'test').toPandas()
    X = test_feature_df.drop('sha', 'failed', 'test').toPandas()
    predicted = clf.predict_proba(X)
    pmf = map(lambda p: {"classes": clf.classes_.tolist(), "probs": p.tolist()}, predicted)
    pmf = map(lambda p: json.dumps(p), pmf)  # type: ignore
    pdf['predicted'] = pd.Series(list(pmf))
    to_map_expr = funcs.expr('from_json(predicted, "classes array<string>, probs array<double>")')
    to_failed_prob = 'map_from_arrays(pmf.classes, pmf.probs)["1"] failed_prob'
    df_with_failed_probs = test_df.sql_ctx.sparkSession.createDataFrame(pdf) \
        .withColumn('pmf', to_map_expr) \
        .selectExpr('sha', 'test', to_failed_prob)

    # Applied an emprical rule: sets 1.0 to the failed prob. of the modified tests
    regex = '"\/(org\/apache\/spark\/[a-zA-Z0-9/\-]+Suite)\.scala$"'
    replace_test = f'f -> replace(regexp_extract(f, {regex}, 1), "/", ".")'
    extract_test = f'transform(files.file.name, {replace_test})'
    modified_test_df = test_df.selectExpr('sha', f'filter({extract_test}, f -> length(f) > 0) modified_tests')
    corrected_failed_prob = 'case when array_contains(modified_tests, test) then 1.0 else failed_prob end failed_prob'
    df_with_failed_probs = df_with_failed_probs.join(modified_test_df, 'sha', 'INNER') \
        .selectExpr('sha', 'test', corrected_failed_prob)

    compare = lambda x, y: \
        f"case when {x}.failed_prob < {y}.failed_prob then 1 " \
        f"when {x}.failed_prob > {y}.failed_prob then -1 " \
        "else 0 end"
    df_with_failed_probs = df_with_failed_probs.groupBy('sha') \
        .agg(funcs.expr('collect_set(named_struct("test", test, "failed_prob", failed_prob))').alias('tests')) \
        .selectExpr('sha', f'filter(tests, t -> t.test IS NOT NULL) tests') \
        .selectExpr('sha', f'array_sort(tests, (l, r) -> {compare("l", "r")}) tests')

    return df_with_failed_probs


def _predict_failed_probs(df: DataFrame, clf: Any) -> DataFrame:
    pdf = df.toPandas()
    predicted = clf.predict_proba(pdf.drop('test', axis=1))
    pmf = map(lambda p: {"classes": clf.classes_.tolist(), "probs": p.tolist()}, predicted)
    pmf = map(lambda p: json.dumps(p), pmf)  # type: ignore
    pdf['predicted'] = pd.Series(list(pmf))
    to_map_expr = funcs.expr('from_json(predicted, "classes array<string>, probs array<double>")')
    to_failed_prob = 'map_from_arrays(pmf.classes, pmf.probs)["1"] failed_prob'
    df_with_failed_probs = df.sql_ctx.sparkSession.createDataFrame(pdf[['test', 'predicted']]) \
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


def _compute_eval_stats(df: DataFrame) -> DataFrame:
    rank = 'array_position(transform(tests, x -> x.test), failed_test) rank'
    to_test_map = 'map_from_arrays(transform(tests, x -> x.test), transform(tests, x -> x.failed_prob)) tests'
    return df.selectExpr('sha', 'explode(failed_tests) failed_test', 'tests') \
        .selectExpr('sha', 'failed_test', rank, 'tests[0].failed_prob max_score', to_test_map) \
        .selectExpr('sha', 'failed_test', 'rank', 'tests[failed_test] score', 'max_score')


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


def _create_pipelines(name: str, funcs: List[Tuple[Any, List[str]]]) -> Any:
    def _columns_added(src: DataFrame, dst: DataFrame) -> Any:
        return list(set(dst.columns).difference(set(src.columns)))

    def _columns_removed(src: DataFrame, dst: DataFrame) -> Any:
        return list(set(src.columns).difference(set(dst.columns)))

    def _func(df: DataFrame) -> DataFrame:
        pipeline_input_columns = set(df.columns)
        for f, input_cols in funcs:
            assert type(df) is DataFrame
            transformed_df = f(df)
            pipeline_input_columns = pipeline_input_columns.difference(set(input_cols))
            _logger.info(f"{name}: {f.__name__}({','.join(input_cols)}) -> "
                         f"[{','.join(_columns_added(df, transformed_df))}] added, "
                         f"[{','.join(_columns_removed(df, transformed_df))}] removed")
            df = transformed_df

        if pipeline_input_columns:
            _logger.warning(f"{name}: expected unused features are [{','.join(pipeline_input_columns)}]")

        return df

    return _func


def _create_train_test_pipeline(spark: SparkSession,
                                test_files: Dict[str, str],
                                commits: List[datetime],
                                dep_graph: Dict[str, List[str]],
                                corr_map: Dict[str, List[str]],
                                updated_file_stats: Dict[str, List[Tuple[str, str, str, str]]],
                                contributor_stats: Optional[List[Tuple[str, str]]],
                                failed_tests: Dict[str, List[str]]) -> Any:
    # This pipeline extracts features from a dataset of historical test outcomes.
    # The current features used in our model are as follows:
    #  - Change history for files: the count of commits made to modified files in the last 3, 14, and 56 days
    #    (`updated_num_3d`, `updated_num_14d`, and `updated_num_15d`, respectively).
    #  - File update statistics: the total number of additions, deletions, and changes made to modified files
    #    (`num_adds`, `num_dels`, and `num_chgs`, respectively).
    #  - File cardinality: the number of files touched in a test run (`file_card`).
    #  - Historical failure rates: the count of target test failures occurred in the last 7, 14, and 28 days
    #    (`failed_num_7d`, `failed_num_14d`, and `failed_num_28d`, respectively) and the total count
    #    of target test failures in historical test results (`total_failed_num`).
    #  - Minimal path difference between modified files and a target test file: the number of different directories
    #    between file paths (`path_difference`). Let's say that we have two files: their file paths are
    #    'xxx/yyy/zzz/file' and 'xxx/aaa/zzz/test_file'. In this example, the number of a path difference is 1.
    #  - Shortest distance between modified files and a target test file: shortest path distance
    #    in a call graph (`distance`) and the graph will be described soon after.
    #  - Interacted features: the multiplied values of some feature pairs, e.g.,
    #    `total_failed_num * num_commits` and `failed_num_7d * num_commits`.
    #
    # NOTE: The Facebook paper [2] reports that the best performance predictive model uses a change history,
    # failure rates, target cardinality, and minimal distances (For more details, see
    # "Section 6.B. Feature Selection" in the paper [2]). Even in our model, change history for files
    # and historical failure rates tend to be more important than the other features.
    #
    # TODO: Needs to improve predictive model performance by checking the other feature candidates
    # that can be found in the Facebook paper (See "Section 4.A. Feature Engineering") [2]
    # and the Google paper (See "Section 4. Hypotheses, Models and Results") [1].
    expected_train_features = [
        'num_commits',
        'updated_num_3d',
        'updated_num_14d',
        'updated_num_56d',
        'updated_num_3c',
        'updated_num_14c',
        'updated_num_56c',
        'num_adds',
        'num_dels',
        'num_chgs',
        # 'target_card',
        'file_card',
        'failed_num_7d',
        'failed_num_14d',
        'failed_num_28d',
        'failed_num_7c',
        'failed_num_14c',
        'failed_num_28c',
        'total_failed_num',
        'path_difference',
        'distance',
        'total_failed_num__x__num_commits',
        'total_failed_num__x__num_chgs',
        'total_failed_num__x__updated_num_56d',
        'total_failed_num__x__updated_num_56c',
        'total_failed_num__x__path_difference',
        'total_failed_num__x__distance',
        'failed_num_7d__x__num_commits',
        'failed_num_7d__x__num_chgs',
        'failed_num_7d__x__updated_num_56d',
        'failed_num_7d__x__updated_num_56c',
        'failed_num_7d__x__path_difference',
        'failed_num_7d__x__distance',
        'distance__x__num_commits',
        'distance__x__num_chgs',
        'distance__x__updated_num_56d',
        'distance__x__updated_num_56c',
        'distance__x__path_difference'
    ]

    enrich_authors = _create_func_to_enrich_authors(spark, contributor_stats, input_col='author')
    enrich_files = _create_func_to_enrich_files(spark, commits, updated_file_stats,
                                                input_commit_date='commit_date',
                                                input_filenames='files.file.name')
    enumerate_related_tests = _create_func_to_enumerate_related_tests(spark, dep_graph, corr_map, test_files,
                                                                      input_files='files.file.name',
                                                                      depth=2, max_num_tests=16)
    enrich_tests = _create_func_to_enrich_tests(spark, commits, failed_tests,
                                                input_commit_date='commit_date',
                                                input_test='test')
    compute_distances = _create_func_to_compute_distances(spark, dep_graph, test_files,
                                                          input_files='files.file.name', input_test='test')
    compute_file_cardinality = _create_func_to_compute_file_cardinality(input_col='files')
    interacted_features = [
        ('total_failed_num', 'num_commits'),
        ('total_failed_num', 'num_chgs'),
        ('total_failed_num', 'updated_num_56d'),
        ('total_failed_num', 'updated_num_56c'),
        ('total_failed_num', 'path_difference'),
        ('total_failed_num', 'distance'),
        ('failed_num_7d', 'num_commits'),
        ('failed_num_7d', 'num_chgs'),
        ('failed_num_7d', 'updated_num_56d'),
        ('failed_num_7d', 'updated_num_56c'),
        ('failed_num_7d', 'path_difference'),
        ('failed_num_7d', 'distance'),
        ('distance', 'num_commits'),
        ('distance', 'num_chgs'),
        ('distance', 'updated_num_56d'),
        ('distance', 'updated_num_56c'),
        ('distance', 'path_difference')
    ]
    compute_interaction_features = _create_func_to_compute_interaction_features(input_cols=interacted_features)
    expand_updated_stats = _create_func_to_expand_updated_stats()
    add_failed_column = _create_func_to_add_failed_column()
    select_train_features = lambda df: df.selectExpr(['failed', *expected_train_features]), \
        ['failed', *expected_train_features]

    to_train_features = _create_pipelines(
        'to_train_features', [
            enrich_authors,
            enrich_files,
            expand_updated_stats,
            enumerate_related_tests,
            add_failed_column,
            enrich_tests,
            compute_distances,
            compute_file_cardinality,
            compute_interaction_features,
            select_train_features
        ])

    expected_test_features = ['sha', 'test', *expected_train_features]
    explode_tests = lambda df: df.selectExpr('*', 'explode_outer(related_tests) test'), ['related_tests']
    select_test_features = lambda df: df.selectExpr(expected_test_features), expected_test_features

    to_test_features = _create_pipelines(
        'to_test_features', [
            enrich_authors,
            enrich_files,
            expand_updated_stats,
            enumerate_related_tests,
            explode_tests,
            enrich_tests,
            compute_distances,
            compute_file_cardinality,
            compute_interaction_features,
            select_test_features
        ])

    return to_train_features, to_test_features


def _build_corr_map(commits: List[Tuple[str, str, List[str]]], train_df: DataFrame) -> Dict[str, List[str]]:
    import itertools
    parse_path = re.compile(f"[a-zA-Z0-9/\-]+/(org\/apache\/spark\/.+\/)([a-zA-Z0-9\-]+)\.scala")
    parse_scala_file = re.compile("class\s+([a-zA-Z0-9]+Suite)\s+extends\s+")
    corr_map: Dict[str, Any] = {}
    for _, _, files in commits:
        group = []
        for f in files:
            qs = parse_path.search(f)
            if qs:
                package = qs.group(1).replace('/', '.')
                if 'SPARK_REPO' in os.environ:
                    try:
                        file_as_string = Path(f'{os.environ["SPARK_REPO"]}/{f}').read_text()
                        classes = parse_scala_file.findall(file_as_string)
                        if classes:
                            group.append((f, list(map(lambda c: f'{package}{c}', classes))))
                        else:
                            clazz = qs.group(2)
                            group.append((f, [f'{package}{clazz}']))
                    except:
                        pass
                else:
                    clazz = qs.group(2)
                    group.append((f, [f'{package}{clazz}']))
            else:
                group.append((f, []))

        # for x, y in filter(lambda p: p[0] != p[1], itertools.product(group, group)):
        for (path1, classes1), (path2, classes2) in itertools.product(group, group):
            if path1 not in corr_map:
                corr_map[path1] = set()

            corr_map[path1].update(classes1 + classes2)

    failed_maps = train_df.where('size(failed_tests) > 0') \
        .selectExpr('failed_tests', 'files.file.name').toPandas().to_dict(orient='records')
    for failed_map in failed_maps:
        for c in failed_map['failed_tests']:
            for f in failed_map['name']:
                if f not in corr_map:
                    corr_map[f] = set()

                corr_map[f].add(c)

    for k, v in corr_map.items():
        corr_map[k] = list(v)

    return corr_map


def _train_and_eval_ptest_model(output_path: str, spark: SparkSession, df: DataFrame,
                                test_files: Dict[str, str],
                                commits: List[Tuple[str, str, List[str]]],
                                dep_graph: Dict[str, List[str]],
                                updated_file_stats: Dict[str, List[Tuple[str, str, str, str]]],
                                contributor_stats: Optional[List[Tuple[str, str]]],
                                test_ratio: float = 0.20) -> None:
    def num_failed_tests(df: DataFrame) -> int:
        return df.selectExpr('explode(failed_tests)').count()

    train_df, test_df = _train_test_split(df, test_ratio=test_ratio)
    _logger.info('Split data: #total={}(#failed={}), #train={}(#failed={}), #test={}(#failed={})'.format(
        df.count(), num_failed_tests(df), train_df.count(), num_failed_tests(train_df),
        test_df.count(), num_failed_tests(test_df)))

    corr_map = _build_corr_map(commits, train_df)
    repo_commits = list(map(lambda c: github_utils.from_github_datetime(c[0]), commits))
    failed_tests = _build_failed_tests(train_df)
    to_train_features, to_test_features = \
        _create_train_test_pipeline(spark, test_files, repo_commits, dep_graph, corr_map, updated_file_stats,
                                    contributor_stats, failed_tests)
    clf = _build_predictive_model(train_df, to_train_features)

    with open(f"{output_path}/correlated-map.json", 'w') as f:
        f.write(json.dumps(corr_map, indent=2))
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
    parser.add_argument('--updated-file-stats', type=str, required=True)
    parser.add_argument('--contributor-stats', type=str, required=False)
    # TODO: Makes `--build-dep` optional
    parser.add_argument('--build-dep', type=str, required=True)
    parser.add_argument('--excluded-tests', type=str, required=False)
    args = parser.parse_args(argv)

    if not os.path.exists(args.output) or not os.path.isdir(args.output):
        raise ValueError(f"Output directory not found in {os.path.abspath(args.output)}")
    if not os.path.exists(args.train_log_data):
        raise ValueError(f"Training data not found in {os.path.abspath(args.train_log_data)}")
    if not os.path.exists(args.test_files):
        raise ValueError(f"Test list file not found in {os.path.abspath(args.test_files)}")
    if not os.path.exists(args.commits):
        raise ValueError(f"Commit history file not found in {os.path.abspath(args.commits)}")
    if not os.path.exists(args.updated_file_stats):
        raise ValueError(f"Updated file stats not found in {os.path.abspath(args.updated_file_stats)}")
    if args.contributor_stats and not os.path.exists(args.contributor_stats):
        raise ValueError(f"Contributor stats not found in {os.path.abspath(args.contributor_stats)}")
    if args.build_dep and not os.path.exists(args.build_dep):
        raise ValueError(f"Dependency graph file not found in {os.path.abspath(args.build_dep)}")
    if args.excluded_tests and not os.path.exists(args.excluded_tests):
        raise ValueError(f"Excluded test list file not found in {os.path.abspath(args.excluded_tests)}")

    test_files = json.loads(Path(args.test_files).read_text())
    commits = json.loads(Path(args.commits).read_text())
    updated_file_stats = json.loads(Path(args.updated_file_stats).read_text())
    contributor_stats = json.loads(Path(args.contributor_stats).read_text()) \
        if args.contributor_stats else None
    dep_graph = json.loads(Path(args.build_dep).read_text()) \
        if args.build_dep else None
    excluded_tests = json.loads(Path(args.excluded_tests).read_text()) \
        if args.excluded_tests else []

    # Removes comment entries from `excluded_tests`
    excluded_tests = list(filter(lambda t: not t.startswith('$comment'), excluded_tests))

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

        _train_and_eval_ptest_model(args.output, spark, log_data_df, test_files, commits, dep_graph,
                                    updated_file_stats, contributor_stats,
                                    test_ratio=0.10)
    finally:
        spark.stop()


def _exec_subprocess(cmd: str, raise_error: bool = True) -> Tuple[Any, Any, Any]:
    import subprocess
    child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = child.communicate()
    rt = child.returncode
    if rt != 0 and raise_error:
        raise RuntimeError(f"command return code is not 0. got {rt}. stderr = {stderr}")  # type: ignore

    return stdout, stderr, rt


def _get_updated_files(target: str, num_commits: int) -> List[str]:
    stdout, _, _ = _exec_subprocess(f'git -C {target} diff --name-only HEAD~{num_commits}')
    return list(filter(lambda f: f, stdout.decode().split('\n')))


def _get_updated_file_stats(target: str, num_commits: int) -> Tuple[int, int, int]:
    stdout, _, _ = _exec_subprocess(f'git -C {target} diff --shortstat HEAD~{num_commits}')
    stat_summary = stdout.decode()
    m = re.search('\d+ files changed, (\d+) insertions\(\+\), (\d+) deletions\(\-\)', stdout.decode())
    num_adds = int(m.group(1))  # type: ignore
    num_dels = int(m.group(2))  # type: ignore
    return num_adds, num_dels, num_adds + num_dels


def _get_latest_commit_date(target: str) -> str:
    stdout, _, _ = _exec_subprocess(f'git -C {target} log -1 --format=%cd --date=iso')
    import dateutil.parser  # type: ignore
    commit_date = dateutil.parser.parse(stdout.decode())
    return commit_date.utcnow().strftime('%Y/%m/%d %H:%M:%S')


def _format_for_scalatest(tests: List[str]) -> str:
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
    parser.add_argument('--correlated-map', type=str, required=True)
    parser.add_argument('--failed-tests', type=str, required=True)
    parser.add_argument('--updated-file-stats', type=str, required=True)
    parser.add_argument('--contributor-stats', type=str, required=False)
    # TODO: Makes `--build-dep` optional
    parser.add_argument('--build-dep', type=str, required=True)
    parser.add_argument('--excluded-tests', type=str, required=False)
    parser.add_argument('--format', dest='format', action='store_true')
    args = parser.parse_args(argv)

    if not os.path.isdir(f'{args.target}/.git'):
        raise ValueError(f"Git-managed directory not found in {os.path.abspath(args.target)}")
    if args.num_commits <= 0:
        raise ValueError(f"Target #commits must be positive, but {args.num_commits}")
    if args.num_selected_tests <= 0:
        raise ValueError(f"Predicted #tests must be positive, but {args.num_selected_tests}")
    if not os.path.exists(args.model):
        raise ValueError(f"Predictive model not found in {os.path.abspath(args.model)}")
    if not os.path.exists(args.test_files):
        raise ValueError(f"Test list file not found in {os.path.abspath(args.test_files)}")
    if not os.path.exists(args.commits):
        raise ValueError(f"Commit history file not found in {os.path.abspath(args.commits)}")
    if not os.path.exists(args.correlated_map):
        raise ValueError(f"Correlated file map not found in {os.path.abspath(args.correlated_map)}")
    if not os.path.exists(args.failed_tests):
        raise ValueError(f"Failed test list file not found in {os.path.abspath(args.failed_tests)}")
    if not os.path.exists(args.updated_file_stats):
        raise ValueError(f"Updated file stats not found in {os.path.abspath(args.updated_file_stats)}")
    if args.contributor_stats and not os.path.exists(args.contributor_stats):
        raise ValueError(f"Contributor stats not found in {os.path.abspath(args.contributor_stats)}")
    if args.build_dep and not os.path.exists(args.build_dep):
        raise ValueError(f"Dependency graph file not found in {os.path.abspath(args.build_dep)}")
    if args.excluded_tests and not os.path.exists(args.excluded_tests):
        raise ValueError(f"Excluded test list file not found in {os.path.abspath(args.excluded_tests)}")

    clf = pickle.loads(Path(args.model).read_bytes())
    test_files = json.loads(Path(args.test_files).read_text())
    commits = json.loads(Path(args.commits).read_text())
    repo_commits = list(map(lambda c: github_utils.from_github_datetime(c[0]), commits))
    corr_map = json.loads(Path(args.correlated_map).read_text())
    updated_file_stats = json.loads(Path(args.updated_file_stats).read_text())
    failed_tests = json.loads(Path(args.failed_tests).read_text())
    contributor_stats = json.loads(Path(args.contributor_stats).read_text()) \
        if args.contributor_stats else None
    dep_graph = json.loads(Path(args.build_dep).read_text()) \
        if args.build_dep else None
    excluded_tests = json.loads(Path(args.excluded_tests).read_text()) \
        if args.excluded_tests else []

    # Removes comment entries from `excluded_tests`
    excluded_tests = list(filter(lambda t: not t.startswith('$comment'), excluded_tests))

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
        commit_date = _get_latest_commit_date(args.target)
        updated_files = ",".join(list(map(lambda f: f'"{f}"', _get_updated_files(args.target, args.num_commits))))
        num_adds, num_dels, num_chgs = _get_updated_file_stats(args.target, args.num_commits)

        df = spark.range(1).selectExpr([
            '0 sha',
            f'"{args.username}" author',
            f'"{commit_date}" commit_date',
            f'array({updated_files}) filenames',
            f'{num_adds} num_adds',
            f'{num_dels} num_dels',
            f'{num_chgs} num_chgs'
        ])

        expected_features = [
            'test',
            'num_commits',
            'updated_num_3d',
            'updated_num_14d',
            'updated_num_56d',
            'updated_num_3c',
            'updated_num_14c',
            'updated_num_56c',
            'num_adds',
            'num_dels',
            'num_chgs',
            # 'target_card',
            'file_card',
            'failed_num_7d',
            'failed_num_14d',
            'failed_num_28d',
            'failed_num_7c',
            'failed_num_14c',
            'failed_num_28c',
            'total_failed_num',
            'path_difference',
            'distance',
            'total_failed_num__x__num_commits',
            'total_failed_num__x__num_chgs',
            'total_failed_num__x__updated_num_56d',
            'total_failed_num__x__updated_num_56c',
            'total_failed_num__x__path_difference',
            'total_failed_num__x__distance',
            'failed_num_7d__x__num_commits',
            'failed_num_7d__x__num_chgs',
            'failed_num_7d__x__updated_num_56d',
            'failed_num_7d__x__updated_num_56c',
            'failed_num_7d__x__path_difference',
            'failed_num_7d__x__distance',
            'distance__x__num_commits',
            'distance__x__num_chgs',
            'distance__x__updated_num_56d',
            'distance__x__updated_num_56c',
            'distance__x__path_difference'
        ]

        enrich_authors = _create_func_to_enrich_authors(spark, contributor_stats, input_col='author')
        enrich_files = _create_func_to_enrich_files(spark, repo_commits, updated_file_stats,
                                                    input_commit_date='commit_date',
                                                    input_filenames='filenames')
        enumerate_related_tests = _create_func_to_enumerate_related_tests(spark, dep_graph, corr_map, test_files,
                                                                          input_files='filenames',
                                                                          depth=2, max_num_tests=16)
        enrich_tests = _create_func_to_enrich_tests(spark, repo_commits, failed_tests,
                                                    input_commit_date='commit_date',
                                                    input_test='test')
        compute_distances = _create_func_to_compute_distances(spark, dep_graph, test_files,
                                                              input_files='filenames', input_test='test')
        interacted_features = [
            ('total_failed_num', 'num_commits'),
            ('total_failed_num', 'num_chgs'),
            ('total_failed_num', 'updated_num_56d'),
            ('total_failed_num', 'updated_num_56c'),
            ('total_failed_num', 'path_difference'),
            ('total_failed_num', 'distance'),
            ('failed_num_7d', 'num_commits'),
            ('failed_num_7d', 'num_chgs'),
            ('failed_num_7d', 'updated_num_56d'),
            ('failed_num_7d', 'updated_num_56c'),
            ('failed_num_7d', 'path_difference'),
            ('failed_num_7d', 'distance'),
            ('distance', 'num_commits'),
            ('distance', 'num_chgs'),
            ('distance', 'updated_num_56d'),
            ('distance', 'updated_num_56c'),
            ('distance', 'path_difference')
        ]
        compute_interaction_features = _create_func_to_compute_interaction_features(input_cols=interacted_features)
        compute_file_cardinality = _create_func_to_compute_file_cardinality(input_col='filenames')
        explode_tests = lambda df: df.selectExpr('*', 'explode_outer(related_tests) test'), ['related_tests']
        select_features = lambda df: df.selectExpr(expected_features), expected_features

        to_features = _create_pipelines(
            'to_features', [
                enrich_authors,
                enrich_files,
                enumerate_related_tests,
                explode_tests,
                enrich_tests,
                compute_distances,
                compute_file_cardinality,
                compute_interaction_features,
                select_features
            ])

        predicted = _predict_failed_probs(to_features(df), clf)
        selected_test_df = predicted \
            .selectExpr(f'slice(tests, 1, {args.num_selected_tests}) selected_tests') \
            .selectExpr('selected_tests.test selected_tests')

        selected_tests = selected_test_df.collect()[0].selected_tests
        if args.format:
            print(_format_for_scalatest(selected_tests))
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
