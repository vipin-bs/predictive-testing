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
import time
import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from pyspark.sql import Column, DataFrame, SparkSession, functions
from typing import Any, Dict, List, Optional, Tuple


def _setup_logger() -> Any:
    from logging import getLogger, Formatter, StreamHandler, DEBUG, INFO
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(ch)
    return logger


_logger = _setup_logger()


def _build_lgb_model(X: pd.DataFrame, y: pd.Series, n_jobs: int = -1, opts: Dict[str, str] = {}) -> Tuple[Any, float]:
    import lightgbm as lgb  # type: ignore[import]

    # TODO: Validate given parameter values
    def _get_option(key: str, default_value: Optional[str]) -> Any:
        return opts[str(key)] if str(key) in opts else default_value

    def _boosting_type() -> str:
        return _get_option("lgb.boosting_type", "gbdt")

    def _class_weight() -> str:
        return _get_option("lgb.class_weight", "balanced")

    def _learning_rate() -> float:
        return float(_get_option("lgb.learning_rate", "0.01"))

    def _max_depth() -> int:
        return int(_get_option("lgb.max_depth", "7"))

    def _max_bin() -> int:
        return int(_get_option("lgb.max_bin", "255"))

    def _reg_alpha() -> float:
        return float(_get_option("lgb.reg_alpha", "0.0"))

    def _min_split_gain() -> float:
        return float(_get_option("lgb.min_split_gain", "0.0"))

    def _n_estimators() -> int:
        return int(_get_option("lgb.n_estimators", "300"))

    def _importance_type() -> str:
        return _get_option("lgb.importance_type", "gain")

    def _n_splits() -> int:
        return int(_get_option("cv.n_splits", "3"))

    def _timeout() -> Optional[int]:
        opt_value = _get_option("hp.timeout", None)
        return int(opt_value) if opt_value is not None else None

    def _max_eval() -> int:
        return int(_get_option("hp.max_evals", "100000000"))

    def _no_progress_loss() -> int:
        return int(_get_option("hp.no_progress_loss", "50"))

    fixed_params = {
        "boosting_type": _boosting_type(),
        "objective": "binary",
        "class_weight": _class_weight(),
        "learning_rate": _learning_rate(),
        "max_depth": _max_depth(),
        "max_bin": _max_bin(),
        "reg_alpha": _reg_alpha(),
        "min_split_gain": _min_split_gain(),
        "n_estimators": _n_estimators(),
        "importance_type": _importance_type(),
        "random_state": 42,
        "n_jobs": n_jobs
    }

    def _create_model(params: Dict[str, Any]) -> Any:
        # Some params must be int
        for k in ["num_leaves", "subsample_freq", "min_child_samples"]:
            if k in params:
                params[k] = int(params[k])
        import copy
        p = copy.deepcopy(fixed_params)
        p.update(params)
        return lgb.LGBMClassifier(**p)

    from hyperopt import hp, tpe, Trials  # type: ignore[import]
    from hyperopt.early_stop import no_progress_loss  # type: ignore[import]
    from hyperopt.fmin import fmin  # type: ignore[import]
    from sklearn.model_selection import cross_val_score, StratifiedKFold  # type: ignore[import]

    # Forcibly disable INFO-level logging in the `hyperopt` module
    from logging import getLogger, WARN
    getLogger("hyperopt").setLevel(WARN)

    param_space = {
        "num_leaves": hp.quniform("num_leaves", 2, 100, 1),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "subsample_freq": hp.quniform("subsample_freq", 1, 20, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.01, 1.0),
        "min_child_samples": hp.quniform("min_child_samples", 1, 50, 1),
        "min_child_weight": hp.loguniform("min_child_weight", -3, 1),
        "reg_lambda": hp.loguniform("reg_lambda", -2, 3)
    }

    def _objective(params: Dict[str, Any]) -> float:
        model = _create_model(params)
        fit_params = {
            # TODO: Raises an error if a single regressor is used
            # "categorical_feature": "auto",
            "verbose": 0
        }
        try:
            # TODO: Replace with `lgb.cv` to remove the `sklearn` dependency
            cv = StratifiedKFold(n_splits=_n_splits(), shuffle=True)
            scores = cross_val_score(
                model, X, y, scoring="f1_macro", cv=cv, fit_params=fit_params, n_jobs=n_jobs)
            return -scores.mean()

        # it might throw an exception because `y` contains
        # previously unseen labels.
        except Exception as e:
            _logger.warning(f"{e.__class__}: {e}")
            return 0.0

    def _early_stop_fn() -> Any:
        no_progress_loss_fn = no_progress_loss(_no_progress_loss())
        if _timeout() is None:
            return no_progress_loss_fn

        # Set base time for budget mechanism
        start_time = time.time()

        def _timeout_fn(trials, best_loss=None, iteration_no_progress=0):  # type: ignore
            no_progress_loss, meta = no_progress_loss_fn(trials, best_loss, iteration_no_progress)
            timeout = time.time() - start_time > _timeout()
            return no_progress_loss or timeout, meta

        return _timeout_fn

    trials = Trials()
    best_params = fmin(
        fn=_objective,
        space=param_space,
        algo=tpe.suggest,
        trials=trials,
        max_evals=_max_eval(),
        early_stop_fn=_early_stop_fn(),
        rstate=np.random.RandomState(42),
        show_progressbar=False,
        verbose=False)

    _logger.info("hyperopt: #eval={}/{}".format(len(trials.trials), _max_eval()))

    # Builds a model with `best_params`
    # TODO: Could we extract constraint rules (e.g., FD and CFD) from built statistical models?
    model = _create_model(best_params)
    model.fit(X, y)

    def _feature_importances() -> List[Any]:
        f = filter(lambda x: x[1] > 0.0, zip(model.feature_name_, model.feature_importances_))
        return list(sorted(f, key=lambda x: x[1], reverse=True))

    _logger.info(f"lightgbm: feature_importances={_feature_importances()}")

    sorted_lst = sorted(trials.trials, key=lambda x: x['result']['loss'])
    min_loss = sorted_lst[0]['result']['loss']
    return model, -min_loss


def _rebalance_training_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    from imblearn.under_sampling import RepeatedEditedNearestNeighbours
    renn = RepeatedEditedNearestNeighbours()
    X_res, y_res = renn.fit_resample(X, y)
    from collections import Counter
    _logger.info(f"Rebalancing training data: {dict(Counter(y).items())} => {dict(Counter(y_res).items())}")
    return X_res, y_res


def _create_func_to_enumerate_related_tests(spark: SparkSession,
                                            rev_dep_graph: Dict[str, List[str]],
                                            depth: int) -> Any:
    broadcasted_rev_dep_graph = spark.sparkContext.broadcast(rev_dep_graph)
    broadcasted_depth = spark.sparkContext.broadcast(depth)

    @functions.pandas_udf("string")  # type: ignore
    def _enumerate_tests(idents: pd.Series) -> pd.Series:
        rev_dep_graph = broadcasted_rev_dep_graph.value
        depth = broadcasted_depth.value

        def _traverse(target):  # type: ignore
            if not target:
                return []

            subgraph = {}
            visited_nodes = set()
            keys = list([target])
            for i in range(0, depth):
                next_keys = set()
                for key in keys:
                    if key in rev_dep_graph and key not in visited_nodes:
                        nodes = rev_dep_graph[key]
                        next_keys.update(nodes)
                visited_nodes.update(keys)
                keys = list(next_keys)
            if keys is not None:
                visited_nodes.update(keys)
            tests = list(filter(lambda n: n.endswith('Suite'), visited_nodes))
            return tests

        ret = []
        for ident in idents:
            s = _traverse(ident)
            ret.append(json.dumps({'tests': s}))

        return pd.Series(ret)

    def _func(df: DataFrame) -> DataFrame:
        test_df = df.selectExpr('sha', 'explode_outer(files.file.name) filename') \
            .selectExpr('sha', 'regexp_extract(filename, ".*(org\/apache\/spark\/.+?)\.scala", 1) ident') \
            .withColumn('tests', _enumerate_tests(functions.expr('ident'))) \
            .selectExpr('sha', 'ident', 'from_json(tests, "tests ARRAY<STRING>").tests tests') \
            .selectExpr('sha', 'explode_outer(tests) test') \
            .groupBy('sha') \
            .agg(functions.expr('collect_set(test) tests')) \
            .selectExpr('sha', 'size(tests) target_card', 'transform(tests, p -> replace(p, "\/", ".")) related_tests')

        return test_df

    return _func


def _create_func_to_compute_distance(spark: SparkSession, test_files: Dict[str, str]) -> Any:
    broadcasted_test_files = spark.sparkContext.broadcast(test_files)

    @functions.pandas_udf("int")  # type: ignore
    def _distance(filenames: pd.Series, test: pd.Series) -> pd.Series:
        test_files = broadcasted_test_files.value

        ret = []
        for names, t in zip(filenames, test):
            if t in test_files:
                fpath = test_files[t].split('/')
                distances = []
                for n in json.loads(names):
                    f = n.split('/')
                    distances.append(len(set(f) ^ set(fpath)) - 2)
                ret.append(min(distances))
            else:
                ret.append(256)

        return pd.Series(ret)

    def _func(df: DataFrame, files: str, test: str) -> DataFrame:
        distance = _distance(functions.expr(f'to_json({files})'), functions.expr(test))
        return df.withColumn('minimal_distance', distance)

    return _func


def _expand_failed_tests_by_failed_rate(df: DataFrame) -> DataFrame:
    def _intvl(d: int) -> str:
        return f"current_timestamp() - interval {d} days"

    def _failed(d: int) -> str:
        return f'CASE WHEN commit_date > intvl_{d}days THEN 1 ELSE 0 END'

    failed_test_df = df \
        .selectExpr(
            'to_timestamp(commit_date, "yyy/MM/dd HH:mm:ss") commit_date',
            'explode_outer(failed_tests) failed_test',
            f'{_intvl(7)} intvl_7days',
            f'{_intvl(14)} intvl_14days',
            f'{_intvl(28)} intvl_28days') \
        .where('failed_test IS NOT NULL') \
        .selectExpr(
            f'{_failed(7)} failed_7d',
            f'{_failed(14)} failed_14d',
            f'{_failed(28)} failed_28d',
            'failed_test') \
        .groupBy('failed_test').agg(
            functions.expr('sum(failed_7d) failed_7d'),
            functions.expr('sum(failed_14d) failed_14d'),
            functions.expr('sum(failed_28d) failed_28d'))

    df = df.join(failed_test_df, df.test == failed_test_df.failed_test, 'LEFT_OUTER') \
        .na.fill({'failed_7d': 0, 'failed_14d': 0, 'failed_28d': 0}) \
        .drop('failed_test')

    return df


def _expand_updated_files(df: DataFrame) -> DataFrame:
    def _sum(c: str) -> Column:
        return functions.expr(f'aggregate({c}, 0, (x, y) -> int(x) + int(y))')

    def _expand_by_update_rate(df: DataFrame) -> DataFrame:
        return df \
            .withColumn('total_udpated_3d', _sum('files.updated[0]')) \
            .withColumn('total_udpated_14d', _sum('files.updated[1]')) \
            .withColumn('total_udpated_56d', _sum('files.updated[2]')) \
            .na.fill({'total_udpated_3d': 0, 'total_udpated_14d': 0, 'total_udpated_56d': 0})

    df = _expand_by_update_rate(df) \
        .withColumn('total_adds', _sum('files.file.additions')) \
        .withColumn('total_dels', _sum("files.file.deletions")) \
        .withColumn('total_chgs', _sum("files.file.changes"))

    return df


def _create_feature_from(df: DataFrame,
                         enumerate_related_tests: Any,
                         compute_minimal_file_distance: Any) -> DataFrame:
    # TODO: Revisit this feature engineering
    df = df.selectExpr('sha', 'commit_date', 'files', 'failed_tests')
    df = _expand_updated_files(df) \
        .selectExpr('*', 'files.file.name filenames', 'size(files) file_card')

    related_test_df = enumerate_related_tests(df)
    df = df.join(related_test_df, 'sha', 'LEFT_OUTER') \
        .selectExpr('*', 'CASE WHEN size(failed_tests) = 0 THEN 0 ELSE 1 END failed') \
        .selectExpr('*', 'CASE WHEN failed = 1 THEN failed_tests ELSE related_tests END tests') \
        .drop('files', 'related_tests') \
        .where('tests IS NOT NULL') \
        .selectExpr('*', 'explode_outer(tests) test') \
        .drop('tests')

    df = _expand_failed_tests_by_failed_rate(df).drop('commit_date', 'failed_tests')
    df = compute_minimal_file_distance(df, 'filenames', 'test').drop('filenames')

    return df


def _build_model(df: DataFrame) -> Any:
    pdf = df.drop('sha', 'test').toPandas()
    X = pdf[pdf.columns[pdf.columns != 'failed']]  # type: ignore
    y = pdf['failed']
    X, y = _rebalance_training_data(X, y)
    clf, score = _build_lgb_model(X, y)
    _logger.info(f"model score: {score}")
    return clf


def _create_temp_name(prefix: str = "temp") -> str:
    import datetime
    return f'{prefix}_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}'


def _train_test_split(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    test_nrows = int(df.count() * 0.20)
    test_df = df.orderBy(functions.expr('to_timestamp(commit_date, "yyy/MM/dd HH:mm:ss")').desc()).limit(test_nrows)
    train_df = df.subtract(test_df)
    return train_df, test_df


def _predict_failed_probs(spark: SparkSession, clf: Any, test_df: DataFrame) -> DataFrame:
    df = test_df.selectExpr('sha', 'test').toPandas()
    X = test_df.drop('sha', 'failed', 'test').toPandas()
    predicted = clf.predict_proba(X)
    pmf = map(lambda p: {"classes": clf.classes_.tolist(), "probs": p.tolist()}, predicted)
    pmf = map(lambda p: json.dumps(p), pmf)  # type: ignore
    df['predicted'] = pd.Series(list(pmf))
    to_map_expr = functions.expr('from_json(predicted, "classes array<string>, probs array<double>")')
    to_failed_prob = 'map_from_arrays(pmf.classes, pmf.probs)["1"] failed_prob'
    df_with_failed_probs = spark.createDataFrame(df) \
        .withColumn('pmf', to_map_expr).selectExpr('sha', 'test', to_failed_prob)
    return df_with_failed_probs


def _compute_eval_metrics(df: DataFrame, predicted: DataFrame) -> List[Tuple[float, float]]:
    def _metric(thres: float) -> float:
        p = predicted.where(f'failed_prob > {thres}') \
            .groupBy('sha').agg(functions.expr('collect_set(test)').alias('tests'))
        eval_df = df.selectExpr('sha', 'failed_tests').join(p, 'sha', 'LEFT_OUTER') \
            .selectExpr('sha', 'failed_tests', 'coalesce(tests, array()) tests')
        eval_df = eval_df.selectExpr(
            'sha',
            'size(failed_tests) num_failed_tests',
            'size(tests) num_tests',
            'size(array_intersect(failed_tests, tests)) covered'
        )
        row = eval_df.selectExpr('SUM(covered) / SUM(num_failed_tests) p').collect()[0]
        return row.p

    metrics = [(thres, _metric(thres)) for thres in [0.0, 0.2, 0.4, 0.6, 0.8]]
    return metrics


def _format_eval_metrics(metrics: List[Tuple[float, float]]) -> str:
    strbuf: List[str]  = []
    strbuf.append('|  failed prob. threshold  |  coverage  |')
    strbuf.append('| ---- | ---- |')
    for prob, coverage in metrics:
        strbuf.append(f'| {prob} | {coverage} |')

    return '\n'.join(strbuf)


def _train_ptest_model(output_path: str, train_log_fpath: str, build_deps: str) -> None:
    dep_graph_fpath = f'{build_deps}/dep-graph.json'
    rev_dep_graph_fpath = f'{build_deps}/rev-dep-graph.json'
    test_list_fpath = f'{build_deps}/test-files.json'

    if not os.path.exists(output_path) or not os.path.isdir(output_path):
        raise ValueError(f"Output directory not found in {os.path.abspath(output_path)}")
    if not os.path.exists(train_log_fpath):
        raise ValueError(f"Training data not found in {os.path.abspath(train_log_fpath)}")
    if not os.path.exists(dep_graph_fpath):
        raise ValueError(f"Dependency graph file not found in {os.path.abspath(dep_graph_fpath)}")
    if not os.path.exists(rev_dep_graph_fpath):
        raise ValueError(f"Reverse dependency graph file not found in {os.path.abspath(rev_dep_graph_fpath)}")
    if not os.path.exists(test_list_fpath):
        raise ValueError(f"Dependency graph file not found in {os.path.abspath(test_list_fpath)}")

    from pathlib import Path
    dep_graph = json.loads(Path(dep_graph_fpath).read_text())
    rev_dep_graph = json.loads(Path(rev_dep_graph_fpath).read_text())
    test_files = json.loads(Path(test_list_fpath).read_text())

    # Initializes a Spark session
    # NOTE: Since learning tasks run longer, we set a large value (6h)
    # to the network timeout value.
    spark = SparkSession.builder \
        .enableHiveSupport() \
        .getOrCreate()

    # Suppress user warinig messages in Python
    import warnings
    warnings.simplefilter("ignore", UserWarning)

    # Supresses `WARN` messages in JVM
    spark.sparkContext.setLogLevel("ERROR")

    enumerate_related_tests = _create_func_to_enumerate_related_tests(spark, rev_dep_graph, depth=2)
    compute_distance = _create_func_to_compute_distance(spark, test_files)

    def _to_feature(df: DataFrame) -> DataFrame:
        return _create_feature_from(df, enumerate_related_tests, compute_distance)

    try:
        # Assigns sha if it is an empty string (TODO: Needs to assign sha when collecting GitHub logs)
        df = spark.read.format('json').load(train_log_fpath) \
            .withColumn('sha_', functions.expr('case when length(sha) > 0 then sha else sha(string(random())) end')) \
            .drop('sha') \
            .withColumnRenamed('sha_', 'sha') \
            .distinct()

        train_df, test_df = _train_test_split(df)
        _logger.info(f"Split data: #train={train_df.count()}, #test={test_df.count()}")

        clf = _build_model(_to_feature(train_df))

        output_prefix = f"{output_path}/ptesting-model"
        with open(f"{output_prefix}.pkl", 'wb') as f:
            import pickle
            pickle.dump(clf, f)

        predicted = _predict_failed_probs(spark, clf, _to_feature(test_df))
        metrics = _compute_eval_metrics(test_df, predicted)
        with open(f"{output_prefix}-metrics.md", 'w') as f:
            f.write(_format_eval_metrics(metrics))

    finally:
        spark.stop()


def main() -> None:
    # Parses command-line arguments
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--train-log-data', type=str, required=True)
    parser.add_argument('--build-deps', type=str, required=True)
    args = parser.parse_args()

    _train_ptest_model(args.output, args.train_log_data, args.build_deps)


if __name__ == '__main__':
    main()
