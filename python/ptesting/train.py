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

import time
import numpy as np  # type: ignore[import]
import pandas as pd  # type: ignore[import]
from typing import Any, Dict, List, Optional, Tuple


def _setup_logger() -> Any:
    from logging import getLogger, NullHandler, INFO
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    logger.addHandler(NullHandler())
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
        return int(_get_option("hp.no_progress_loss", "1000"))

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

    _logger.debug("hyperopt: #eval={}/{}".format(len(trials.trials), _max_eval()))

    # Builds a model with `best_params`
    # TODO: Could we extract constraint rules (e.g., FD and CFD) from built statistical models?
    model = _create_model(best_params)
    model.fit(X, y)

    def _feature_importances() -> List[Any]:
        f = filter(lambda x: x[1] > 0.0, zip(model.feature_name_, model.feature_importances_))
        return list(sorted(f, key=lambda x: x[1], reverse=True))

    _logger.debug(f"lightgbm: feature_importances={_feature_importances()}")

    sorted_lst = sorted(trials.trials, key=lambda x: x['result']['loss'])
    min_loss = sorted_lst[0]['result']['loss']
    return model, -min_loss


def build_model(X: pd.DataFrame, y: pd.Series, opts: Dict[str, str] = {}) -> Any:
    return _build_lgb_model(X, y, opts=opts)


def rebalance_training_data(X: pd.DataFrame, y: pd.Series, coeff: float = 1.0) -> Tuple[pd.DataFrame, pd.Series]:
    # TODO: To improve model performance, we need to reconsider this sampling method?
    from collections import Counter
    from imblearn.under_sampling import RandomUnderSampler
    y_target_hist = dict(Counter(y).items())
    min_key, min_value = min(y_target_hist.items(), key=lambda kv: kv[1])
    for k in y_target_hist.keys():
        if k != min_key:
            y_target_hist[k] = int(min_value * coeff)

    rus = RandomUnderSampler(sampling_strategy=y_target_hist, random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    _logger.info(f"Sampling training data (strategy={y_target_hist}): {dict(Counter(y).items())}"
                 f" => {dict(Counter(y_res).items())}")
    return X_res, y_res
