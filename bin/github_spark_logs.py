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

import re
from typing import Any, List, Optional, Tuple


_target_workflow_runs = [
    'Build and test'
]


_target_workflow_jobs = [
    'pyspark-sql, pyspark-mllib, pyspark-resource',
    'pyspark-core, pyspark-streaming, pyspark-ml',
    'pyspark-pandas',
    'pyspark-pandas-slow',
    'core, unsafe, kvstore, avro, network-common, network-shuffle, repl, launcher, examples, sketch, graphx',
    'catalyst, hive-thriftserver',
    'streaming, sql-kafka-0-10, streaming-kafka-0-10, mllib-local, mllib, yarn, mesos, kubernetes, hadoop-cloud, spark-ganglia-lgpl',
    'hive - slow tests',
    'hive - other tests',
    'sql - slow tests',
    'sql - other tests',
    'Run docker integration tests',
    'Run TPC-DS queries with SF=1'
]


_RE_COMPILE_FAILURE = re.compile(r"error.+? Compilation failed")
_RE_SCALA_TEST = re.compile(r"error.+?(org\.apache\.spark\.[a-zA-Z0-9\.]+Suite)")
_RE_PYTHON_TEST = re.compile(r"Had test failures in (pyspark\.[a-zA-Z0-9\._]+) with python")


# TODO: Needs to generalize this method
def _extract_spark_failed_tests_from(logs: str) -> Optional[List[str]]:
    if _RE_COMPILE_FAILURE.search(logs) is not None:
        return None

    tests: List[str] = []

    # TODO: Revisits the regex pattern
    tests.extend(_RE_SCALA_TEST.findall(logs))
    tests.extend(_RE_PYTHON_TEST.findall(logs))
    return list(set(tests))


def create_spark_workflow_handlers() -> Tuple[List[str], List[str], Any]:
    return _target_workflow_runs, _target_workflow_jobs, \
        _extract_spark_failed_tests_from
