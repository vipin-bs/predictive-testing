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

import os
import warnings
from pyspark.sql import SparkSession


def _train_ptest_model(output_path: str, train_data: str, build_deps: str) -> None:
    dep_graph_fpath = f'{build_deps}/dep-graph.json'
    rev_dep_graph_fpath = f'{build_deps}/rev-dep-graph.json'
    test_list_fpath = f'{build_deps}/test-files.json'

    if not os.path.exists(output_path) or not os.path.isdir(output_path):
        raise ValueError(f"Output directory not found in {os.path.abspath(output_path)}")
    if not os.path.exists(dep_graph_fpath):
        raise ValueError(f"Dependency graph file not found in {os.path.abspath(dep_graph_fpath)}")
    if not os.path.exists(rev_dep_graph_fpath):
        raise ValueError(f"Reverse dependency graph file not found in {os.path.abspath(rev_dep_graph_fpath)}")
    if not os.path.exists(test_list_fpath):
        raise ValueError(f"Dependency graph file not found in {os.path.abspath(test_list_fpath)}")

    # Initializes a Spark session
    # NOTE: Since learning tasks run longer, we set a large value (6h)
    # to the network timeout value.
    spark = SparkSession.builder \
        .enableHiveSupport() \
        .getOrCreate()

    # Suppress warinig messages in Python
    warnings.simplefilter('ignore')

    # Supresses `WARN` messages in JVM
    spark.sparkContext.setLogLevel("ERROR")

    try:
        spark.range(1).show()
        print(output_path)
        print(dep_graph_fpath)
        print(rev_dep_graph_fpath)
        print(test_list_fpath)
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
