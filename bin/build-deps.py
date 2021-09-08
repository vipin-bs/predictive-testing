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

"""
Analyze build dependencies in a coarse-grained way
"""

import os
import pickle
from typing import Any, Dict

from ptesting import callgraph
from ptesting import javaclass


def _setup_logger():
    from logging import getLogger, StreamHandler, Formatter, DEBUG, INFO
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(Formatter('%(asctime)s.%(msecs)03d: %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(ch)
    return logger


# For logging setup
_logger = _setup_logger()


# Sets output names for call graphs
env = dict(os.environ)
CALL_GRAPH_NAME = env['CALL_GRAPH_NAME'] if 'CALL_GRAPH_NAME' in env \
    else 'dep-graph'
REV_CALL_GRAPH_NAME = env['REV_CALL_GRAPH_NAME'] if 'REV_CALL_GRAPH_NAME' in env \
    else 'rev-dep-graph'


def _select_handlers_from_file_type(type: str) -> Any:
    # TDOO: Supports other types
    if type == 'java':
        return javaclass.list_classes, javaclass.list_test_classes, javaclass.extract_refs
    else:
        raise ValueError(f'Unknown file type: {type}')


def _write_data_as(prefix: str, path: str, data: Any) -> None:
    output_path = f"{path}/{prefix}.pkl"
    _logger.info(f"Writing data as '{output_path}'...")
    with open(output_path, mode='wb') as f:
        pickle.dump(data, f)


def _analyze_build_deps(root_paths: str,
                        output_path: str,
                        target_package: str,
                        list_files: Any, list_test_files: Any,
                        extract_refs: Any) -> None:
    if len(root_paths) == 0:
        raise ValueError("At least one path must be specified in '--root-paths'")
    if len(output_path) == 0:
        raise ValueError("Output path must be specified in '--output'")
    if len(target_package) == 0:
        raise ValueError("Target package must be specified in '--target-package'")

    # Make an output dir in advance
    _logger.info(f"Making an output directory in {os.path.abspath(output_path)}...")
    os.mkdir(output_path)

    adj_list, rev_adj_list = callgraph.build_call_graphs(
        root_paths.split(','), target_package, list_files, list_test_files, extract_refs)

    # Writes call graphes into files
    _write_data_as(CALL_GRAPH_NAME, output_path, adj_list)
    _write_data_as(REV_CALL_GRAPH_NAME, output_path, rev_adj_list)


def _list_test_files(root_paths: str, target_package: str, list_test_files: Any) -> None:
    if len(root_paths) == 0:
        raise ValueError("At least one path must be specified in '--root-paths'")
    if len(target_package) == 0:
        raise ValueError("Target package must be specified in '--target-package'")

    test_classes: List[str] = []
    for p in root_paths.split(','):
        test_classes.extend(map(lambda x: x[0], list_test_files(p, target_package)))

    print("\n".join(set(test_classes)))


def main():
    # Parses command-line arguments
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--command', dest='command', type=str, required=True)
    parser.add_argument('--root-paths', dest='root_paths', type=str, required=False, default='')
    parser.add_argument('--file-type', dest='file_type', type=str, required=False, default='java')
    parser.add_argument('--output', dest='output', type=str, required=False, default='')
    parser.add_argument('--target-package', dest='target_package', type=str, required=False, default='')
    parser.add_argument('--graph', dest='graph', type=str, required=False, default='')
    parser.add_argument('--targets', dest='targets', type=str, required=False, default='')
    parser.add_argument('--depth', dest='depth', type=int, required=False, default=3)
    args = parser.parse_args()

    if args.command in ['analyze', 'list']:
        list_files, list_test_files, extract_refs = _select_handlers_from_file_type(args.file_type)
        if args.command == 'analyze':
            _analyze_build_deps(args.root_paths, args.output, args.target_package, list_files, list_test_files, extract_refs)
        else:
            _list_test_files(args.root_paths, args.target_package, list_test_files)
    elif args.command == 'visualize':
        callgraph.generate_call_graph(args.graph, args.targets, args.depth)
    else:
        import sys
        print(f"Unknown command: {args.command}")
        sys.exit(-1)


if __name__ == "__main__":
    main()
