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

import json
import os
from typing import Any

from ptesting import depgraph
from ptesting import javaclass


def _select_handlers_from_file_type(type: str) -> Any:
    # TDOO: Supports other types
    if type == 'java':
        return javaclass.list_classes, javaclass.list_test_classes, javaclass.extract_refs
    else:
        raise ValueError(f'Unknown file type: {type}')


def _write_data_as(prefix: str, path: str, data: Any) -> None:
    with open(f"{path}/{prefix}.json", mode='w') as f:
        f.write(json.dumps(data, indent=2))


def _analyze_build_deps(output_path: str, overwrite: bool, root_paths: str, target_package: str,
                        list_files: Any, list_test_files: Any,
                        extract_refs: Any) -> None:
    if len(root_paths) == 0:
        raise ValueError("At least one path must be specified in '--root-paths'")
    if len(output_path) == 0:
        raise ValueError("Output path must be specified in '--output'")
    if len(target_package) == 0:
        raise ValueError("Target package must be specified in '--target-package'")

    # Make an output dir in advance
    import shutil
    overwrite and shutil.rmtree(output_path, ignore_errors=True)
    os.mkdir(output_path)

    adj_list, rev_adj_list, test_files = depgraph.build_dependency_graphs(
        root_paths.split(','), target_package, list_files, list_test_files, extract_refs)

    # Writes dependency graphes into files
    _write_data_as('dep-graph', output_path, adj_list)
    _write_data_as('rev-dep-graph', output_path, rev_adj_list)
    _write_data_as('test-files', output_path, test_files)


def _generate_dependency_graph(path: str, targets: str, depth: int) -> str:
    if len(path) == 0:
        raise ValueError("Path of dependency graph must be specified in '--graph'")
    if not os.path.isfile(path):
        raise ValueError("File must be specified in '--graph'")
    if len(targets) == 0:
        raise ValueError("At least one target must be specified in '--targets'")
    if depth <= 0:
        raise ValueError("'depth' must be positive")

    from pathlib import Path
    dep_graph = json.loads(Path(path).read_text())
    target_nodes = targets.replace(".", "/").split(",")
    subgraph, subnodes = depgraph.select_subgraph(target_nodes, dep_graph, depth)
    return depgraph.generate_graph(subnodes, target_nodes, subgraph)


def main():
    # Parses command-line arguments
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--command', dest='command', type=str, required=True)
    parser.add_argument('--root-paths', dest='root_paths', type=str, required=False, default='')
    parser.add_argument('--file-type', dest='file_type', type=str, required=False, default='java')
    parser.add_argument('--output', dest='output', type=str, required=False, default='')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    parser.add_argument('--target-package', dest='target_package', type=str, required=False, default='')
    parser.add_argument('--graph', dest='graph', type=str, required=False, default='')
    parser.add_argument('--targets', dest='targets', type=str, required=False, default='')
    parser.add_argument('--depth', dest='depth', type=int, required=False, default=3)
    args = parser.parse_args()

    if args.command == 'analyze':
        list_files, list_test_files, extract_refs = _select_handlers_from_file_type(args.file_type)
        _analyze_build_deps(args.output, args.overwrite, args.root_paths, args.target_package,
                            list_files, list_test_files, extract_refs)
    elif args.command == 'visualize':
        print(_generate_dependency_graph(args.graph, args.targets, args.depth))
    else:
        import sys
        print(f"Unknown command: {args.command}")
        sys.exit(-1)


if __name__ == "__main__":
    main()
