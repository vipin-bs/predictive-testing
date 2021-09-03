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
import re
from typing import Any, Dict, List, Set
from functools import reduce

import javaclass


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
    else 'call_graph'
REV_CALL_GRAPH_NAME = env['REV_CALL_GRAPH_NAME'] if 'REV_CALL_GRAPH_NAME' in env \
    else 'rev_call_graph'


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


def _build_call_graphs(root_paths: List[str],
                       target_package: str,
                       list_files: Any, list_test_files: Any,
                       extract_refs: Any):
    adj_list: Dict[str, Set[str]] = {}
    rev_adj_list: Dict[str, Set[str]] = {}

    # Collects class files from given paths
    files = reduce(lambda x, y: x.extend(y), [ list_files(p, target_package) for p in root_paths])
    test_files = reduce(lambda x, y: x.extend(y), [ list_test_files(p, target_package) for p in root_paths])
    all_files = [*files, *test_files]
    if len(all_files) == 0:
        raise Exception(f"No file found in [{', '.join(root_paths)}]")

    _logger.info(f"{len(all_files)} files ({len(test_files)} test files included) "
                 f"found in {','.join(root_paths)}")

    import tqdm
    n_files = len(all_files)
    for i in tqdm.tqdm(range(n_files)):
        node, path = all_files[i]
        refs = adj_list[node] if node in adj_list else set()
        extracted_refs = extract_refs(path, target_package)
        for ref in extracted_refs:
            if ref != node:
                if ref not in rev_adj_list:
                    rev_adj_list[ref] = set()
                rev_adj_list[ref].add(node)
                refs.add(ref)

        adj_list[node] = refs

    def to_graph(g: Dict[str, Set[str]]) -> Dict[str, List[str]]:
        return { k: list(v) for k, v in g.items() }

    return to_graph(adj_list), to_graph(rev_adj_list)


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

    adj_list, rev_adj_list = _build_call_graphs(
        root_paths.split(','), target_package, list_files, list_test_files, extract_refs)

    # Writes call graphes into files
    _write_data_as(CALL_GRAPH_NAME, output_path, adj_list)
    _write_data_as(REV_CALL_GRAPH_NAME, output_path, rev_adj_list)


def _generate_graph(nodes: List[str], targets: List[str], edges: Dict[str, List[str]]) -> str:
    # TODO: Normalize node strings
    def ns(s: str) -> str:
        return s.replace('/', '.')

    node_entries = []
    for node in nodes:
      if node in targets:
          node_entries.append(f"\"{ns(node)}\" [shape=\"oval\"];")
      else:
          node_entries.append(f"\"{ns(node)}\";")

    edge_entries = []
    for key, values in edges.items():
        for value in values:
            edge_entries.append(f"\"{ns(key)}\" -> \"{ns(value)}\";")

    node_defs = '\n'.join(node_entries)
    edge_defs = '\n'.join(edge_entries)
    return f"""
        digraph {{
            graph [pad="0.5", nodesep="0.5", ranksep="2", fontname="Helvetica"];
            node [shape=box]
            rankdir=LR;

            {node_defs}
            {edge_defs}
        }}
    """


def _select_subgraph(targets: List[str], edges: Dict[str, List[str]], depth: int):
    import json
    print(json.dumps(edges, indent=4))
    subgraph = {}
    visited_nodes = set()
    keys = targets
    for i in range(0, depth):
        next_keys = set()
        for key in keys:
            if key in edges and key not in visited_nodes:
                nodes = edges[key]
                subgraph[key] = nodes
                next_keys.update(nodes)

        visited_nodes.update(keys)
        keys = list(next_keys)

    if keys is not None:
        visited_nodes.update(keys)

    return subgraph, list(visited_nodes)


def _generate_call_graph(path: str, targets: str, depth: int) -> None:
    if len(path) == 0:
        raise ValueError("Path of call graph must be specified in '--graph'")
    if not os.path.isfile(path):
        raise ValueError("File must be specified in '--graph'")
    if len(targets) == 0:
        raise ValueError("At least one target must be specified in '--targets'")
    if depth <= 0:
        raise ValueError("'depth' must be positive")

    with open(path, mode='rb') as f:
        call_graph = pickle.load(f)

    target_nodes = targets.replace(".", "/").split(",")
    subgraph, subnodes = _select_subgraph(target_nodes, call_graph, depth)
    print(_generate_graph(subnodes, target_nodes, subgraph))


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

    if args.command == 'analyze':
        list_files, list_test_files, extract_refs = _select_handlers_from_file_type(args.file_type)
        _analyze_build_deps(args.root_paths, args.output, args.target_package, list_files, list_test_files, extract_refs)
    elif args.command == 'visualize':
        _generate_call_graph(args.graph, args.targets, args.depth)
    else:
        import sys
        print(f"Unknown command: {args.command}")
        sys.exit(-1)


if __name__ == "__main__":
    main()
