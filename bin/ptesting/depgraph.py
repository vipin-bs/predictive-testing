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
import pickle
from typing import Any, Dict, List, Set, Tuple
from functools import reduce


def build_dependency_graphs(root_paths: List[str], target_package: str,
                            list_files: Any, list_test_files: Any,
                            extract_refs: Any) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[str]]:
    adj_list: Dict[str, Set[str]] = {}
    rev_adj_list: Dict[str, Set[str]] = {}

    # Collects class files from given paths
    files = reduce(lambda x, y: x.extend(y), [ list_files(p, target_package) for p in root_paths])
    test_files = reduce(lambda x, y: x.extend(y), [ list_test_files(p, target_package) for p in root_paths])
    all_files = [*files, *test_files]
    if len(all_files) == 0:
        raise RuntimeError(f"No file found in [{', '.join(root_paths)}]")

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

    return to_graph(adj_list), to_graph(rev_adj_list), \
        list(map(lambda x: x[0], test_files))


def generate_graph(nodes: List[str], targets: List[str], edges: Dict[str, List[str]]) -> str:
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


def select_subgraph(targets: List[str], edges: Dict[str, List[str]],
                    depth: int) -> Tuple[Dict[str, List[str]], List[str]]:
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
