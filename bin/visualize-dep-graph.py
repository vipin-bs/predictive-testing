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
from ptesting import depgraph
from pathlib import Path


def _generate_dependency_graph(path: str, targets: str, depth: int) -> str:
    if not os.path.isfile(path):
        raise ValueError("File must be specified in '--graph'")
    if depth <= 0:
        raise ValueError("'depth' must be positive")

    dep_graph = json.loads(Path(path).read_text())
    target_nodes = targets.replace(".", "/").split(",")
    subgraph, subnodes = depgraph.select_subgraph(target_nodes, dep_graph, depth)
    return depgraph.generate_graph(subnodes, target_nodes, subgraph)


def main() -> None:
    # Parses command-line arguments
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--graph', dest='graph', type=str, required=True)
    parser.add_argument('--targets', dest='targets', type=str, required=True)
    parser.add_argument('--depth', dest='depth', type=int, required=False, default=2)
    args = parser.parse_args()

    print(_generate_dependency_graph(args.graph, args.targets, args.depth))


if __name__ == "__main__":
    main()
