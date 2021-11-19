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
Analyze Spark repository to extract build dependency and test list
"""

import json
import glob
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ptesting import depgraph
from ptesting import javaclass


# Compiled regex patterns to extract information from a Spark repository
# TODO: Generalize this
RE_IS_JAVA_CLASS = re.compile('\/[a-zA-Z0-9/\-_]+[\.class|\$]')
RE_IS_JAVA_TEST = re.compile('\/[a-zA-Z0-9/\-_]+Suite\.class$')
RE_IS_PYTHON_TEST = re.compile('\/test_[a-zA-Z0-9/\-_]+\.py$')
RE_JAVA_CLASS_PATH = re.compile("[a-zA-Z0-9/\-_]+\/classes\/(org\/apache\/spark\/[a-zA-Z0-9/\-_]+)[\.class|\$]")
RE_JAVA_TEST_PATH = re.compile("[a-zA-Z0-9/\-_]+\/test-classes\/(org\/apache\/spark\/[a-zA-Z0-9/\-_]+Suite)\.class$")
RE_PYTHON_TEST_PATH = re.compile("[a-zA-Z0-9/\-_]+\/python\/(pyspark\/[a-zA-Z0-9/\-_]+)\.py$")
RE_PARSE_PATH = re.compile(f"[a-zA-Z0-9/\-]+/(org\/apache\/spark\/.+\/)([a-zA-Z0-9\-]+)\.scala")
RE_PARSE_SCALA_FILE = re.compile("class\s+([a-zA-Z0-9]+Suite)\s+extends\s+")


def _is_java_class(path: str) -> bool:
    return RE_IS_JAVA_CLASS.search(path) is not None and \
        RE_IS_JAVA_TEST.search(path) is None


def _is_java_test(path: str) -> bool:
    return RE_IS_JAVA_TEST.search(path) is not None


def _is_python_test(path: str) -> bool:
    return RE_IS_PYTHON_TEST.search(path) is not None


def _extract_package(path: str, regex: Any) -> Optional[str]:
    result = regex.search(path)
    if result:
        return result.group(1).replace('/', '.')
    else:
        return None


def _enumerate_java_classes(paths: List[str], root_path: str) -> Dict[str, str]:
    paths = filter(lambda p: _is_java_class(p), paths)  # type: ignore
    tests = map(lambda p: (_extract_package(p, RE_JAVA_CLASS_PATH), p), paths)
    tests = filter(lambda t: t[0] is not None, tests)  # type: ignore
    return dict(tests)  # type: ignore


def _enumerate_java_tests(paths: List[str], root_path: str) -> Dict[str, str]:
    paths = filter(lambda p: _is_java_test(p), paths)  # type: ignore
    tests = map(lambda p: (_extract_package(p, RE_JAVA_TEST_PATH), p), paths)
    tests = filter(lambda t: t[0] is not None, tests)  # type: ignore
    return dict(tests)  # type: ignore


def _enumerate_python_tests(paths: List[str], root_path: str) -> Dict[str, str]:
    paths = filter(lambda p: _is_python_test(p), paths)  # type: ignore
    tests = map(lambda p: (_extract_package(p, RE_PYTHON_TEST_PATH), p), paths)
    tests = filter(lambda t: t[0] is not None, tests)  # type: ignore
    return dict(tests)  # type: ignore


def _enumerate_spark_files(root_path: str) -> Any:
    files = [p for p in glob.glob(f'{root_path}/**', recursive=True)]
    java_classes = _enumerate_java_classes(files, root_path)
    java_tests = _enumerate_java_tests(files, root_path)
    python_tests = _enumerate_python_tests(files, root_path)
    return java_classes, java_tests, python_tests


def _format_path(path: str, root_path: str) -> str:
    prefix_path = os.path.abspath(root_path)
    return path[len(prefix_path) + 1:]


def _build_correlated_file_map(root_path: str, commits: List[Tuple[str, str, List[str]]]) -> Dict[str, List[str]]:
    import itertools
    correlated_files: Dict[str, Any] = {}
    for _, _, files in commits:
        group = []
        for f in files:
            qs = RE_PARSE_PATH.search(f)
            if qs:
                package = qs.group(1).replace('/', '.')
                try:
                    file_as_string = Path(f'{root_path}/{f}').read_text()
                    classes = RE_PARSE_SCALA_FILE.findall(file_as_string)
                    if classes:
                        group.append((f, list(map(lambda c: f'{package}{c}', classes))))
                    else:
                        clazz = qs.group(2)
                        group.append((f, [f'{package}{clazz}']))
                except:
                    pass
            else:
                group.append((f, []))

        # for x, y in filter(lambda p: p[0] != p[1], itertools.product(group, group)):
        for (path1, classes1), (path2, classes2) in itertools.product(group, group):
            if path1 not in correlated_files:
                correlated_files[path1] = set()

            correlated_files[path1].update(classes1 + classes2)

    for k, v in correlated_files.items():
        correlated_files[k] = list(v)

    return correlated_files


def _write_data_as(prefix: str, path: str, data: Any) -> None:
    with open(f"{path}/{prefix}.json", mode='w') as f:
        f.write(json.dumps(data, indent=2))


def main() -> None:
    # Parses command-line arguments
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--root-path', dest='root_path', type=str, required=True)
    parser.add_argument('--commits', type=str, required=True)
    parser.add_argument('--output', dest='output', type=str, required=True)
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    args = parser.parse_args()

    if not os.path.isdir(args.root_path):
        raise ValueError(f"Spark root dir not found in {os.path.abspath(args.root_path)}")
    if not os.path.isfile(args.commits):
        raise ValueError(f"Commit history file not found in {os.path.abspath(args.commits)}")

    if args.overwrite:
        import shutil
        shutil.rmtree(args.output, ignore_errors=True)

    # Make an output dir in advance
    os.mkdir(args.output)

    # Extract a list of all the available tests from the repository
    java_classes, java_tests, python_tests = _enumerate_spark_files(args.root_path)
    tests = dict(map(lambda t: (t[0], _format_path(t[1], args.root_path)), ({**java_tests, **python_tests}).items()))
    _write_data_as('test-files', args.output, tests)

    # Build a control folow graph from compiled class files
    java_files = list(({**java_classes, **java_tests}).items())
    extract_edges_from_path = javaclass.create_func_to_extract_refs_from_class_file('org.apache.spark')
    dep_graph = depgraph.build_dependency_graphs(java_files, extract_edges_from_path)
    _write_data_as('dep-graph', args.output, dep_graph)

    # Extract file correlation from a sequence of commit logs
    commits = json.loads(Path(args.commits).read_text())
    correlated_files = _build_correlated_file_map(args.root_path, commits)
    _write_data_as('correlated-files', args.output, correlated_files)


if __name__ == "__main__":
    main()
