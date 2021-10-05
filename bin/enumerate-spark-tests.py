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
Enumerate all Spark tests in a specified path
"""

import json
import glob
import os
import re
from typing import Dict, List, Optional


# Compiled regex patterns
RE_IS_JAVA_TEST = re.compile('\/[a-zA-Z0-9/\-_]+Suite\.class$')
RE_IS_PYTHON_TEST = re.compile('\/test_[a-zA-Z0-9/\-_]+\.py$')
RE_FORMAT_JAVA_TEST_PATH = re.compile("[a-zA-Z0-9/\-_]+\/test-classes\/(org\/apache\/spark\/[a-zA-Z0-9/\-_]+)\.class$")
RE_FORMAT_PYTHON_TEST_PATH = re.compile("[a-zA-Z0-9/\-_]+\/python\/(pyspark\/[a-zA-Z0-9/\-_]+)\.py$")


def _is_java_test(path: str) -> bool:
    return RE_IS_JAVA_TEST.search(path) is not None


def _is_python_test(path: str) -> bool:
    return RE_IS_PYTHON_TEST.search(path) is not None


def _extract_class_package(path: str) -> Optional[str]:
    result = RE_FORMAT_JAVA_TEST_PATH.search(path)
    if result:
        return result.group(1).replace('/', '.')
    else:
        return None


def _extract_python_package(path: str) -> Optional[str]:
    result = RE_FORMAT_PYTHON_TEST_PATH.search(path)
    if result:
        return result.group(1).replace('/', '.')
    else:
        return None


def _format_path(path: str, root_path: str) -> str:
    prefix_path = os.path.abspath(root_path)
    return path[len(prefix_path) + 1:]


def _enumerate_java_tests(paths: List[str], root_path: str) -> Dict[str, str]:
    paths = filter(lambda p: _is_java_test(p), paths)  # type: ignore
    tests = map(lambda p: (_extract_class_package(p), _format_path(p, root_path)), paths)
    tests = filter(lambda t: t[0] is not None, tests)  # type: ignore
    return dict(tests)  # type: ignore


def _enumerate_python_tests(paths: List[str], root_path: str) -> Dict[str, str]:
    paths = filter(lambda p: _is_python_test(p), paths)  # type: ignore
    tests = map(lambda p: (_extract_python_package(p), _format_path(p, root_path)), paths)
    tests = filter(lambda t: t[0] is not None, tests)  # type: ignore
    return dict(tests)  # type: ignore


def _enumerate_tests(root_path: str) -> None:
    paths = [p for p in glob.glob(f'{root_path}/**', recursive=True)]
    java_tests = _enumerate_java_tests(paths, root_path)
    python_tests = _enumerate_python_tests(paths, root_path)
    print(json.dumps({**java_tests, **python_tests}, indent=2))


def main() -> None:
    # Parses command-line arguments
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--path', dest='path', type=str, required=True)
    args = parser.parse_args()

    _enumerate_tests(args.path)


if __name__ == "__main__":
    main()
