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
Helper functions to analyze java class files
"""

import re
from typing import Any, List, Optional, Tuple


# Compiled regex patterns
RE_IS_TEST_CLASS = re.compile('Suite|Spec')


def _exec_subprocess(cmd: str, raise_error: bool = True) -> Tuple[Any, Any, Any]:
    import subprocess
    child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = child.communicate()
    rt = child.returncode
    if rt != 0 and raise_error:
        raise RuntimeError(f"command return code is not 0. got {rt}. stderr = {stderr}")  # type: ignore

    return stdout, stderr, rt


def _get_cmd_path(cmd: str) -> Any:
    from shutil import which
    if which(cmd) is None:
        raise RuntimeError(f"Could not find '{cmd}' executable")
    return cmd


def _format_class_path(path: str, package_prefix: str) -> Optional[str]:
    format_class_path = re.compile(f"[a-zA-Z0-9/\-]+/({package_prefix.replace('.', '/')}/[a-zA-Z0-9/\-]+)[\.class|\$]")
    result = format_class_path.search(path)
    if result:
        return result.group(1)
    else:
        return None


def _list_class_files(root: str, p: str) -> List[str]:
    import glob
    return [p for p in glob.glob(f'{root}/**/{p}', recursive=True)]


def _is_test_class(path: str) -> bool:
    return RE_IS_TEST_CLASS.search(path) is not None


def list_classes(root: str, target_package: str) -> List[Tuple[str, str]]:
    classes = list(filter(lambda c: not _is_test_class(c), _list_class_files(root, '*.class')))
    qualified_classes = []
    for path in classes:
        clazz = _format_class_path(path, target_package)
        if clazz is not None:
            qualified_classes.append((clazz, path))

    return qualified_classes


def list_test_classes(root: str, target_package: str) -> List[Tuple[str, str]]:
    test_classes = list(filter(lambda c: _is_test_class(c), _list_class_files(root, '*.class')))
    qualified_test_classes = []
    for path in test_classes:
        clazz = _format_class_path(path, target_package)
        if clazz is not None:
            qualified_test_classes.append((clazz, path))

    return qualified_test_classes


def extract_refs(path: str, target_package: str) -> List[str]:
    stdout, stderr, rt = _exec_subprocess(f"{_get_cmd_path('javap')} -c -p {path}", raise_error=False)
    opcodes = stdout.decode().split('\n')
    invoke_opcodes = list(filter(lambda op: re.search('invoke', op), opcodes))
    refs: List[str] = []
    extract_refs = re.compile(f"({target_package.replace('.', '/')}/[a-zA-Z0-9/\-]+)")
    for invoke_opcode in invoke_opcodes:
        for ref in extract_refs.findall(invoke_opcode):
            refs.append(ref)

    return refs
