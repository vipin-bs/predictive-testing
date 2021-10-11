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

import glob
import re
from typing import Any, List, Optional, Tuple


def list_classes(root_path: str, target_package: str) -> List[Tuple[str, str]]:
    java_package = re.compile(f"[a-zA-Z0-9/\-_]+\/({target_package.replace('.', '/')}\/[a-zA-Z0-9/\-_]+)[\.class|\$]")

    def _extract_package(path: str) -> Optional[str]:
        result = java_package.search(path)
        if result:
            return result.group(1).replace('/', '.')
        else:
            return None

    paths = [p for p in glob.glob(f'{root_path}/**/*.class', recursive=True)]
    classes = list(filter(lambda t: t[0] is not None, map(lambda p: (_extract_package(p), p), paths)))
    return classes


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


def create_func_to_extract_refs_from_class_file(target_package: str) -> Any:
    re_extract_refs = re.compile(f"({target_package.replace('.', '/')}/[a-zA-Z0-9/\-]+)")

    def extract_refs(path: str) -> List[str]:
        stdout, _, _ = _exec_subprocess(f"{_get_cmd_path('javap')} -c -p {path}", raise_error=False)
        opcodes = stdout.decode().split('\n')
        invoke_opcodes = list(filter(lambda op: re.search('invoke', op), opcodes))
        refs: List[str] = []
        for invoke_opcode in invoke_opcodes:
            for ref in re_extract_refs.findall(invoke_opcode):
                refs.append(ref.replace('/', '.'))

        return refs

    return extract_refs
