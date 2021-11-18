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

import re
from typing import Any, List, Tuple


def _exec_subprocess(cmd: str, raise_error: bool = True) -> Tuple[Any, Any, Any]:
    import subprocess
    child = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = child.communicate()
    rt = child.returncode
    if rt != 0 and raise_error:
        raise RuntimeError(f"command return code is not 0. got {rt}. stderr = {stderr}")  # type: ignore

    return stdout, stderr, rt


def get_updated_files(target: str, num_commits: int) -> List[str]:
    stdout, _, _ = _exec_subprocess(f'git -C {target} diff --name-only HEAD~{num_commits}')
    return list(filter(lambda f: f, stdout.decode().split('\n')))


def get_updated_file_stats(target: str, num_commits: int) -> Tuple[int, int, int]:
    stdout, _, _ = _exec_subprocess(f'git -C {target} diff --shortstat HEAD~{num_commits}')
    stat_summary = stdout.decode()
    m = re.search('\d+ files changed, (\d+) insertions\(\+\), (\d+) deletions\(\-\)', stdout.decode())
    num_adds = int(m.group(1))  # type: ignore
    num_dels = int(m.group(2))  # type: ignore
    return num_adds, num_dels, num_adds + num_dels


def get_latest_commit_date(target: str) -> str:
    stdout, _, _ = _exec_subprocess(f'git -C {target} log -1 --format=%cd --date=iso')
    import dateutil.parser  # type: ignore
    commit_date = dateutil.parser.parse(stdout.decode())
    return commit_date.utcnow().strftime('%Y/%m/%d %H:%M:%S')
