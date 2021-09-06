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

import github_apis

import datetime
from typing import Any, List, Optional


def count_file_updates(path: str, base_date: str, days: List[int], owner: str, repo: str,
                       token: str, logger: Any = None) -> List[int]:
    update_counts: List[int] = []
    base = github_apis.from_github_datetime(base_date)
    for day in days:
        since_date = github_apis.to_github_datetime(base - datetime.timedelta(day))
        file_commits = github_apis.list_repo_commits(
            owner, repo, token, path=path, since=since_date, until=base_date,
            logger=logger)
        update_counts.append(len(file_commits))

    return update_counts


# Generates an extractor for failed tests from specified regex patterns
def create_failed_test_extractor(test_failure_patterns: List[str],
                                 compilation_failure_patterns: Optional[List[str]] = None) -> Any:
    import re
    test_failures = list(map(lambda p: re.compile(p), test_failure_patterns))
    if compilation_failure_patterns is not None:
        compilation_failures = list(map(lambda p: re.compile(p), compilation_failure_patterns))

    # TODO: It seems to be better to search the specified patterns from a tail to a head because
    # the strings that represents build results (e.g., compilation/test failures)
    # are likely to be placed in the end of logs.
    def extractor(logs: str) -> Optional[List[str]]:
        if compilation_failure_patterns is not None:
            for p in compilation_failures:
                if p.search(logs) is not None:
                    # 'None' represents a compilation failure
                    return None

            failed_tests: List[str] = []
            for p in test_failures:
                failed_tests.extend(p.findall(logs))

            return failed_tests

    return extractor


def trim_text(s: str, max_num: int) -> str:
    return s[0:max_num] + '...' if len(s) > max_num else s
