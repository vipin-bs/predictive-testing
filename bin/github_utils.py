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

import tqdm
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import github_apis

def count_file_updates(path: str, base_date: str, days: List[int], owner: str, repo: str,
                       token: str, logger: Any = None) -> List[int]:
    update_counts: List[int] = []
    base = github_apis.from_github_datetime(base_date)
    for day in days:
        since_date = github_apis.to_github_datetime(base - timedelta(day))
        file_commits = github_apis.list_repo_commits(
            owner, repo, token, path=path, since=since_date, until=base_date,
            logger=logger)
        update_counts.append(len(file_commits))

    return update_counts


# Generates an extractor for failed tests from specified regex patterns
def _create_failed_test_extractor(test_failure_patterns: List[str],
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


def _create_name_filter(targets: Optional[List[str]]) -> Any:
    if targets is not None:
        def name_filter(name: str) -> bool:
            for target in targets:
              if name.find(target) != -1:
                  return True
            return False

        return name_filter
    else:
        def pass_thru(name: str) -> bool:
            return True

        return pass_thru


def get_test_results_from(owner: str, repo: str, params: Dict[str, str],
                          target_runs: Optional[List[str]],
                          target_jobs: Optional[List[str]],
                          test_failure_patterns: List[str],
                          compilation_failure_patterns: List[str],
                          until: Optional[datetime], since: Optional[datetime],
                          tqdm_leave: bool,
                          logger: Any) -> Dict[str, Tuple[str, str, List[Dict[str, str]], List[str]]]:
    test_results: Dict[str, Tuple[str, str, List[Dict[str, str]], List[str]]] = {}

    # Creates filter functions based on the specified target lists
    run_filter = _create_name_filter(target_runs)
    job_filter = _create_name_filter(target_jobs)

    extract_failed_tests_from = _create_failed_test_extractor(test_failure_patterns, compilation_failure_patterns)

    runs = github_apis.list_workflow_runs(owner, repo, params['GITHUB_TOKEN'], until=until, since=since, logger=logger)
    for run_id, run_name, head_sha, event, conclusion, pr_number, head, base in tqdm.tqdm(runs, desc=f"Workflow Runs ({owner}/{repo})", leave=tqdm_leave):
        logger.info(f"run_id:{run_id}, run_name:{run_name}, event:{event}, head_sha={head_sha}")

        if run_filter(run_name) and conclusion in ['success', 'failure']:
            if pr_number.isdigit():
                # List up all the updated files between 'base' and 'head' as corresponding to this run
                commit_date, commit_message, changed_files = \
                    github_apis.list_change_files_between(base, head, owner, repo, params['GITHUB_TOKEN'], logger=logger)
            else:
                commit_date, commit_message, changed_files = \
                    github_apis.list_change_files_from(head_sha, owner, repo, params['GITHUB_TOKEN'], logger=logger)

            files: List[Dict[str, str]] = []
            for file in changed_files:
                filename, additions, deletions, changes = file
                files.append({'name': filename, 'additions': additions, 'deletions': deletions, 'changes': changes})

            if conclusion == 'success':
                test_results[head] = (commit_date, commit_message, files, [])
            else:  # failed run case
                jobs = github_apis.list_workflow_jobs(run_id, owner, repo, params['GITHUB_TOKEN'], logger=logger)
                selected_jobs: List[Tuple[str, str, str]] = []
                for job in jobs:
                    job_id, job_name, conclusion = job
                    if job_filter(job_name):
                        selected_jobs.append(job)
                    else:
                        logger.info(f"Job (run_id/job_id:{job_id}/{run_id}, name:'{run_name}':'{job_name}') skipped")

                all_selected_jobs_passed = len(list(filter(lambda j: j[2] == 'failure', selected_jobs))) == 0
                if all_selected_jobs_passed:
                    test_results[head] = (commit_date, commit_message, files, [])
                else:
                    failed_tests = []
                    for job_id, job_name, conclusion in selected_jobs:
                        logger.info(f"job_id:{job_id}, job_name:{job_name}, conclusion:{conclusion}")
                        if conclusion == 'failure':
                            # NOTE: In case of a compilation failure, it returns None
                            logs = github_apis.get_workflow_job_logs(job_id, owner, repo, params['GITHUB_TOKEN'], logger=logger)
                            tests = extract_failed_tests_from(logs)
                            if tests is not None:
                                if len(tests) > 0:
                                    failed_tests.extend(tests)
                                else:
                                    logger.warning(f"Cannot find any test failure in workfolow job (owner={owner}, repo={repo}, "
                                                   f"run_id={run_id} job_name='{job_name}')")
                            else:
                                # If `tests` is None, it represents a compilation failure
                                logger.info(f"Compilation failure found: job_id={job_id}")

                    # If we cannot detect any failed test in logs, just ignore it
                    if len(failed_tests) > 0:
                        test_results[head] = (commit_date, commit_message, files, failed_tests)
                    else:
                        logger.info(f"No test failure found in workfolow run (owner={owner}, repo={repo}, "
                                    f"run_id={run_id} run_name='{run_name}')")

        else:
            logger.info(f"Run (run_id:{run_id}, run_name:'{run_name}', event={event}, conclusion={conclusion}) skipped")

    logger.info(f"{len(test_results)} test results found in workflows ({owner}/{repo})")
    return test_results


def trim_text(s: str, max_num: int) -> str:
    return s[0:max_num] + '...' if len(s) > max_num else s
