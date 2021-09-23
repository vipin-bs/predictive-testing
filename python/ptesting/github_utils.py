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
import retrying
import tqdm
from datetime import datetime, timedelta, timezone
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple

from ptesting import github_apis


# The GitHub time format (UTC)
# See: https://docs.github.com/en/rest/overview/resources-in-the-rest-api#timezones
GITHUB_DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%SZ'


def from_github_datetime(d: str) -> datetime:
    # Parses GitHub timestamp string into timezone-aware datetime
    return datetime.strptime(d, GITHUB_DATETIME_FORMAT).replace(tzinfo=timezone.utc)


def format_github_datetime(d: str, format: str) -> str:
    return from_github_datetime(d).strftime(format)


def to_github_datetime(d: datetime) -> str:
    return d.strftime(GITHUB_DATETIME_FORMAT)


def count_file_updates(path: str, base_date: str, days: List[int], owner: str, repo: str,
                       token: str, logger: Any = None) -> List[int]:
    update_counts: List[int] = []
    base = from_github_datetime(base_date)
    for day in days:
        since_date = to_github_datetime(base - timedelta(day))
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
    def extractor(logs: str) -> Optional[List[str]]:  # type: ignore
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
            for target in targets:  # type: ignore
                if name.find(target) != -1:
                    return True
            return False

        return name_filter
    else:
        def pass_thru(name: str) -> bool:
            return True

        return pass_thru


def _retry_if_except(caught: Exception) -> bool:
    if isinstance(caught, RuntimeError) and github_apis.is_rate_limit_exceeded(str(caught)):
        # If rate limit happens, do not retry
        return False
    else:
        return isinstance(caught, Exception)


@retrying.retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=4000,
                retry_on_exception=_retry_if_except,
                wrap_exception=False)
def get_test_results_from(owner: str, repo: str, token: str,
                          target_runs: Optional[List[str]],
                          target_jobs: Optional[List[str]],
                          test_failure_patterns: List[str],
                          compilation_failure_patterns: List[str],
                          until: Optional[datetime], since: Optional[datetime],
                          resume_path: str,
                          tqdm_leave: bool,
                          logger: Any) -> Dict[str, Tuple[str, str, List[Dict[str, str]], List[str]]]:
    assert os.path.exists(resume_path), "resume path '{resume_path}' does not exists"

    test_results: Dict[str, Tuple[str, str, List[Dict[str, str]], List[str]]] = {}

    # Creates filter functions based on the specified target lists
    run_filter = _create_name_filter(target_runs)
    job_filter = _create_name_filter(target_jobs)

    extract_failed_tests_from = _create_failed_test_extractor(test_failure_patterns, compilation_failure_patterns)

    import json
    from pathlib import Path
    user_resume_path = f"{resume_path}/{owner}"
    wrun_fpath = f"{user_resume_path}/workflow-runs.json"
    test_result_fpath = f"{user_resume_path}/test-results.json"
    resume_meta_fpath = f"{user_resume_path}/resume-meta.lst"
    if os.path.exists(wrun_fpath) and os.path.exists(test_result_fpath) \
            and os.path.exists(resume_meta_fpath):
        processed_wrun_set = set(Path(resume_meta_fpath).read_text().split('\n'))
        workflow_runs = list(filter(lambda r: r[0] not in processed_wrun_set, json.loads(Path(wrun_fpath).read_text())))
        test_results = json.loads(Path(test_result_fpath).read_text())
    else:
        shutil.rmtree(user_resume_path, ignore_errors=True)
        os.mkdir(user_resume_path)

        workflow_runs = []
        wruns = github_apis.list_workflow_runs(owner, repo, token, until=until, since=since, logger=logger)
        for wrun in wruns:
            run_id, run_name, _, _, conclusion, _, _, _ = wrun
            if run_filter(run_name) and conclusion in ['success', 'failure']:
                workflow_runs.append(wrun)
            else:
                logger.info(f"Run (run_id={run_id}, run_name='{run_name}') skipped")

        with open(wrun_fpath, "w") as f:
            f.write(json.dumps(workflow_runs))
            f.flush()

    with open(resume_meta_fpath, "a") as rf:
        for run_id, run_name, head_sha, event, conclusion, pr_number, head, base \
                in tqdm.tqdm(workflow_runs, desc=f"Workflow Runs ({owner}/{repo})", leave=tqdm_leave):
            logger.info(f"run_id:{run_id}, run_name:{run_name}, event:{event}, head_sha={head_sha}")

            if pr_number.isdigit():
                # List up all the updated files between 'base' and 'head' as corresponding to this run
                commit_date, commit_message, changed_files = \
                    github_apis.list_change_files_between(base, head, owner, repo, token, logger=logger)
            else:
                commit_date, commit_message, changed_files = \
                    github_apis.list_change_files_from(head_sha, owner, repo, token, logger=logger)

            files: List[Dict[str, str]] = []
            for file in changed_files:
                filename, additions, deletions, changes = file
                files.append({'name': filename, 'additions': additions, 'deletions': deletions, 'changes': changes})

            if conclusion == 'success':
                test_results[head] = (commit_date, commit_message, files, [])
            else:  # failed run case
                jobs = github_apis.list_workflow_jobs(run_id, owner, repo, token, logger=logger)
                selected_jobs: List[Tuple[str, str, str]] = []
                for job in jobs:
                    job_id, job_name, conclusion = job
                    if job_filter(job_name):
                        selected_jobs.append(job)
                    else:
                        logger.info(f"Job (run_id={run_id}, job_id={job_id}) skipped")

                all_selected_jobs_passed = len(list(filter(lambda j: j[2] == 'failure', selected_jobs))) == 0
                if all_selected_jobs_passed:
                    test_results[head] = (commit_date, commit_message, files, [])
                else:
                    failed_tests = []
                    for job_id, job_name, conclusion in selected_jobs:
                        logger.info(f"job_id:{job_id}, job_name:{job_name}, conclusion:{conclusion}")
                        if conclusion == 'failure':
                            # NOTE: In case of a compilation failure, it returns None
                            logs = github_apis.get_workflow_job_logs(job_id, owner, repo, token, logger=logger)
                            tests = extract_failed_tests_from(logs)
                            if tests is not None:
                                if len(tests) > 0:
                                    failed_tests.extend(tests)
                                else:
                                    logger.warning(f"Cannot find any test failure in workfolow job (owner={owner}, "
                                                   f"repo={repo}, run_id={run_id} job_name='{job_name}')")
                            else:
                                # If `tests` is None, it represents a compilation failure
                                logger.info(f"Compilation failure found: job_id={job_id}")

                    # If we cannot detect any failed test in logs, just ignore it
                    if len(failed_tests) > 0:
                        test_results[head] = (commit_date, commit_message, files, failed_tests)
                    else:
                        logger.info(f"No test failure found in workfolow run (owner={owner}, repo={repo}, "
                                    f"run_id={run_id} run_name='{run_name}')")

            # Writes the current snapshot of `test_results`
            with open(test_result_fpath, "w") as f:
                f.write(json.dumps(test_results))
                f.flush()

            # Writes a flag indicating run completion
            rf.write(f"{run_id}\n")
            rf.flush()

    logger.info(f"{len(test_results)} test results found in workflows ({owner}/{repo})")
    return test_results


@retrying.retry(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=4000,
                retry_on_exception=_retry_if_except,
                wrap_exception=False)
def generate_commit_logs(owner: str, repo: str, token: str,
                         until: Optional[datetime], since: Optional[datetime],
                         pullreqs: List[Any],
                         repo_test_results: Dict[str, Any],
                         user_test_results: Dict[str, Any],
                         sleep_if_limit_exceeded: bool,
                         commit_day_intervals: List[int],
                         logger: Any) -> List[Dict[str, Any]]:
    # Per-user buffer to write github logs
    per_user_logs: List[Dict[str, Any]] = []

    def _write(pr_user, sha, commit_date, commit_message, files, tests,  # type: ignore
               pr_title='', pr_body=''):
        buf: Dict[str, Any] = {}
        buf['author'] = pr_user
        buf['sha'] = sha  # Needed as a unique ID
        buf['commit_date'] = format_github_datetime(commit_date, '%Y/%m/%d %H:%M:%S')
        buf['commit_message'] = commit_message
        buf['title'] = pr_title
        buf['body'] = pr_body
        buf['failed_tests'] = tests
        buf['files'] = []
        for file in files:
            update_counts = count_file_updates(file['name'], commit_date, commit_day_intervals, owner, repo, token)
            buf['files'].append({'file': file, 'updated': update_counts})

        per_user_logs.append(buf)

    # To track which commits are used
    matched: Set[str] = set()

    for pr_number, pr_created_at, pr_updated_at, pr_title, pr_body, \
            pr_user, pr_repo, pr_branch in pullreqs:
        commits = github_apis.list_commits_for(pr_number, owner, repo, token,
                                               until=until, since=since, logger=logger)
        logger.info(f"pullreq#{pr_number} has {len(commits)} commits (created_at:{pr_created_at}, "
                    f"updated_at:{pr_updated_at})")

        for (commit, commit_date, commit_message) in commits:
            logger.info(f"commit:{commit}, commit_date:{commit_date}")
            if commit in user_test_results:
                _, _, files, tests = user_test_results[commit]
                _write(pr_user, commit, commit_date, commit_message, files, tests, pr_title, pr_body)
                matched.add(commit)

    # Writes left entries into the output file
    for head_sha, (commit_date, commit_message, files, tests) in user_test_results.items():
        if head_sha not in repo_test_results and head_sha not in matched:
            _write(pr_user, head_sha, commit_date, commit_message, files, tests)

    return per_user_logs


def get_rate_limit(github_token: str) -> Tuple[int, int, int, int]:
    import time
    rate_limit = github_apis.get_rate_limit(github_token)
    c = rate_limit['resources']['core']
    renewal = c['reset'] - int(time.time())
    return c['limit'], c['used'], c['remaining'], renewal


def trim_text(s: str, max_num: int) -> str:
    return s[0:max_num] + '...' if len(s) > max_num else s
