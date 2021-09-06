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
import tqdm
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone

import github_apis
import github_utils
import spark_logs


# Suppress warinig messages in REST APIs
warnings.simplefilter('ignore')


# NOTE: In case of a compilation failure, it returns None
def _get_failed_tests(pr_user: str, pr_repo: str, job_name: str, job_id: str,
                      extract_failed_tests_from: Any,
                      params: Dict[str, str],
                      logger: Any) -> Optional[List[str]]:
   logs = github_apis.get_workflow_job_logs(job_id, pr_user, pr_repo, params['GITHUB_TOKEN'], logger=logger)
   return extract_failed_tests_from(logs)


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


def _get_test_results_from(owner: str, repo: str, params: Dict[str, str],
                           target_runs: Optional[List[str]], target_jobs: Optional[List[str]],
                           extract_failed_tests_from: Any,
                           since: Optional[datetime],
                           logger: Any) -> Dict[str, Tuple[str, str, List[Dict[str, str]], List[str]]]:
    test_results: Dict[str, Tuple[str, str, List[Dict[str, str]], List[str]]] = {}

    # Creates filter functions based on the specified target lists
    run_filter = _create_name_filter(target_runs)
    job_filter = _create_name_filter(target_jobs)

    runs = github_apis.list_workflow_runs(owner, repo, params['GITHUB_TOKEN'], since=since, logger=logger)
    for run_id, run_name, head_sha, event, conclusion, pr_number, head, base in tqdm.tqdm(runs, desc=f"Workflow Runs ({owner}/{repo})", leave=False):
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
                            tests = _get_failed_tests(owner, repo, job_name, job_id, extract_failed_tests_from,
                                                      params, logger=logger)
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


def _create_workflow_handlers(proj: str) -> Tuple[Any, Any, Any]:
    if proj == 'spark':
        return spark_logs.create_spark_workflow_handlers()
    else:
        raise ValueError(f'Unknown project type: {proj}')


def _to_rate_limit_msg(rate_limit: Dict[str, Any]) -> str:
    import time
    c = rate_limit['resources']['core']
    renewal = c['reset'] - int(time.time())
    return f"limit={c['limit']}, used={c['used']}, remaining={c['remaining']}, reset={renewal}s"


def _setup_logger(logfile: str) -> Any:
    from logging import getLogger, FileHandler, Formatter, StreamHandler, DEBUG, INFO, WARNING
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)

    formatter = Formatter('%(asctime)s.%(msecs)03d: %(message)s', '%Y-%m-%d %H:%M:%S')

    fh = FileHandler(logfile)
    fh.setLevel(INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = StreamHandler()
    ch.setLevel(WARNING)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def _traverse_pull_requests(output_path: str, since: Optional[datetime], max_num_pullreqs: int, params: Dict[str, str], logger: Any) -> None:
    logger.info(f"Fetching candidate pull requests in {params['GITHUB_OWNER']}/{params['GITHUB_REPO']}...")
    pullreqs = github_apis.list_pullreqs(params['GITHUB_OWNER'], params['GITHUB_REPO'], params['GITHUB_TOKEN'],
                                         since=since, nmax=max_num_pullreqs, logger=logger)
    if len(pullreqs) == 0:
        raise RuntimeError('No valid pull request found')

    # Dumps all the pull request logs to resume job
    with open(f"{output_path}/pullreqs.json", "w") as output:
        output.write(json.dumps(pullreqs))
        output.flush()

    # Groups pull requests by a user
    pullreqs_by_user: Dict[Tuple[str, str], List[Any]] = {}
    for pullreq in pullreqs:
        pr_user, pr_repo = pullreq[5], pullreq[6]
        if (pr_user, pr_repo) not in pullreqs_by_user:
            pullreqs_by_user[(pr_user, pr_repo)] = []

        pullreqs_by_user[(pr_user, pr_repo)].append(pullreq)

    # Generates project-dependent run/job filters and log extractor
    target_runs, target_jobs, extract_failed_tests_from = _create_workflow_handlers('spark')

    # Fetches test results from mainstream-side workflow jobs
    test_results = _get_test_results_from(params['GITHUB_OWNER'], params['GITHUB_REPO'], params,
                                          target_runs, target_jobs, extract_failed_tests_from,
                                          since=since, logger=logger)

    with open(f"{output_path}/github-logs.json", "w") as output:

        def _to_github_log(pr_user, commit_date, commit_message, files, tests, pr_title='', pr_body=''):
            buf: Dict[str, Any] = {}

            buf['author'] = pr_user
            buf['commit_date'] = github_apis.format_github_datetime(commit_date, '%Y/%m/%d %H:%M:%S')
            buf['commit_message'] = commit_message
            buf['title'] = pr_title
            buf['body'] = pr_body
            buf['files'] = []

            for file in files:
                update_counts = github_utils.count_file_updates(
                    file['name'], commit_date, [3, 14, 56],
                    params['GITHUB_OWNER'], params['GITHUB_REPO'], params['GITHUB_TOKEN'])
                buf['files'].append({'file': file, 'updated': update_counts})

            buf['failed_tests'] = tests
            return buf

        pb_title = f"Pull Reqests ({params['GITHUB_OWNER']}/{params['GITHUB_REPO']})"
        # TODO: Could we parallelize crawling jobs by users?
        for (pr_user, pr_repo), pullreqs in tqdm.tqdm(pullreqs_by_user.items(), desc=pb_title):
            logger.info(f"pr_user:{pr_user}, pr_repo:{pr_repo}, #pullreqs:{len(pullreqs)}")

            # Fetches test results from folk-side workflow jobs
            user_test_results = _get_test_results_from(pr_user, pr_repo, params,
                                                       target_runs, target_jobs, extract_failed_tests_from,
                                                       since=since, logger=logger)

            # Merges the tests results with mainstream's ones
            user_test_results.update(test_results)

            for pr_number, pr_created_at, pr_updated_at, pr_title, pr_body, pr_user, pr_repo, pr_branch in pullreqs:
                commits = github_apis.list_commits_for(pr_number, params['GITHUB_OWNER'], params['GITHUB_REPO'], params['GITHUB_TOKEN'],
                                                       since=None, logger=logger)
                logger.info(f"pullreq#{pr_number} has {len(commits)} commits (created_at:{pr_created_at}, updated_at:{pr_updated_at})")

                matched: Set[str] = set()
                for (commit, commit_date, commit_message) in commits:
                    logger.info(f"commit:{commit}, commit_date:{commit_date}")
                    if commit in user_test_results:
                        matched.add(commit)

                        _, _, files, tests = user_test_results[commit]
                        data = _to_github_log(pr_user, commit_date, commit_message, files, tests, pr_title, pr_body)
                        output.write(json.dumps(data))
                        output.flush()

                for head_sha, (commit_date, commit_message, files, tests) in user_test_results.items():
                    if head_sha not in test_results and head_sha not in matched:
                        data = _to_github_log(pr_user, commit_date, commit_message, files, tests)
                        output.write(json.dumps(data))
                        output.flush()


def _traverse_github_logs(traverse_func: Any, output_path: str, since: Optional[str], max_num_pullreqs: int, params: Dict[str, str]) -> None:
    if len(output_path) == 0:
        raise ValueError("Output Path must be specified in '--output'")
    if len(params['GITHUB_TOKEN']) == 0:
        raise ValueError("GitHub token must be specified in '--github-token'")
    if len(params['GITHUB_OWNER']) == 0:
        raise ValueError("GitHub owner must be specified in '--github-owner'")
    if len(params['GITHUB_REPO']) == 0:
        raise ValueError("GitHub repository must be specified in '--github-repo'")

    # Make an output dir in advance
    os.mkdir(output_path)

    # For logger setup
    logger = _setup_logger(f'{output_path}/debug-info.log')

    # logger rate limit
    logger.info(f"rate_limit: {_to_rate_limit_msg(github_apis.get_rate_limit(params['GITHUB_TOKEN']))}")

    # Parses a specified datetime string if possible
    import dateutil.parser as parser
    if since is not None:
        since = parser.parse(since)
        logger.info(f"Target timestamp: since={github_apis.to_github_datetime(since)} "
                    f"until={github_apis.to_github_datetime(datetime.now(timezone.utc))}")

    traverse_func(output_path, since, max_num_pullreqs, params, logger)


def _show_rate_limit(params: Dict[str, str]) -> None:
    rate_limit = github_apis.get_rate_limit(params['GITHUB_TOKEN'])
    print('======== GitHub Rate Limit ========')
    print(_to_rate_limit_msg(rate_limit))


def main():
    # Parses command-line arguments
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--output', dest='output', type=str, default='')
    parser.add_argument('--max-num-pullreqs', dest='max_num_pullreqs', type=int, default=100000)
    parser.add_argument('--since', dest='since', type=str)
    parser.add_argument('--github-token', dest='github_token', type=str, default='')
    parser.add_argument('--github-owner', dest='github_owner', type=str, default='')
    parser.add_argument('--github-repo', dest='github_repo', type=str, default='')
    parser.add_argument('--show-rate-limit', dest='show_rate_limit', action='store_true')
    args = parser.parse_args()

    params = {
        "GITHUB_TOKEN": args.github_token,
        "GITHUB_OWNER": args.github_owner,
        "GITHUB_REPO": args.github_repo
    }

    if not args.show_rate_limit:
        _traverse_github_logs(_traverse_pull_requests,
                              args.output, args.since, args.max_num_pullreqs,
                              params)
    else:
        _show_rate_limit(params)


if __name__ == "__main__":
    main()
