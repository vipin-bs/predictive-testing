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

import logging
import json
import os
import tqdm
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone

import github_apis
import github_features
import github_spark_logs


# Suppress warinig messages in REST APIs
warnings.simplefilter('ignore')


def _trim_text(s: str, max_num: int) -> str:
    return s[0:max_num] + '...' if len(s) > max_num else s


def _get_failed_tests(pr_user: str, pr_repo: str, job_name: str, job_id: str,
                      extract_failed_tests_from: Any,
                      params: Dict[str, str]) -> List[str]:
   logs = github_apis.get_workflow_job_logs(job_id, pr_user, pr_repo, params['GITHUB_TOKEN'])
   return extract_failed_tests_from(logs)


def _get_test_results_from(pr_user: str, pr_repo: str, params: Dict[str, str],
                           run_filter: Any, job_filter: Any, extract_failed_tests_from: Any,
                           since: Optional[datetime] = None) -> Dict[str, Any]:
    test_results: Dict[str, Tuple[List[str], List[str]]] = {}

    runs = github_apis.list_workflow_runs(pr_user, pr_repo, params['GITHUB_TOKEN'], since=since)
    for run_id, run_name, event, pr_number, head, base in tqdm.tqdm(runs, desc=f"Workflow Runs ({pr_user}/{pr_repo})"):
        logging.info(f"run_id:{run_id}, pr_number:{pr_number}, run_name:{run_name}")

        if not run_filter(run_name):
            logging.info(f"Run (run_id:{run_id}, run_name:'{run_name}') skipped")
        else:
            # List up all the updated files between 'base' and 'head' as corresponding to this run
            files = github_apis.list_change_files(base, f"{pr_user}:{head}", pr_user, pr_repo, params['GITHUB_TOKEN'])

            jobs = github_apis.list_workflow_jobs(run_id, pr_user, pr_repo, params['GITHUB_TOKEN'])
            selected_jobs: List[Tuple[str, str, str]] = []
            for job in jobs:
                job_id, job_name, conclusion = job
                if not job_filter(job_name):
                    logging.info(f"Job (run_id/job_id:{job_id}/{run_id}, name:'{run_name}':'{job_name}') skipped")
                else:
                    selected_jobs.append(job)

            tests_passed = len(list(filter(lambda j: j[2] == 'failure', selected_jobs))) == 0
            if tests_passed:
                test_results[head] = (files, [])
            else:  # failed case
                failed_tests = []
                for job_id, job_name, conclusion in selected_jobs:
                    logging.info(f"job_id:{job_id}, job_name:{job_name}, conclusion:{conclusion}")
                    if conclusion == 'failure':
                        tests = _get_failed_tests(pr_user, pr_repo, job_name, job_id, extract_failed_tests_from, params)
                        failed_tests.extend(tests)

                # If we cannot detect any failed test in logs, just ignore it
                if len(failed_tests) > 0:
                    test_results[head] = (files, failed_tests)
                else:
                    logging.warning(f"No test failure found: run_id={run_id} run_name='{run_name}'")

    return test_results


def _create_workflow_handlers(proj: str) -> Tuple[Any, Any, Any]:
    if proj == 'spark':
        return github_spark_logs.create_spark_workflow_handlers()
    else:
        raise ValueError(f'Unknown project type: {proj}')


def _traverse_pull_requests(output_path: str, since: Optional[str], max_num_pullreqs: int, params: Dict[str, str]) -> None:
    # Make an output dir in advance
    os.mkdir(output_path)

    # For logging setup
    logging.basicConfig(
        filename=f'{output_path}/debug-info.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # Parses a specified datetime string if possible
    import dateutil.parser as parser
    if since is not None:
        since = parser.parse(since)
        logging.info(f"Target timestamp: since={github_apis.to_github_datetime(since)} "
                     f"until={github_apis.to_github_datetime(datetime.now(timezone.utc))}")

    logging.info(f"Fetching all pull requests in {params['GITHUB_OWNER']}/{params['GITHUB_REPO']}...")
    pullreqs = github_apis.list_pullreqs(params['GITHUB_OWNER'], params['GITHUB_REPO'], params['GITHUB_TOKEN'], since=since, nmax=max_num_pullreqs)
    if len(pullreqs) == 0:
        raise Exception('No valid pull request found')

    # Grouping pull requests by a user
    pullreqs_by_user: Dict[Tuple[str, str], List[Any]] = {}
    for pullreq in pullreqs:
        pr_user, pr_repo = pullreq[5], pullreq[6]
        if (pr_user, pr_repo) not in pullreqs_by_user:
            pullreqs_by_user[(pr_user, pr_repo)] = []

        pullreqs_by_user[(pr_user, pr_repo)].append(pullreq)

    # Generates project-dependent run/job filters and log extractor
    run_filter, job_filter, extract_failed_tests_from = _create_workflow_handlers('spark')

    with open(f"{output_path}/github-logs.json", "w") as output:
        pb_title = f"Pull Reqests ({params['GITHUB_OWNER']}/{params['GITHUB_REPO']})"
        for (pr_user, pr_repo), pullreqs in tqdm.tqdm(pullreqs_by_user.items(), desc=pb_title):
            logging.info(f"pr_user:{pr_user}, pr_repo:{pr_repo}, #pullreqs:{len(pullreqs)}")

            # Fetches test results from workflow jobs
            test_results = _get_test_results_from(pr_user, pr_repo, params,
                                                  run_filter, job_filter, extract_failed_tests_from,
                                                  since=since)

            for pr_number, pr_created_at, pr_updated_at, pr_title, pr_body, pr_user, pr_repo, pr_branch in pullreqs:
                if pr_repo != '':
                    commits = github_apis.list_commits_for(pr_number, params['GITHUB_OWNER'], params['GITHUB_REPO'], params['GITHUB_TOKEN'],
                                                           since=since)
                    logging.info(f"pullreq#{pr_number} has {len(commits)} commits")

                    for (commit, commit_date) in commits:
                        if commit in test_results:
                            buf: Dict[str, Any] = {}
                            buf['author'] = pr_user
                            buf['commit_date'] = github_apis.format_github_datetime(commit_date, '%Y/%m/%d %H:%M:%S')
                            buf['title'] = pr_title
                            buf['body'] = pr_body
                            buf['files'] = []
                            (files, tests) = test_results[commit]
                            for file in files:
                                update_counts = github_features.count_file_updates(
                                    file, commit_date, [3, 14, 56],
                                    params['GITHUB_OWNER'], params['GITHUB_REPO'], params['GITHUB_TOKEN'])
                                buf['files'].append({'file': file, 'updated': update_counts})

                            buf['failed_tests'] = tests
                            output.write(json.dumps(buf))
                            output.write("\n")
                            output.flush()


def main():
    # Parses command-line arguments
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--output', dest='output', type=str, required=True)
    parser.add_argument('--max-num-pullreqs', dest='max_num_pullreqs', type=int, default=100000)
    parser.add_argument('--since', dest='since', type=str)
    parser.add_argument('--github-token', dest='github_token', type=str, required=True)
    parser.add_argument('--github-owner', dest='github_owner', type=str, required=True)
    parser.add_argument('--github-repo', dest='github_repo', type=str, required=True)
    args = parser.parse_args()

    params = {
        "GITHUB_TOKEN": args.github_token,
        "GITHUB_OWNER": args.github_owner,
        "GITHUB_REPO": args.github_repo
    }

    _traverse_pull_requests(args.output, args.since, args.max_num_pullreqs, params)


if __name__ == "__main__":
    main()
