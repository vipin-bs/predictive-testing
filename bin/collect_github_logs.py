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


def _setup_logger(logfile: str) -> Any:
    from logging import getLogger, FileHandler, Formatter, StreamHandler, DEBUG, INFO, WARNING
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)

    formatter = Formatter('%(asctime)s.%(msecs)03d: %(message)s', '%Y-%m-%d %H:%M:%S')

    fh = FileHandler(logfile, mode='a')
    fh.setLevel(INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = StreamHandler()
    ch.setLevel(WARNING)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def _create_workflow_handlers(proj: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    if proj == 'spark':
        return spark_logs.create_spark_workflow_handlers()
    else:
        raise ValueError(f'Unknown project type: {proj}')


def _traverse_pull_requests(output_path: str, since: Optional[datetime], max_num_pullreqs: int,
                            resume: bool, params: Dict[str, str],
                            logger: Any) -> None:

    pullreq_file_path = f"{output_path}/pullreqs.json"
    resume_file_path = f"{output_path}/.process-resume.txt"

    if not resume:
        logger.info(f"Fetching candidate pull requests in {params['GITHUB_OWNER']}/{params['GITHUB_REPO']}...")
        pullreqs = github_apis.list_pullreqs(params['GITHUB_OWNER'], params['GITHUB_REPO'], params['GITHUB_TOKEN'],
                                             since=since, nmax=max_num_pullreqs, logger=logger)
        if len(pullreqs) == 0:
            raise RuntimeError('No valid pull request found')

        # Dumps all the pull request logs to resume job
        with open(pullreq_file_path, "w") as output:
            output.write(json.dumps(pullreqs))
            output.flush()
    else:
        if not os.path.exists(resume_file_path):
            raise RuntimeError(f'Resume file not found in {os.path.abspath(resume_file_path)}')

        from pathlib import Path
        processed_user_set = set(Path(resume_file_path).read_text().split('\n'))
        loaded_pullreqs = json.loads(Path(pullreq_file_path).read_text())
        pullreqs = list(filter(lambda p: p[5] not in processed_user_set, loaded_pullreqs))
        logger.info(f"{len(loaded_pullreqs)} pull requests loaded and {len(pullreqs)} ones "
                    f"filtered by {os.path.abspath(resume_file_path)}")

    # Groups pull requests by a user
    pullreqs_by_user: Dict[Tuple[str, str], List[Any]] = {}
    for pullreq in pullreqs:
        pr_user, pr_repo = pullreq[5], pullreq[6]
        if (pr_user, pr_repo) not in pullreqs_by_user:
            pullreqs_by_user[(pr_user, pr_repo)] = []

        pullreqs_by_user[(pr_user, pr_repo)].append(pullreq)

    # Generates project-dependent run/job filters and log extractor
    target_runs, target_jobs, test_failure_patterns, compilation_failure_patterns = \
        _create_workflow_handlers('spark')

    # Fetches test results from mainstream-side workflow jobs
    test_results = github_utils.get_test_results_from(params['GITHUB_OWNER'], params['GITHUB_REPO'], params,
                                                      target_runs, target_jobs,
                                                      test_failure_patterns, compilation_failure_patterns,
                                                      since=since, tqdm_leave=True,
                                                      logger=logger)

    with open(f"{output_path}/github-logs.json", "a") as of, open(resume_file_path, "a") as rf:
        # TODO: Could we parallelize crawling jobs by users?
        pb_title = f"Pull Reqests ({params['GITHUB_OWNER']}/{params['GITHUB_REPO']})"
        for (pr_user, pr_repo), pullreqs in tqdm.tqdm(pullreqs_by_user.items(), desc=pb_title):
            logger.info(f"pr_user:{pr_user}, pr_repo:{pr_repo}, #pullreqs:{len(pullreqs)}")

            # Per-user buffer to write github logs
            per_user_logs: List[Dict[str, Any]] = []

            def _write(pr_user, commit_date, commit_message, files, tests, pr_title='', pr_body=''):
                buf: Dict[str, Any] = {}
                buf['author'] = pr_user
                buf['commit_date'] = github_apis.format_github_datetime(commit_date, '%Y/%m/%d %H:%M:%S')
                buf['commit_message'] = commit_message
                buf['title'] = pr_title
                buf['body'] = pr_body
                buf['failed_tests'] = tests
                buf['files'] = []
                for file in files:
                    update_counts = github_utils.count_file_updates(
                        file['name'], commit_date, [3, 14, 56],
                        params['GITHUB_OWNER'], params['GITHUB_REPO'], params['GITHUB_TOKEN'])
                    buf['files'].append({'file': file, 'updated': update_counts})

                per_user_logs.append(buf)

            def _flush():
                for log in per_user_logs:
                    of.write(json.dumps(log))

                of.flush()

            # Fetches test results from folk-side workflow jobs
            user_test_results = github_utils.get_test_results_from(pr_user, pr_repo, params,
                                                                   target_runs, target_jobs,
                                                                   test_failure_patterns, compilation_failure_patterns,
                                                                   since=since, tqdm_leave=False,
                                                                   logger=logger)

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
                        _, _, files, tests = user_test_results[commit]
                        _write(pr_user, commit_date, commit_message, files, tests, pr_title, pr_body)
                        matched.add(commit)

                # Writes left entries into the output file
                for head_sha, (commit_date, commit_message, files, tests) in user_test_results.items():
                    if head_sha not in test_results and head_sha not in matched:
                        _write(pr_user, commit_date, commit_message, files, tests)

            _flush()

            rf.write(f"{pr_user}\n")
            rf.flush()

    # If all things done successfully, removes the resume file
    os.remove(resume_file_path)


def _to_rate_limit_msg(rate_limit: Dict[str, Any]) -> str:
    import time
    c = rate_limit['resources']['core']
    renewal = c['reset'] - int(time.time())
    return f"limit={c['limit']}, used={c['used']}, remaining={c['remaining']}, reset={renewal}s"


def _traverse_github_logs(traverse_func: Any, output_path: str, since: Optional[str], max_num_pullreqs: int,
                          resume: bool, params: Dict[str, str]) -> None:
    if len(output_path) == 0:
        raise ValueError("Output Path must be specified in '--output'")
    if len(params['GITHUB_TOKEN']) == 0:
        raise ValueError("GitHub token must be specified in '--github-token'")
    if len(params['GITHUB_OWNER']) == 0:
        raise ValueError("GitHub owner must be specified in '--github-owner'")
    if len(params['GITHUB_REPO']) == 0:
        raise ValueError("GitHub repository must be specified in '--github-repo'")

    if resume and not os.path.exists(output_path):
        raise RuntimeError(f'Output path not found in {os.path.abspath(output_path)}')
    elif not resume:
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

    traverse_func(output_path, since, max_num_pullreqs, resume, params, logger)


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
    parser.add_argument('--resume', dest='resume', action='store_true')
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
                              args.resume, params)
    else:
        _show_rate_limit(params)


if __name__ == "__main__":
    main()
