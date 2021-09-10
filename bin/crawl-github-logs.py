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
import shutil
import time
import tqdm
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone

from ptesting import github_apis
from ptesting import github_utils
from ptesting import spark_logs


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


def _get_rate_limit(github_token: str) -> Tuple[int, int, int, int]:
    rate_limit = github_apis.get_rate_limit(github_token)
    c = rate_limit['resources']['core']
    renewal = c['reset'] - int(time.time())
    return c['limit'], c['used'], c['remaining'], renewal


def _rate_limit_msg(github_token: str) -> str:
    limit, used, remaining, reset = _get_rate_limit(github_token)
    return f"limit={limit}, used={used}, remaining={remaining}, reset={reset}s"


def _traverse_pull_requests(output_path: str,
                            until: Optional[datetime], since: Optional[datetime],
                            max_num_pullreqs: int, resume: bool,
                            sleep_if_limit_exceeded: bool,
                            params: Dict[str, str],
                            logger: Any) -> None:

    owner = params['GITHUB_OWNER']
    repo = params['GITHUB_REPO']
    token = params['GITHUB_TOKEN']

    # List of output file paths
    run_meta_fpath = f"{output_path}/.run-meta.json"
    resume_meta_fpath = f"{output_path}/.resume-meta.lst"
    pullreq_fpath = f"{output_path}/pullreqs.json"

    # Makes a resume dir for workflow runs if it does not exists
    wrun_resume_path = f"{output_path}/.resume-workflow-runs"
    if not os.path.exists(wrun_resume_path):
        os.mkdir(wrun_resume_path)

    if not resume:
        logger.info(f"Fetching candidate pull requests in {owner}/{repo}...")
        pullreqs = github_apis.list_pullreqs(owner, repo, token,
                                             until=until, since=since,
                                             nmax=max_num_pullreqs,
                                             logger=logger)
        if len(pullreqs) == 0:
            raise RuntimeError('No valid pull request found')

        with open(run_meta_fpath, "w") as f:
            meta: Dict[str, str] = {}
            meta['owner'] = owner
            meta['repo'] = repo
            meta['until'] = github_utils.to_github_datetime(datetime.now(timezone.utc))
            if since is not None:
                meta['since'] = github_utils.to_github_datetime(since)
            f.write(json.dumps(meta))
            f.flush()
        with open(pullreq_fpath, "w") as f:
            f.write(json.dumps(pullreqs))
            f.flush()
    else:
        if not os.path.exists(run_meta_fpath):
            raise RuntimeError(f'Run meta file not found in {os.path.abspath(run_meta_fpath)}')
        if not os.path.exists(resume_meta_fpath):
            raise RuntimeError(f'Resume file not found in {os.path.abspath(resume_meta_fpath)}')

        from pathlib import Path
        run_meta = json.loads(Path(run_meta_fpath).read_text())
        owner = run_meta['owner']
        repo = run_meta['repo']
        until = github_utils.from_github_datetime(run_meta['until'])
        since = github_utils.from_github_datetime(run_meta['since']) if 'since' in run_meta else None
        processed_user_set = set(Path(resume_meta_fpath).read_text().split('\n'))
        pullreqs = list(filter(lambda p: p[5] not in processed_user_set, json.loads(Path(pullreq_fpath).read_text())))
        logger.info("Resuming process: owner={}, repo={}, #pullreqs={}, until={}{}".format(
            owner, repo, len(pullreqs), run_meta['until'], f", since={run_meta['since']}" if since else ""))

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

    try:
        with open(f"{output_path}/github-logs.json", "a") as of, open(resume_meta_fpath, "a") as rf:
            # Fetches test results from mainstream-side workflow jobs
            test_results = github_utils.get_test_results_from(owner, repo, params,
                                                              target_runs, target_jobs,
                                                              test_failure_patterns, compilation_failure_patterns,
                                                              until=until, since=since,
                                                              resume_path=wrun_resume_path,
                                                              tqdm_leave=True,
                                                              logger=logger)

            # TODO: Could we parallelize crawling jobs by users?
            pb_title = f"Pull Reqests ({owner}/{repo})"
            for (pr_user, pr_repo), pullreqs in tqdm.tqdm(pullreqs_by_user.items(), desc=pb_title):
                logger.info(f"pr_user:{pr_user}, pr_repo:{pr_repo}, #pullreqs:{len(pullreqs)}")

                finished = False
                while not finished:
                    try:
                        # Per-user buffer to write github logs
                        per_user_logs: List[Dict[str, Any]] = []

                        def _write(pr_user, commit_date, commit_message, files, tests,  # type: ignore
                                   pr_title='', pr_body=''):
                            buf: Dict[str, Any] = {}
                            buf['author'] = pr_user
                            buf['commit_date'] = github_utils.format_github_datetime(commit_date, '%Y/%m/%d %H:%M:%S')
                            buf['commit_message'] = commit_message
                            buf['title'] = pr_title
                            buf['body'] = pr_body
                            buf['failed_tests'] = tests
                            buf['files'] = []
                            for file in files:
                                update_counts = github_utils.count_file_updates(
                                    file['name'], commit_date, [3, 14, 56],
                                    owner, repo, token)
                                buf['files'].append({'file': file, 'updated': update_counts})

                            per_user_logs.append(buf)

                        def _flush() -> None:
                            for log in per_user_logs:
                                of.write(json.dumps(log))

                            of.flush()

                        # Fetches test results from folk-side workflow jobs
                        user_test_results = github_utils.get_test_results_from(pr_user, pr_repo, params,
                                                                               target_runs, target_jobs,
                                                                               test_failure_patterns,
                                                                               compilation_failure_patterns,
                                                                               until=until, since=since,
                                                                               resume_path=wrun_resume_path,
                                                                               tqdm_leave=False,
                                                                               logger=logger)

                        # Merges the tests results with mainstream's ones
                        user_test_results.update(test_results)

                        for pr_number, pr_created_at, pr_updated_at, pr_title, pr_body, \
                                pr_user, pr_repo, pr_branch in pullreqs:
                            commits = github_apis.list_commits_for(pr_number, owner, repo, token,
                                                                   until=until, since=since, logger=logger)
                            logger.info(f"pullreq#{pr_number} has {len(commits)} commits (created_at:{pr_created_at}, "
                                        f"updated_at:{pr_updated_at})")

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

                    except RuntimeError as e:
                        if sleep_if_limit_exceeded and str(e).find('API rate limit exceeded') != -1:
                            _, _, _, renewal = _get_rate_limit(params['GITHUB_TOKEN'])
                            logger.info(f"API rate limit exceeded, so this process sleeps for {renewal}s")
                            time.sleep(renewal + 16)
                        else:
                            raise e
                    else:
                        # Proceeds to the next
                        finished = True

                _flush()

                # Writes a flag indicating run completion
                rf.write(f"{pr_user}\n")
                rf.flush()

    except Exception as e:
        logger.info(f"{e.__class__}: {e}")
        logger.warning("Crawling logs failed, but you can resume it by '--resume' option")

    else:
        # If all things done successfully, removes the resume file/dir
        shutil.rmtree(wrun_resume_path, ignore_errors=True)
        os.remove(resume_meta_fpath)


def _traverse_github_logs(traverse_func: Any, output_path: str, overwrite: bool,
                          until: Optional[str], since: Optional[str],
                          max_num_pullreqs: int, resume: bool, sleep_if_limit_exceeded: bool,
                          params: Dict[str, str]) -> None:
    if len(output_path) == 0:
        raise ValueError("Output Path must be specified in '--output'")
    if len(params['GITHUB_TOKEN']) == 0:
        raise ValueError("GitHub token must be specified in '--github-token'")
    if not resume and len(params['GITHUB_OWNER']) == 0:
        raise ValueError("GitHub owner must be specified in '--github-owner'")
    if not resume and len(params['GITHUB_REPO']) == 0:
        raise ValueError("GitHub repository must be specified in '--github-repo'")

    if resume and not os.path.exists(output_path):
        raise RuntimeError(f'Output path not found in {os.path.abspath(output_path)}')
    elif not resume:
        if overwrite:
            shutil.rmtree(output_path, ignore_errors=True)

        # Make an output dir in advance
        os.mkdir(output_path)

    # For logger setup
    logger = _setup_logger(f'{output_path}/debug-info.log')

    # logger rate limit
    logger.info(f"rate_limit: {_rate_limit_msg(params['GITHUB_TOKEN'])}")

    # Parses a specified datetime string if necessary
    import dateutil.parser as parser  # type: ignore
    until = parser.parse(until) if until else None
    since = parser.parse(since) if since else None

    traverse_func(output_path, until, since, max_num_pullreqs, resume,
                  sleep_if_limit_exceeded, params, logger)


def _show_rate_limit(params: Dict[str, str]) -> None:
    print('======== GitHub Rate Limit ========')
    print(_rate_limit_msg(params['GITHUB_TOKEN']))


def main() -> None:
    # Parses command-line arguments
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--output', dest='output', type=str, default='')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    parser.add_argument('--max-num-pullreqs', dest='max_num_pullreqs', type=int, default=100000)
    parser.add_argument('--until', dest='until', type=str)
    parser.add_argument('--since', dest='since', type=str)
    parser.add_argument('--github-token', dest='github_token', type=str, default='')
    parser.add_argument('--github-owner', dest='github_owner', type=str, default='')
    parser.add_argument('--github-repo', dest='github_repo', type=str, default='')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--sleep-if-limit-exceeded', dest='sleep_if_limit_exceeded', action='store_true')
    parser.add_argument('--show-rate-limit', dest='show_rate_limit', action='store_true')
    args = parser.parse_args()

    params = {
        "GITHUB_TOKEN": args.github_token,
        "GITHUB_OWNER": args.github_owner,
        "GITHUB_REPO": args.github_repo
    }

    if not args.show_rate_limit:
        _traverse_github_logs(_traverse_pull_requests, args.output, args.overwrite,
                              args.until, args.since,
                              args.max_num_pullreqs, args.resume,
                              args.sleep_if_limit_exceeded,
                              params)
    else:
        _show_rate_limit(params)


if __name__ == "__main__":
    main()
