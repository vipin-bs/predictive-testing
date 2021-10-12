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

import dateutil  # type: ignore
import json
import os
import shutil
import tqdm
import warnings
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

import spark_logs
from ptesting import github_apis
from ptesting import github_utils


# Suppress warinig messages in REST APIs
warnings.simplefilter('ignore')


def _setup_logger(logfile: str) -> Any:
    from logging import getLogger, FileHandler, Formatter, StreamHandler, DEBUG, ERROR, INFO
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)

    formatter = Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(module)s: %(message)s', '%Y-%m-%d %H:%M:%S')

    fh = FileHandler(logfile, mode='a')
    fh.setLevel(INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = StreamHandler()
    ch.setLevel(ERROR)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def _create_workflow_handlers(proj: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    if proj == 'spark':
        return spark_logs.create_spark_workflow_handlers()
    else:
        raise ValueError(f'Unknown project type: {proj}')


def _rate_limit_msg(github_token: str) -> str:
    limit, used, remaining, reset = github_utils.get_rate_limit(github_token)
    return f"limit={limit}, used={used}, remaining={remaining}, reset={reset}s"


def _traverse_pull_requests(output_path: str,
                            owner: str, repo: str, token: str,
                            until: Optional[datetime], since: Optional[datetime],
                            max_num_pullreqs: int, resume: bool,
                            sleep_if_limit_exceeded: bool,
                            logger: Any) -> None:
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
            repo_test_results = github_utils.get_test_results_from(owner, repo, token,
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
                        # Fetches test results from folk-side workflow jobs
                        user_test_results = github_utils.get_test_results_from(pr_user, pr_repo, token,
                                                                               target_runs, target_jobs,
                                                                               test_failure_patterns,
                                                                               compilation_failure_patterns,
                                                                               until=until, since=since,
                                                                               resume_path=wrun_resume_path,
                                                                               tqdm_leave=False,
                                                                               logger=logger)

                        # Merges the tests results with mainstream's repository ones
                        user_test_results.update(repo_test_results)
                        per_user_logs = github_utils.generate_commit_logs(owner, repo, token, until, since,
                                                                          pullreqs, repo_test_results,
                                                                          user_test_results,
                                                                          sleep_if_limit_exceeded,
                                                                          commit_day_intervals=[3, 14, 56],
                                                                          logger=logger)

                        for log in per_user_logs:
                            of.write(json.dumps(log))
                            of.write("\n")

                        of.flush()

                    except RuntimeError as e:
                        if sleep_if_limit_exceeded and github_apis.is_rate_limit_exceeded(str(e)):
                            import time
                            _, _, _, renewal = github_utils.get_rate_limit(token)
                            logger.info(f"API rate limit exceeded, so this process sleeps for {renewal}s")
                            time.sleep(renewal + 4)
                        elif github_apis.is_not_found(str(e)):
                            logger.warning(f"Request (pr_user:{pr_user}, pr_repo:{pr_repo}, "
                                           f"#pullreqs:{len(pullreqs)}) skipped")
                            finished = True
                        else:
                            raise
                    else:
                        finished = True

                # Writes a flag indicating run completion
                rf.write(f"{pr_user}\n")
                rf.flush()

    except Exception as e:
        logger.info(f"{e.__class__}: {e}")
        logger.error("Crawling logs failed, but you can resume it by '--resume' option")

    else:
        # If all things done successfully, removes the resume file/dir
        shutil.rmtree(wrun_resume_path, ignore_errors=True)
        os.remove(resume_meta_fpath)


def _traverse_github_logs(argv: Any) -> None:
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--max-num-pullreqs', type=int, default=100000)
    parser.add_argument('--until', type=str)
    parser.add_argument('--since', type=str)
    parser.add_argument('--github-token', type=str, required=True)
    parser.add_argument('--github-owner', type=str, default='')
    parser.add_argument('--github-repo', type=str, default='')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--sleep-if-limit-exceeded', action='store_true')
    args = parser.parse_args(argv)

    if not args.resume and len(args.github_owner) == 0:
        raise ValueError("GitHub owner must be specified in '--github-owner'")
    if not args.resume and len(args.github_repo) == 0:
        raise ValueError("GitHub repository must be specified in '--github-repo'")

    if args.resume and not os.path.exists(args.output):
        raise RuntimeError(f'Output path not found in {os.path.abspath(args.output)}')
    elif not args.resume:
        if args.overwrite:
            shutil.rmtree(args.output, ignore_errors=True)

        # Make an output dir in advance
        os.mkdir(args.output)

    # For logger setup
    logger = _setup_logger(f'{args.output}/debug-info.log')

    # logger rate limit
    logger.info(f"rate_limit: {_rate_limit_msg(args.github_token)}")

    # Parses a specified datetime string if necessary
    until = dateutil.parser.parse(args.until) if args.until else None
    since = dateutil.parser.parse(args.since) if args.since else None

    _traverse_pull_requests(args.output, args.github_owner, args.github_repo, args.github_token,
                            until, since, args.max_num_pullreqs, args.resume,
                            args.sleep_if_limit_exceeded,
                            logger)


def _show_rate_limit(argv: Any) -> None:
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--github-token', type=str, required=True)
    args, _ = parser.parse_known_args(argv)

    print('======== GitHub Rate Limit ========')
    print(_rate_limit_msg(args.github_token))


def _list_contributor_stats(argv: Any) -> None:
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--github-token', type=str, required=True)
    parser.add_argument('--github-owner', type=str, required=True)
    parser.add_argument('--github-repo', type=str, required=True)
    args = parser.parse_args(argv)

    if args.overwrite:
        shutil.rmtree(args.output, ignore_errors=True)

    # Make an output dir in advance
    os.mkdir(args.output)

    contributor_stats = github_apis.list_contributor_stats(args.github_owner, args.github_repo, args.github_token)
    with open(f"{args.output}/contributor-stats.json", mode='w') as f:
        f.write(json.dumps(contributor_stats, indent=2))


def _list_updated_file_stats(argv: Any) -> None:
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--github-token', type=str, required=True)
    parser.add_argument('--github-owner', type=str, required=True)
    parser.add_argument('--github-repo', type=str, required=True)
    parser.add_argument('--since', type=str, required=True)
    args = parser.parse_args(argv)

    if args.overwrite:
        shutil.rmtree(args.output, ignore_errors=True)

    # Make an output dir in advance
    os.mkdir(args.output)

    # Parses a specified datetime string
    since_date = dateutil.parser.parse(args.since)

    updated_files: List[Tuple[str, str, str, str, str]] = []
    repo_commits = github_apis.list_repo_commits(args.github_owner, args.github_repo, args.github_token,
                                                 since=since_date)
    for sha, _, date, _ in tqdm.tqdm(repo_commits, desc=f"Commits ({args.github_owner}/{args.github_repo})"):
        _, _, files = github_apis.list_change_files_from(sha, args.github_owner, args.github_repo, args.github_token)
        for filename, adds, dels, chgs in files:
            updated_files.append((filename, date, adds, dels, chgs))

    updated_file_stats: Dict[str, List[Tuple[str, str, str, str]]] = {}
    for filename, date, adds, dels, chgs in updated_files:
        if filename not in updated_file_stats:
            updated_file_stats[filename] = []

        updated_file_stats[filename].append((date, adds, dels, chgs))

    for key in updated_file_stats.keys():
        f = lambda v: github_utils.from_github_datetime(v[0])
        updated_file_stats[key] = sorted(updated_file_stats[key], key=f, reverse=True)

    with open(f"{args.output}/updated-file-stats.json", mode='w') as f:  # type: ignore
        f.write(json.dumps(updated_file_stats, indent=2))  # type: ignore


def main() -> None:
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--list-updated-file-stats', action='store_true')
    parser.add_argument('--list-contributor-stats', action='store_true')
    parser.add_argument('--show-rate-limit', action='store_true')
    args, rest_argv = parser.parse_known_args()

    if args.list_updated_file_stats:
        _list_updated_file_stats(rest_argv)
    elif args.list_contributor_stats:
        _list_contributor_stats(rest_argv)
    elif args.show_rate_limit:
        _show_rate_limit(rest_argv)
    else:
        _traverse_github_logs(rest_argv)


if __name__ == "__main__":
    main()
