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
import requests
import retrying
import timeout_decorator
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union


def _setup_default_logger():
    from logging import getLogger, NullHandler, DEBUG
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    logger.addHandler(NullHandler())
    logger.propagate = False
    return logger


_default_logger = _setup_default_logger()


# The GitHub time format (UTC)
GITHUB_DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%SZ'


def from_github_datetime(d: str) -> datetime:
    # Parses GitHub timestamp string into timezone-aware datetime
    return datetime.strptime(d, GITHUB_DATETIME_FORMAT).replace(tzinfo=timezone.utc)


def format_github_datetime(d: str, format: str) -> str:
    return from_github_datetime(d).strftime(format)


def to_github_datetime(d: datetime) -> str:
    return d.strftime(GITHUB_DATETIME_FORMAT)


def _to_debug_msg(ret: Any) -> str:
    if type(ret) == dict:
        return f"top-level keys:{','.join(sorted(ret.keys()))}"
    elif type(ret) == list:
        return f"list length:{len(ret)}"
    else:
        return "ret:<unknown>"


def _to_error_msg(text: str) -> str:
    try:
        return (json.loads(text))['message']
    except:
        return text


# For a list of requests's exceptions, see:
# https://docs.python-requests.org/en/latest/user/quickstart/#errors-and-exceptions
def _retry_if_timeout(caught: Any) -> Any:
    return isinstance(caught, requests.exceptions.Timeout)


@retrying.retry(stop_max_attempt_number=4, wait_exponential_multiplier=1000, wait_exponential_max=4000,
                retry_on_exception=_retry_if_timeout,
                wrap_exception=False)
def _request_github_api(api: str, token: str, params: Dict[str, str] = {}, pass_thru: bool = False,
                        logger: Any = _default_logger) -> Any:
    headers = { 'Accept': 'application/vnd.github.v3+json', 'Authorization': f'Token {token}', 'User-Agent': 'github-apis' }
    ret = requests.get(f'https://api.github.com/{api}', timeout=10, headers=headers, params=params, verify=False)
    if ret.status_code != 200:
        error_msg = "{} request (params={}) failed because: {}"
        if ret.status_code == 403 and ret.text.find('API rate limit exceeded') != -1:
            error_msg = error_msg.format(
                api, str(params), 'the GitHub API rate limit exceeded')
        else:
            error_msg = error_msg.format(
                api, str(params), f"status_code={ret.status_code}, msg='{_to_error_msg(ret.text)}'")

        raise RuntimeError(error_msg)

    if not pass_thru:
        result = json.loads(ret.text)
        logger.info(f"api:/{api}, params:{params}, {_to_debug_msg(result)}")
        logger.debug(f"ret:{json.dumps(result, indent=4)}")
        return result
    else:
        return ret.text


def _assert_github_prams(owner, repo, token):
    def is_valid_str(s):
        return type(s) is str and len(s) > 0

    assert is_valid_str(owner) and is_valid_str(repo) and is_valid_str(token), \
        f"Invalid input found: owner={owner}, repo={repo}, token={token}"


def _always_false(d: str) -> bool:
    return False


def _create_since_validator(since: datetime) -> Any:
    def validate_with_since(d: str) -> bool:
        return not since < from_github_datetime(d)

    return validate_with_since


def _create_date_filter(since: Optional[datetime]) -> Any:
    f = _always_false
    if since is not None:
        f = _create_since_validator(since)

    return f


def _validate_dict_keys(d: Any, expected_keys: List[str], logger: Any = _default_logger) -> bool:
    if type(d) is not dict:
        logger.warning(f"Expected type is 'dict', but '{type(d).__name__}' found")
        return False
    else:
        nonexistent_keys = list(filter(lambda k: k not in d, expected_keys))
        if len(nonexistent_keys) != 0:
            logger.warning(f"Expected keys ({','.join(nonexistent_keys)}) are not found "
                           f"in {','.join(d.keys())} ")
            return False

    return True


def get_rate_limit(token: str, logger: Any = None) -> Dict[str, Any]:
    return _request_github_api(f"rate_limit", token)


# https://docs.github.com/en/rest/reference/pulls#list-pull-requests
@timeout_decorator.timeout(1800, timeout_exception=StopIteration)
def list_pullreqs(owner: str, repo: str, token: str, since: Optional[datetime] = None, nmax: int = 100000,
                  logger: Any = None) -> List[Tuple[str, str, str, str, str, str, str, str]]:
    _assert_github_prams(owner, repo, token)

    logger = logger or _default_logger

    pullreqs: List[Tuple[str, str, str, str, str, str, str, str]] = []
    check_updated = _create_date_filter(since)
    rem_pages = nmax
    npage = 1
    while True:
        per_page = 100 if rem_pages >= 100 else rem_pages
        params = { 'page': str(npage), 'per_page': str(per_page), 'state': 'all', 'sort': 'updated', 'direction': 'desc' }
        prs = _request_github_api(f"repos/{owner}/{repo}/pulls", token, params=params, logger=logger)
        for pullreq in prs:
            expected_keys = ['number', 'created_at', 'updated_at', 'title', 'body', 'user', 'head']
            if not _validate_dict_keys(pullreq, expected_keys, logger=logger):
                return pullreqs

            if check_updated(pullreq['updated_at']):
                return pullreqs

            if pullreq['head']['repo'] is not None:
                pr_number = str(pullreq['number'])
                pr_created_at = pullreq['created_at']
                pr_updated_at = pullreq['updated_at']
                pr_title = pullreq['title']
                pr_body = pullreq['body']
                pr_user = pullreq['user']['login']
                pr_repo = pullreq['head']['repo']['name']
                pr_branch = pullreq['head']['ref']
                pullreqs.append((pr_number, pr_created_at, pr_updated_at, pr_title, pr_body,
                                 pr_user, pr_repo, pr_branch))
            else:
                logger.warning(f"repository not found: pr_number={pullreq['number']}, "
                               f"pr_user={pullreq['user']['login']}")

        rem_pages -= per_page
        npage += 1
        if len(prs) == 0 or rem_pages == 0:
            return pullreqs

    assert False, 'unreachable path'
    return []


# https://docs.github.com/en/rest/reference/pulls#list-commits-on-a-pull-request
def list_commits_for(pr_number: str, owner: str, repo: str, token: str,
                     since: Optional[datetime] = None, nmax: int = 100000,
                     logger: Any = None) -> List[Tuple[str, str, str]]:
    _assert_github_prams(owner, repo, token)

    logger = logger or _default_logger

    commits: List[Tuple[str, str, str]] = []
    check_date = _create_date_filter(since)
    rem_pages = nmax
    npage = 1
    while True:
        per_page = 100 if rem_pages >= 100 else rem_pages
        params = { 'page': str(npage), 'per_page': str(per_page) }
        pr_commits = _request_github_api(f"repos/{owner}/{repo}/pulls/{pr_number}/commits", token,
                                         params=params, logger=logger)
        for commit in pr_commits:
            if not _validate_dict_keys(commit, ['sha', 'commit'], logger=logger):
                return commits

            commit_date = commit['commit']['author']['date']
            if check_date(commit_date):
                return commits

            commits.append((commit['sha'], commit_date, commit['commit']['message']))

        rem_pages -= per_page
        npage += 1
        if len(pr_commits) == 0 or rem_pages == 0:
            return commits

    assert False, 'unreachable path'
    return []


# https://docs.github.com/en/rest/reference/repos#list-commits
def list_repo_commits(owner: str, repo: str, token: str,
                      path: Optional[str], since: Optional[str] = None, until: Optional[str] = None,
                      nmax: int = 100000, logger: Any = None) -> List[Tuple[str, str, str, str]]:
    _assert_github_prams(owner, repo, token)

    logger = logger or _default_logger

    # Adds some optional params if necessary
    extra_params = {}
    if path is not None:
        extra_params['path'] = str(path)
    if since is not None:
        extra_params['since'] = str(since)
    if until is not None:
        extra_params['until'] = str(until)

    commits: List[Tuple[str, str, str, str]] = []
    rem_pages = nmax
    npage = 1
    while True:
        per_page = 100 if rem_pages >= 100 else rem_pages
        params = { 'page': str(npage), 'per_page': str(per_page) }
        params.update(extra_params)
        file_commits = _request_github_api(f"repos/{owner}/{repo}/commits", token,
                                           params=params, logger=logger)
        for commit in file_commits:
            if not _validate_dict_keys(commit, ['sha', 'author', 'commit'], logger=logger):
                return commits

            sha = commit['sha']
            commit_date = commit['commit']['author']['date']
            commit_message = commit['commit']['message']
            commit_user = ''
            if commit['author'] is not None:
                commit_user = commit['author']['login']
            elif commit['committer'] is not None:
                commit_user = commit['committer']['login']

            commits.append((sha, commit_user, commit_date, commit_message))

        rem_pages -= per_page
        npage += 1
        if len(file_commits) == 0 or rem_pages == 0:
            return commits

    assert False, 'unreachable path'
    return []


def _list_change_files(api: str, token: str, logger: Any) -> List[Tuple[str, str, str, str]]:
    logger = logger or _default_logger

    files: List[Tuple[str, str, str, str]] = []
    npage = 1
    while True:
        params = { 'page': str(npage), 'per_page': '100' }
        changed_files= _request_github_api(api, token, params=params, logger=logger)
        if 'files' not in changed_files or len(changed_files['files']) == 0:
             return files

        for file in changed_files['files']:
            files.append((file['filename'], str(file['additions']), str(file['deletions']), str(file['changes'])))

        npage += 1

    assert False, 'unreachable path'
    return []


# https://docs.github.com/en/rest/reference/repos#get-a-commit
def list_change_files_from(ref: str, owner: str, repo: str, token: str,
                           logger: Any = None) -> List[Tuple[str, str, str, str]]:
    _assert_github_prams(owner, repo, token)
    return _list_change_files(f"repos/{owner}/{repo}/commits/{ref}", token, logger)


# https://docs.github.com/en/rest/reference/repos#compare-two-commits
def list_change_files_between(base: str, head: str, owner: str, repo: str, token: str,
                              logger: Any = None) -> List[Tuple[str, str, str, str]]:
    _assert_github_prams(owner, repo, token)
    return _list_change_files(f"repos/{owner}/{repo}/compare/{base}...{head}", token, logger)


# https://docs.github.com/en/rest/reference/actions#list-workflow-runs-for-a-repository
def list_workflow_runs(owner: str, repo: str, token: str, since: Optional[datetime] = None,
                       nmax: int = 100000, logger: Any = None) -> List[Tuple[str, str, str, str, str, str, str, str]]:
    _assert_github_prams(owner, repo, token)

    logger = logger or _default_logger

    api = f'repos/{owner}/{repo}/actions/runs'
    latest_run = _request_github_api(api, token, params={ 'per_page': '1' }, logger=logger)
    if not _validate_dict_keys(latest_run, ['total_count', 'workflow_runs'], logger=logger):
        return []

    runs: List[Tuple[str, str, str, str, str, str, str, str]] = []
    check_updated = _create_date_filter(since)
    total_runs_count = int(latest_run['total_count'])
    num_pages = int(total_runs_count / 100) + 1
    rem_pages = nmax
    for page in range(0, num_pages):
        per_page = 100 if rem_pages >= 100 else rem_pages
        params = { 'page': str(page), 'per_page': str(per_page) }
        wruns = _request_github_api(api, token=token, params=params, logger=logger)
        for run in wruns['workflow_runs']:
            expected_keys = ['id', 'name', 'event', 'status', 'conclusion', 'updated_at', 'pull_requests']
            if not _validate_dict_keys(run, expected_keys, logger=logger):
                return runs

            if check_updated(run['updated_at']):
                return runs

            if run['status'] == 'completed':
                if len(run['pull_requests']) == 0:
                    pr_number, pr_head, pr_base = '', '', ''
                else:
                    pr = run['pull_requests'][0]
                    pr_number = str(pr['number'])
                    pr_head = pr['head']['sha']
                    pr_base = pr['base']['sha']

                runs.append((str(run['id']), run['name'], run['head_sha'], run['event'], run['conclusion'],
                             pr_number, pr_head, pr_base))

        rem_pages -= per_page
        if len(wruns) == 0 or rem_pages == 0:
            return runs

    return runs


# https://docs.github.com/en/rest/reference/actions#list-jobs-for-a-workflow-run
def list_workflow_jobs(run_id: str, owner: str, repo: str, token: str, nmax: int = 100000,
                       logger: Any = None) -> List[Tuple[str, str, str]]:
    _assert_github_prams(owner, repo, token)

    logger = logger or _default_logger

    api = f'repos/{owner}/{repo}/actions/runs/{run_id}/jobs'
    latest_job = _request_github_api(api, token, params={ 'per_page': '1' }, logger=logger)
    if not _validate_dict_keys(latest_job, ['total_count', 'jobs'], logger=logger):
        return []

    jobs: List[Tuple[str, str, str]] = []
    total_jobs_count = int(latest_job['total_count'])
    num_pages = int(total_jobs_count / 100) + 1
    rem_pages = nmax
    for page in range(0, num_pages):
        per_page = 100 if rem_pages >= 100 else rem_pages
        params = { 'page': str(page), 'per_page': str(per_page) }
        wjobs = _request_github_api(api, token=token, params=params, logger=logger)
        for job in wjobs['jobs']:
            if not _validate_dict_keys(job, ['id', 'name', 'conclusion'], logger=logger):
                return jobs

            jobs.append((str(job['id']), job['name'], job['conclusion']))

        rem_pages -= per_page
        if len(wjobs) == 0 or rem_pages == 0:
            return jobs

    return jobs


# https://docs.github.com/en/rest/reference/actions#download-job-logs-for-a-workflow-run
def get_workflow_job_logs(job_id: str, owner: str, repo: str, token: str, logger: Any = None) -> str:
    _assert_github_prams(owner, repo, token)
    try:
        api = f'repos/{owner}/{repo}/actions/jobs/{job_id}/logs'
        return _request_github_api(api, token, pass_thru=True)
    except:
        logger.warning(f"Job logs (job_id={job_id}) not found in {owner}/{repo}")
        return ''


# https://docs.github.com/en/rest/reference/repos#get-all-contributor-commit-activity
def list_contributors_stats(owner: str, repo: str, token: str, logger: Any = None) -> List[Tuple[str, str]]:
    _assert_github_prams(owner, repo, token)

    logger = logger or _default_logger

    contributors: List[Tuple[str, str]] = []
    stats = _request_github_api(f"repos/{owner}/{repo}/stats/contributors", token, logger=logger)
    for stat in stats:
        if not _validate_dict_keys(stat, ['author', 'total'], logger=logger):
            return []

        contributors.append((stat['author']['login'], str(stat['total'])))

    res = sorted(contributors, key=lambda c: int(c[1]), reverse=True)  # Sorted by 'total'
    return res
