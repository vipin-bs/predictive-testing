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
import timeout_decorator
import json
import requests
import retrying
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union


# The GitHub time format (UTC)
GITHUB_DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%SZ'


def from_github_datetime(d: str) -> datetime:
    # Parses GitHub timestamp string into timezone-aware datetime
    return datetime.strptime(d, GITHUB_DATETIME_FORMAT).replace(tzinfo=timezone.utc)


def format_github_datetime(d: str, format: str) -> str:
    return from_github_datetime(d).strftime(format)


def to_github_datetime(d: datetime) -> str:
    return d.strftime(GITHUB_DATETIME_FORMAT)


@retrying.retry(stop_max_attempt_number=4, wait_exponential_multiplier=1000, wait_exponential_max=4000)
def _request_github_api(api: str, token: str, params: Dict[str, str] = {}, pass_thru: bool = False) -> Union[str, Dict[str, Any]]:
    headers = { 'Accept': 'application/vnd.github.v3+json', 'Authorization': f'Token {token}' }
    ret = requests.get(f'https://api.github.com/{api}', timeout=10, headers=headers, params=params, verify=False)
    if not pass_thru:
        ret_as_dict = json.loads(ret.text)
        if 'message' in ret_as_dict and ret_as_dict['message'] == 'Not Found':
            logging.warning(f"{api} request (params={params}) not found")
            return {}
        else:
            logging.info(f"api:/{api}, params:{params}, keys:{','.join(ret_as_dict.keys())}")
            logging.debug(f"ret:{json.dumps(ret_as_dict, indent=4)}")
            return ret_as_dict
    else:
        return ret.text


def request_github_api(api: str, token: str, params: Dict[str, str] = {}, pass_thru: bool = False) -> Union[str, Dict[str, Any]]:
    try:
        return _request_github_api(api, token, params, pass_thru)
    except:
        logging.warning(f"{api} request (params={params}) failed")
        return {} if not pass_thru else []


# https://docs.github.com/en/rest/reference/pulls#list-pull-requests
@timeout_decorator.timeout(1800, timeout_exception=StopIteration)
def list_pullreqs(owner: str, repo: str, token, since: Optional[datetime] = None, nmax: int = 100000) -> List[Tuple[str, str, str, str, str, str, str, str]]:
    pullreqs: List[Tuple[str, str, str, str, str, str, str, str]] = []

    def always_false(d: str) -> bool:
        return False

    def validate_with_since(d: str) -> bool:
        return not since < from_github_datetime(d)

    check_updated = always_false
    if since is not None:
        check_updated = validate_with_since

    rem_pages = nmax
    npage = 1
    while True:
        per_page = 100 if rem_pages >= 100 else rem_pages
        params = { 'page': str(npage), 'per_page': str(per_page), 'state': 'all', 'sort': 'updated', 'direction': 'desc' }
        prs = request_github_api(f"repos/{owner}/{repo}/pulls", token, params=params)
        for pullreq in prs:
            if check_updated(pullreq['updated_at']):
                return pullreqs

            pr_number = str(pullreq['number'])
            pr_created_at = pullreq['created_at']
            pr_updated_at = pullreq['updated_at']
            pr_title = pullreq['title']
            pr_body = pullreq['body']
            pr_user = pullreq['user']['login']
            pr_repo = pullreq['head']['repo']['name'] if pullreq['head']['repo'] is not None else ''
            pr_branch = pullreq['head']['ref']
            pullreqs.append((pr_number, pr_created_at, pr_updated_at, pr_title, pr_body,
                             pr_user, pr_repo, pr_branch))

        rem_pages -= per_page
        npage += 1
        if len(prs) == 0 or rem_pages == 0:
            return pullreqs

    assert False, 'unreachable path'
    return []


# https://docs.github.com/en/rest/reference/pulls#get-a-pull-request
def list_commits_for(pr_number: str, owner: str, repo: str, token, nmax: int = 100000) -> List[Tuple[str, str]]:
    commits: List[Tuple[str, str]] = []

    rem_pages = nmax
    npage = 1
    while True:
        per_page = 100 if rem_pages >= 100 else rem_pages
        params = { 'page': str(npage), 'per_page': str(per_page) }
        pr_commits = request_github_api(f"repos/{owner}/{repo}/pulls/{pr_number}/commits", token, params=params)
        for commit in pr_commits:
            commits.append((commit['sha'], commit['commit']['author']['date']))

        rem_pages -= per_page
        npage += 1
        if len(pr_commits) == 0 or rem_pages == 0:
            return commits

    assert False, 'unreachable path'
    return []


# https://docs.github.com/en/rest/reference/repos#list-commits
def list_file_commits_for(path: str, owner: str, repo: str, token,
                          since: Optional[str] = None, until: Optional[str] = None,
                          nmax: int = 100000) -> List[Tuple[str, str]]:
    commits: List[Tuple[str, str]] = []

    # Limits a read scope if 'since' or 'until' specified
    extra_params = {}
    if since is not None:
        extra_params['since'] = str(since)
    if until is not None:
        extra_params['until'] = str(until)

    rem_pages = nmax
    npage = 1
    while True:
        per_page = 100 if rem_pages >= 100 else rem_pages
        params = { 'page': str(npage), 'per_page': str(per_page), 'path': path }
        params.update(extra_params)
        file_commits = request_github_api(f"repos/{owner}/{repo}/commits", token, params=params)
        for commit in file_commits:
            commits.append((commit['sha'], commit['commit']['author']['date']))

        rem_pages -= per_page
        npage += 1
        if len(file_commits) == 0 or rem_pages == 0:
            return commits

    assert False, 'unreachable path'
    return []


# https://docs.github.com/en/rest/reference/repos#compare-two-commits
def list_change_files(base: str, head: str, owner: str, repo: str, token, nmax: int = 100000) -> List[Tuple[str, str, str, str]]:
    files: List[Tuple[str, str, str, str]] = []

    rem_pages = nmax
    npage = 1
    while True:
        per_page = 100 if rem_pages >= 100 else rem_pages
        params = { 'page': str(npage), 'per_page': str(per_page) }
        compare = request_github_api(f"repos/{owner}/{repo}/compare/{base}...{head}", token, params=params)
        if 'files' in compare:
            for file in compare['files']:
                files.append((file['filename'], file['additions'], file['deletions'], file['changes']))

        rem_pages -= per_page
        npage += 1
        if len(compare['commits']) == 0 or rem_pages == 0:
            return files

    assert False, 'unreachable path'
    return []


# https://docs.github.com/en/rest/reference/actions#list-workflow-runs-for-a-repository
def list_workflow_runs(owner: str, repo: str, token: str, since: Optional[datetime] = None, nmax: int = 100000, testing: bool = False) -> List[Tuple[str, str, str, str, str, str]]:
    runs: List[Tuple[str, str, str, str, str, str]] = []

    def always_false(d: str) -> bool:
        return False

    def validate_with_since(d: str) -> bool:
        return not since < from_github_datetime(d)

    check_updated = always_false
    if since is not None:
        check_updated = validate_with_since

    api = f'repos/{owner}/{repo}/actions/runs'
    latest_run = request_github_api(api, token, params={ 'per_page': '1' })
    if len(latest_run) == 0:
        return []

    total_runs_count = int(latest_run['total_count'])
    num_pages = int(total_runs_count / 100) + 1
    rem_pages = nmax
    for page in range(0, num_pages):
        per_page = 100 if rem_pages >= 100 else rem_pages
        params = { 'page': str(page), 'per_page': str(per_page) }
        wruns = request_github_api(api, token=token, params=params)
        for run in wruns['workflow_runs']:
            if check_updated(run['updated_at']):
                return runs

            if run['event'] == 'push' and len(run['pull_requests']) > 0:
                pr = run['pull_requests'][0]
                runs.append((str(run['id']), run['name'], run['event'], pr['number'], pr['head']['sha'], pr['base']['sha']))
            # TODO: Removes 'testing' in future
            elif testing:
                runs.append((str(run['id']), run['name'], run['event'], '', '', ''))

        rem_pages -= per_page
        if len(wruns) == 0 or rem_pages == 0:
            return runs

    return runs


# https://docs.github.com/en/rest/reference/actions#list-jobs-for-a-workflow-run
def list_workflow_jobs(run_id: str, owner: str, repo: str, token, nmax: int = 100000) -> List[Tuple[str, str, str]]:
    jobs: List[Tuple[str, str, str]] = []

    api = f'repos/{owner}/{repo}/actions/runs/{run_id}/jobs'
    latest_job = request_github_api(api, token, params={ 'per_page': '1' })
    if len(latest_job) == 0:
        return []

    total_jobs_count = int(latest_job['total_count'])
    num_pages = int(total_jobs_count / 100) + 1
    rem_pages = nmax

    for page in range(0, num_pages):
        per_page = 100 if rem_pages >= 100 else rem_pages
        params = { 'page': str(page), 'per_page': str(per_page) }
        wjobs = request_github_api(api, token=token, params=params)
        for job in wjobs['jobs']:
            jobs.append((str(job['id']), job['name'], job['conclusion']))

        rem_pages -= per_page
        if len(wjobs) == 0 or rem_pages == 0:
            return jobs

    return jobs


# https://docs.github.com/en/rest/reference/actions#download-job-logs-for-a-workflow-run
def get_workflow_job_logs(job_id: str, owner: str, repo: str, token) -> str:
    api = f'repos/{owner}/{repo}/actions/jobs/{job_id}/logs'
    return request_github_api(api, token, pass_thru=True)
