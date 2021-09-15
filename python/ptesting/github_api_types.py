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

from pydantic import BaseModel, Field, validator
from typing import List, Optional

from ptesting import github_utils

"""
Type validation classes for Github REST APIs
"""


def _validate_datetime(v: str) -> str:
    try:
        github_utils.from_github_datetime(v)
    except:
        raise ValueError(f"Failed to parse input datetime string: {v}")
    return v


class RateLimit(BaseModel):
    limit: int = Field(ge=0)
    remaining: int = Field(ge=0)
    reset: int = Field(ge=0)
    used: int = Field(ge=0)


class ResourceLimit(BaseModel):
    core: RateLimit
    search: RateLimit
    graphql: RateLimit


class RateLimits(BaseModel):
    resources: ResourceLimit
    rate: RateLimit


class User(BaseModel):
    login: str = Field(min_length=1, max_length=39)


class Author(BaseModel):
    name: str = Field(min_length=1, max_length=39)
    date: str

    @validator("date")
    def validate_date(cls, v: str) -> str:
        return _validate_datetime(v)


class Repository(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class Reference(BaseModel):
    repo: Optional[Repository] = None
    ref: str = Field(min_length=1)
    sha: str = Field(min_length=40, max_length=40)


class PullRequest(BaseModel):
    number: int = Field(ge=0)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    title: Optional[str] = Field(min_length=0)
    body: Optional[str] = Field(min_length=0, max_length=65536)
    user: Optional[User] = None
    head: Reference
    base: Reference

    @validator("created_at")
    def validate_created_at(cls, v: str) -> str:
        return _validate_datetime(v)

    @validator("updated_at")
    def validate_updated_at(cls, v: str) -> str:
        return _validate_datetime(v)


class Commit(BaseModel):
    author: Author
    message: str = Field(min_length=1)


class RepoCommit(BaseModel):
    sha: str = Field(min_length=40, max_length=40)
    commit: Commit
    author: Optional[User] = None
    committer: Optional[User] = None


class ChangedFile(BaseModel):
    filename: str = Field(min_length=1)
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    changes: int = Field(ge=0)


class ChangedFiles(BaseModel):
    files: List[ChangedFile] = []


class FileCommits(BaseModel):
    commit: Optional[Commit] = None
    commits: Optional[List[RepoCommit]] = None


class WorkflowRun(BaseModel):
    id: int = Field(ge=1)
    name: str = Field(min_length=1)
    head_sha: str = Field(min_length=40, max_length=40)
    event: str = Field(min_length=1)
    status: str
    conclusion: Optional[str] = None
    updated_at: str
    pull_requests: List[PullRequest] = []

    @validator("status")
    def validate_status(cls, v: str) -> str:
        expected = ['waiting', 'requested', 'completed', 'in_progress', 'queued']
        if v not in expected:
            raise ValueError(f"'status' must be in [{','.join(expected)}], "
                             f"but '{v}' found")
        return v

    @validator("conclusion")
    def validate_conclusion(cls, v: str) -> str:
        expected = ['success', 'failure', 'skipped', 'cancelled', 'startup_failure']
        if not (v is None or v in expected):
            raise ValueError(f"'conclusion' must be in [{','.join(expected)}], "
                             f"but '{v}' found")
        return v

    @validator("updated_at")
    def validate_udpated_at(cls, v: str) -> str:
        return _validate_datetime(v)


class WorkflowRuns(BaseModel):
    total_count: int = Field(ge=0)
    workflow_runs: List[WorkflowRun]


class WorkflowJob(BaseModel):
    id: int = Field(ge=1)
    name: str = Field(min_length=1)
    conclusion: Optional[str] = None

    @validator("conclusion")
    def validate_conclusion(cls, v: str) -> str:
        expected = ['success', 'failure', 'skipped', 'cancelled', 'startup_failure']
        if not (v is None or v in expected):
            raise ValueError(f"'conclusion' must be in [{','.join(expected)}], "
                             f"but '{v}' found")
        return v


class WorkflowJobs(BaseModel):
    total_count: int = Field(ge=0)
    workflow_runs: List[WorkflowJob] = []


class ContributorStat(BaseModel):
    author: User
    total: int = Field(ge=1)
