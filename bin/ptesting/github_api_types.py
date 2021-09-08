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


class RateLimit(BaseModel):
    limit: int = Field(ge=0)
    remaining: int = Field(ge=0)
    reset: int = Field(ge=0)
    used: int = Field(ge=0)


class ResourceLimit(BaseModel):
    core: RateLimit
    search: RateLimit
    graphql: RateLimit
    graphql: RateLimit


class RateLimits(BaseModel):
    resources: ResourceLimit
    rate: RateLimit


class User(BaseModel):
    login: str = Field(min_length=1, max_length=39)


class Repository(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class Head(BaseModel):
    repo: Optional[Repository] = None
    ref: str = Field(min_length=1)
    sha: str = Field(min_length=40, max_length=40)


def _validate_datetime(v):
    try:
        github_utils.from_github_datetime(v)
    except:
        raise ValueError(f"Failed to parse input datetime string: {v}")
    return v


class PullRequest(BaseModel):
    number: int = Field(ge=0)
    created_at: str
    updated_at: str
    title: str = Field(min_length=1, max_length=256)
    body: str = Field(min_length=1, max_length=65536)
    user: User
    head: Head

    @validator("created_at")
    def validate_created_at(cls, v):
        return _validate_datetime(v)

    @validator("updated_at")
    def validate_updated_at(cls, v):
        return _validate_datetime(v)


class Author(BaseModel):
    name: str = Field(min_length=1, max_length=39)
    date: str

    @validator("date")
    def validate_date(cls, v):
        return _validate_datetime(v)


class Commit(BaseModel):
    author: Author
    message: str = Field(min_length=1)


class PullRequestCommit(BaseModel):
    sha: str = Field(min_length=40, max_length=40)
    commit: Commit


class RepoCommit(BaseModel):
    sha: str = Field(min_length=40, max_length=40)
    commit: Commit
    author: Optional[User] = None
    committer: Optional[User] = None


class ContributorStat(BaseModel):
    author: User
    total: int = Field(ge=1)
