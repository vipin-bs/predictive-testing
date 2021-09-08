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
import unittest
import warnings
from datetime import datetime, timezone

from ptesting import github_apis
from ptesting import github_utils


# Suppress warinig messages in REST APIs
warnings.simplefilter('ignore')

# Checks if a github access token is provided
env = dict(os.environ)
github_access_disabled = False
if 'GITHUB_TOKEN' not in env:
    github_access_disabled = True

@unittest.skipIf(
    github_access_disabled,
    "envs 'GITHUB_TOKEN' must be defined to test 'github_apis'")
class GitHubApiTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._github_token = env['GITHUB_TOKEN']
        # Test repository for GitHub APIs
        # TODO: Might it be better to replace it with a more suitable test repository?
        cls._github_owner = 'maropu'
        cls._github_repo = 'spark'

    def _request_github_api(self, api, pass_thru=False):
        return github_apis._request_github_api(api, token=self._github_token, pass_thru=pass_thru)

    def test_request_github_api(self):
        result = self._request_github_api('')
        self.assertEqual(result['repository_url'], 'https://api.github.com/repos/{owner}/{repo}')
        result = self._request_github_api('', pass_thru=True)
        self.assertTrue(type(result) is str)

    def test_to_debug_info(self):
        self.assertEqual(github_apis._to_debug_msg([1, 3, 5]), 'list length:3')
        self.assertEqual(github_apis._to_debug_msg({ 'k1': 'v1', 'k2': 'v2' }), 'top-level keys:k1,k2')

    def _assertIsDateTime(self, t: str) -> None:
        self.assertTrue(type(github_utils.from_github_datetime(t)) is datetime)

    def test_list_pullreqs(self):
        pullreqs = github_apis.list_pullreqs(self._github_owner, self._github_repo, self._github_token, nmax=1)
        self.assertEqual(len(pullreqs), 1)
        pr_number, pr_created_at, pr_updated_at, pr_title, pr_body, pr_user, pr_repo, pr_branch = pullreqs[0]
        self.assertTrue(pr_number.isdigit())
        self._assertIsDateTime(pr_created_at)
        self._assertIsDateTime(pr_updated_at)
        self.assertTrue(len(pr_title) > 0)
        self.assertTrue(len(pr_body) > 0)
        self.assertTrue(len(pr_user) > 0)
        self.assertTrue(len(pr_repo) > 0)
        self.assertTrue(len(pr_branch) > 0)

    def test_list_commits_for(self):
        pullreqs = github_apis.list_pullreqs(self._github_owner, self._github_repo, self._github_token, nmax=1)
        pr_number = pullreqs[0][0]
        commits = github_apis.list_commits_for(pr_number, self._github_owner, self._github_repo, self._github_token, nmax=1)
        self.assertEqual(len(commits), 1)
        sha, commit_date, commit_message = commits[0]
        self.assertTrue(len(sha) > 0)
        self._assertIsDateTime(commit_date)
        self.assertTrue(len(commit_message) > 0)

    def test_list_repo_commits(self):
        commits = github_apis.list_repo_commits(self._github_owner, self._github_repo, self._github_token, path='README.md', nmax=1)
        self.assertEqual(len(commits), 1)
        sha, commit_user, commit_date, commit_message = commits[0]
        self.assertTrue(len(sha) > 0)
        self._assertIsDateTime(commit_date)
        self.assertTrue(len(commit_user) > 0)
        self.assertTrue(len(commit_message) > 0)

    def test_list_change_files_from(self):
        ref = '5a510cf578c84e3edb7fb58d16c332ca141be913'
        commit_date, commit_message, files = github_apis.list_change_files_from(ref, self._github_owner, self._github_repo, self._github_token)
        self.assertEqual(commit_date, '2021-06-22T08:15:35Z')
        self.assertTrue(commit_message.startswith('[SPARK-35726][SPARK-35769][SQL][FOLLOWUP]'))
        self.assertEqual(sorted(files), [
            ('sql/core/src/main/scala/org/apache/spark/sql/execution/HiveResult.scala', '10', '2', '12')
        ])

    def test_list_change_files_between(self):
        base = '5a510cf578c84e3edb7fb58d16c332ca141be913'
        head = 'bc61b62a55c5c3ace181aef53e26a5ddcd6b85bf'
        commit_date, commit_message, files = github_apis.list_change_files_between(base, head, self._github_owner, self._github_repo, self._github_token)
        self.assertEqual(commit_date, '2021-06-22T13:31:24Z')
        self.assertTrue(commit_message.startswith('[SPARK-35838][BUILD][TESTS]'))
        self.assertEqual(sorted(files), [
            ('external/avro/pom.xml', '11', '0', '11'),
            ('external/kafka-0-10-sql/pom.xml', '11', '1', '12'),
            ('sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/datetimeExpressions.scala', '2', '2', '4'),
            ('sql/core/src/test/resources/sql-tests/results/ansi/datetime.sql.out', '2', '2', '4'),
            ('sql/core/src/test/resources/sql-tests/results/ansi/interval.sql.out', '2', '0', '2'),
            ('sql/core/src/test/resources/sql-tests/results/datetime-legacy.sql.out', '2', '2', '4'),
            ('sql/core/src/test/resources/sql-tests/results/datetime.sql.out', '2', '2', '4'),
            ('sql/core/src/test/resources/sql-tests/results/interval.sql.out', '1', '0', '1'),
            ('sql/core/src/test/resources/sql-tests/results/typeCoercion/native/promoteStrings.sql.out', '1', '1', '2'),
            ('sql/core/src/test/scala/org/apache/spark/sql/ColumnExpressionSuite.scala', '4', '1', '5'),
            ('sql/hive-thriftserver/pom.xml', '11', '0', '11')])

    def test_list_file_commits_with_period_specified(self):
        commits = github_apis.list_repo_commits(self._github_owner, self._github_repo, self._github_token,
                                                path='README.md', since='2011-04-1T05:31:29Z', until='2012-04-01T05:31:29Z')
        self.assertEqual(len(commits), 7)
        sha, commit_user, commit_date, commit_message = commits[0]
        self.assertEqual(sha, 'ca64a7ae03f2ba4a965b6f2b55afbd6d9f2a397a')
        self.assertEqual(commit_user, 'mateiz')
        self.assertEqual(commit_message, 'Documentation')
        self.assertEqual(commit_date, '2012-03-17T20:51:29Z')

    def test_list_workflow_runs(self):
        runs = github_apis.list_workflow_runs(self._github_owner, self._github_repo, self._github_token, nmax=1)
        self.assertEqual(len(runs), 1)
        run_id, name, head_sha, event, conclusion, pr_number, pr_head, pr_base = runs[0]
        self.assertTrue(run_id.isdigit())
        self.assertTrue(len(name) > 0)
        self.assertTrue(event in ['workflow_run', 'schedule', 'push'])
        self.assertTrue(len(head_sha) > 0)
        self.assertTrue(conclusion in ['success', 'failure', 'skipped'])
        self.assertTrue(pr_number == '' or pr_number.isdigit())
        self.assertTrue(len(pr_head) >= 0)
        self.assertTrue(len(pr_base) >= 0)

    def test_list_workflow_jobs(self):
        runs = github_apis.list_workflow_runs(self._github_owner, self._github_repo, self._github_token, nmax=1)
        self.assertEqual(len(runs), 1)
        run_id = runs[0][0]
        jobs = github_apis.list_workflow_jobs(run_id, self._github_owner, self._github_repo, self._github_token, nmax=1)
        self.assertEqual(len(jobs), 1)
        job_id, name, conclusion = jobs[0]
        self.assertTrue(job_id.isdigit())
        self.assertTrue(len(name) > 0)
        self.assertTrue(conclusion in ['success', 'failure', 'skipped'])

    def test_list_workflow_job_logs(self):
        runs = github_apis.list_workflow_runs(self._github_owner, self._github_repo, self._github_token, nmax=1)
        self.assertEqual(len(runs), 1)
        run_id = runs[0][0]
        jobs = github_apis.list_workflow_jobs(run_id, self._github_owner, self._github_repo, self._github_token, nmax=1)
        self.assertEqual(len(jobs), 1)
        job_id, _, _ = jobs[0]
        logs = github_apis.get_workflow_job_logs(job_id, self._github_owner, self._github_repo, self._github_token)
        self.assertIsNotNone(logs)

    def test_list_contributors_stats(self):
        contributors = github_apis.list_contributors_stats(self._github_owner, self._github_repo, self._github_token)
        self.assertEqual(len(contributors), 100)
        self.assertEqual(contributors[0:3], [('rxin', 1179), ('cloud-fan', 923), ('HyukjinKwon', 839)])

    def test_get_rate_limit(self):
        rate_limit = github_apis.get_rate_limit(self._github_token)
        self.assertTrue(set(rate_limit.keys()), set(['resources', 'rate']))
        self.assertTrue(set(rate_limit['resources'].keys()), set(['core', 'search', 'graphql']))


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
