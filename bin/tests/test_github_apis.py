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

import github_apis


# Suppress warinig messages in REST APIs
warnings.simplefilter('ignore')

# Checks if a github access token is provided
env = dict(os.environ)
github_access_disabled = False
if 'GITHUB_TOKEN' not in env:
    github_token_not_specified = True

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
        return github_apis.request_github_api(api, token=self._github_token, pass_thru=pass_thru)

    def test_request_github_api(self):
        result = self._request_github_api('')
        self.assertEqual(result['repository_url'], 'https://api.github.com/repos/{owner}/{repo}')
        result = self._request_github_api('', pass_thru=True)
        self.assertTrue(type(result) is str)

    def test_list_pullreqs(self):
        pullreqs = github_apis.list_pullreqs(self._github_owner, self._github_repo, self._github_token, nmax=1)
        self.assertEqual(len(pullreqs), 1)
        pr_number, pr_created_at, pr_updated_at, pr_title, pr_body, pr_user, pr_repo, pr_branch = pullreqs[0]
        self.assertIsNotNone(pr_number)
        self.assertIsNotNone(pr_created_at)
        self.assertIsNotNone(pr_updated_at)
        self.assertIsNotNone(pr_title)
        self.assertIsNotNone(pr_body)
        self.assertIsNotNone(pr_user)
        self.assertIsNotNone(pr_repo)
        self.assertIsNotNone(pr_branch)

    def test_list_commits_for(self):
        pullreqs = github_apis.list_pullreqs(self._github_owner, self._github_repo, self._github_token, nmax=1)
        pr_number = pullreqs[0][0]
        commits = github_apis.list_commits_for(pr_number, self._github_owner, self._github_repo, self._github_token, nmax=1)
        self.assertEqual(len(commits), 1)
        sha, commit_date = commits[0]
        self.assertIsNotNone(sha)
        self.assertIsNotNone(commit_date)

    def test_list_file_commits_for(self):
        commits = github_apis.list_file_commits_for('README.md', self._github_owner, self._github_repo, self._github_token, nmax=1)
        self.assertEqual(len(commits), 1)
        sha, commit_date = commits[0]
        self.assertIsNotNone(sha)
        self.assertIsNotNone(commit_date)

    def test_list_change_files(self):
        base = '5a510cf578c84e3edb7fb58d16c332ca141be913'
        head = 'bc61b62a55c5c3ace181aef53e26a5ddcd6b85bf'
        files = github_apis.list_change_files(base, head, self._github_owner, self._github_repo, self._github_token)
        self.assertEqual(len(files), 11)
        filenames = set([ f[0] for f in files ])
        self.assertEqual(filenames, set([
            'external/avro/pom.xml',
            'external/kafka-0-10-sql/pom.xml',
            'sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/datetimeExpressions.scala',
            'sql/core/src/test/resources/sql-tests/results/ansi/datetime.sql.out',
            'sql/core/src/test/resources/sql-tests/results/ansi/interval.sql.out',
            'sql/core/src/test/resources/sql-tests/results/datetime-legacy.sql.out',
            'sql/core/src/test/resources/sql-tests/results/datetime.sql.out',
            'sql/core/src/test/resources/sql-tests/results/interval.sql.out',
            'sql/core/src/test/resources/sql-tests/results/typeCoercion/native/promoteStrings.sql.out',
            'sql/core/src/test/scala/org/apache/spark/sql/ColumnExpressionSuite.scala',
            'sql/hive-thriftserver/pom.xml']))

    def test_list_file_commits_with_period_specified(self):
        commits = github_apis.list_file_commits_for('README.md', self._github_owner, self._github_repo, self._github_token,
                                                    since='2011-04-1T05:31:29Z', until='2012-04-01T05:31:29Z')
        self.assertEqual(len(commits), 7)
        sha, commit_date = commits[0]
        self.assertEqual(sha, 'ca64a7ae03f2ba4a965b6f2b55afbd6d9f2a397a')
        self.assertEqual(commit_date, '2012-03-17T20:51:29Z')

    def test_list_workflow_runs(self):
        runs = github_apis.list_workflow_runs(self._github_owner, self._github_repo, self._github_token, nmax=1, testing=True)
        self.assertEqual(len(runs), 1)
        run_id, name, event, pr_number, pr_head, pr_base = runs[0]
        self.assertIsNotNone(run_id)
        self.assertIsNotNone(name)
        self.assertIsNotNone(event)
        self.assertIsNotNone(pr_number)
        self.assertIsNotNone(pr_head)
        self.assertIsNotNone(pr_base)

    def test_list_workflow_jobs(self):
        runs = github_apis.list_workflow_runs(self._github_owner, self._github_repo, self._github_token, nmax=1, testing=True)
        self.assertEqual(len(runs), 1)
        run_id = runs[0][0]
        jobs = github_apis.list_workflow_jobs(run_id, self._github_owner, self._github_repo, self._github_token, nmax=1)
        self.assertEqual(len(jobs), 1)
        job_id, name, conclusion = jobs[0]
        self.assertIsNotNone(job_id)
        self.assertIsNotNone(name)
        self.assertIsNotNone(conclusion)

    def test_list_workflow_job_logs(self):
        runs = github_apis.list_workflow_runs(self._github_owner, self._github_repo, self._github_token, nmax=1, testing=True)
        self.assertEqual(len(runs), 1)
        run_id = runs[0][0]
        jobs = github_apis.list_workflow_jobs(run_id, self._github_owner, self._github_repo, self._github_token, nmax=1)
        self.assertEqual(len(jobs), 1)
        job_id = jobs[0][0]
        logs = github_apis.get_workflow_job_logs(job_id, self._github_owner, self._github_repo, self._github_token)
        self.assertIsNotNone(logs)

    def test_github_datetime(self):
        from datetime import datetime, timezone
        import dateutil.parser as parser
        self.assertEqual(github_apis.from_github_datetime('2019-10-29T05:31:29Z'),
                         parser.parse('2019-10-29T05:31:29Z'))
        self.assertEqual(github_apis.format_github_datetime('2019-10-29T05:31:29Z', '%Y/%m/%d'), '2019/10/29')
        self.assertEqual(github_apis.to_github_datetime(datetime.strptime('2021-08-04', '%Y-%m-%d')),
                         '2021-08-04T00:00:00Z')


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
