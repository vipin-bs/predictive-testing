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
from datetime import datetime

import github_utils


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
class GitHubUtilsTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._github_token = env['GITHUB_TOKEN']
        # Test repository for GitHub APIs
        # TODO: Might it be better to replace it with a more suitable test repository?
        cls._github_owner = 'maropu'
        cls._github_repo = 'spark'

        # Test data root path
        _resource_path = os.getenv("PREDICTIVE_TESTING_TESTDATA")
        cls._test_data_path = f"{_resource_path}/spark-logs"

    def test_github_datetime(self):
        import dateutil.parser as parser
        self.assertEqual(github_utils.from_github_datetime('2019-10-29T05:31:29Z'),
                         parser.parse('2019-10-29T05:31:29Z'))
        self.assertEqual(github_utils.format_github_datetime('2019-10-29T05:31:29Z', '%Y/%m/%d'), '2019/10/29')
        self.assertEqual(github_utils.to_github_datetime(datetime.strptime('2021-08-04', '%Y-%m-%d')),
                         '2021-08-04T00:00:00Z')

    def test_count_file_updates(self):
        update_counts = github_utils.count_file_updates(
            'README.md',
            '2020-08-04T05:31:29Z',
            days=[365, 1460, 3650],
            owner=self._github_owner,
            repo=self._github_repo,
            token=self._github_token
        )
        self.assertEqual(update_counts, [2, 14, 93])

    def test_create_failed_test_extractor(self):
        test_failure_patterns = [
            "error.+?(org\.apache\.spark\.[a-zA-Z0-9\.]+Suite)",
            "Had test failures in (pyspark\.[a-zA-Z0-9\._]+) with python"
        ]
        compilation_failure_patterns = [
            "error.+? Compilation failed",
            "Failing because of negative scalastyle result"
        ]
        extractor = github_utils._create_failed_test_extractor(test_failure_patterns, compilation_failure_patterns)

        from pathlib import Path
        compilation_failure_logs_path = f"{self._test_data_path}/compilation_failures.log"
        data = Path(compilation_failure_logs_path).read_text()
        self.assertEqual(extractor(data), None)  # 'None' represents compilation failures

        syntax_failure_logs_path = f"{self._test_data_path}/scalastyle_failures.log"
        data = Path(syntax_failure_logs_path).read_text()
        self.assertEqual(extractor(data), None)  # 'None' represents compilation failures

        test_failure_logs_path = f"{self._test_data_path}/test_failures.log"
        data = Path(test_failure_logs_path).read_text()
        self.assertEqual(extractor(data), [
            'org.apache.spark.sql.hive.StatisticsSuite',
            'org.apache.spark.sql.hive.InsertSuite'])

    def test_trim_text(self):
        self.assertEqual(github_utils.trim_text('abcdefghijk', 4), 'abcd...')


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
