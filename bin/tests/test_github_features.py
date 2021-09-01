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

import github_features


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
class GitHubFeatureTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._github_token = env['GITHUB_TOKEN']
        # Test repository for GitHub APIs
        # TODO: Might it be better to replace it with a more suitable test repository?
        cls._github_owner = 'maropu'
        cls._github_repo = 'spark'

    def test_count_file_updates(self):
        update_counts = github_features.count_file_updates(
            'README.md',
            '2020-08-04T05:31:29Z',
            days=[365, 1460, 3650],
            owner=self._github_owner,
            repo=self._github_repo,
            token=self._github_token
        )
        self.assertEqual(update_counts, [2, 14, 93])


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
