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

from ptesting import javaclass


class JavaClassTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _resource_path = os.getenv("PREDICTIVE_TESTING_TESTDATA")
        cls._java_class_test_path = f"{_resource_path}/java-class-test-project"

    def test_javap_exists(self):
        self.assertEqual(javaclass._get_cmd_path('javap'), 'javap')

    def test_no_cmd_exists(self):
        self.assertRaisesRegexp(
            RuntimeError,
            "Could not find 'unknown' executable",
            lambda: javaclass._get_cmd_path('unknown'))

    def test_exec_subprocess(self):
        stdout, stderr, rt = javaclass._exec_subprocess('echo 1')
        r = stdout.decode().replace('\n', '')
        self.assertEqual(r, '1')

    def test_extract_refs(self):
        classes = javaclass.list_classes(self._java_class_test_path, 'io.github.maropu')
        extract_refs_from_path = javaclass.create_func_to_extract_refs_from_class_file('io.github.maropu')

        results = []
        for (clazz, path) in classes:
            refs = extract_refs_from_path(path)
            other_refs = set(filter(lambda r: r != clazz, refs))
            results.append((clazz, sorted(other_refs)))

        expected_results = [
            ('io.github.maropu.BaseClass', []),
            ('io.github.maropu.MainClass', []),
            ('io.github.maropu.MainClass', ['io.github.maropu.TestClassA', 'io.github.maropu.TestClassC']),
            ('io.github.maropu.MainClassSuite', ['io.github.maropu.MainClass']),
            ('io.github.maropu.TestClassA', ['io.github.maropu.BaseClass']),
            ('io.github.maropu.TestClassB', ['io.github.maropu.BaseClass']),
            ('io.github.maropu.TestClassC', ['io.github.maropu.TestClassA', 'io.github.maropu.TestClassB']),
            ('io.github.maropu.TestClassSuite', ['io.github.maropu.TestClassA', 'io.github.maropu.TestClassB'])
        ]
        self.assertEqual(sorted(results), expected_results)


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
