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

from ptesting import depgraph
from ptesting import javaclass


class DepGraphTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _resource_path = os.getenv("PREDICTIVE_TESTING_TESTDATA")
        cls._java_class_test_path = f"{_resource_path}/java-class-test-project"

    def test_build_call_graphs(self):
        class_path = self._java_class_test_path
        classes = javaclass.list_classes(class_path, 'io.github.maropu')
        extract_edges_from_path = javaclass.create_func_to_extract_refs_from_class_file('io.github.maropu')
        rev_adj_list = depgraph.build_dependency_graphs(classes, extract_edges_from_path)
        _rev_adj_list = sorted(map(lambda kv: (kv[0], sorted(kv[1])), rev_adj_list.items()))
        self.assertEqual(_rev_adj_list, [
            ('io.github.maropu.BaseClass', ['io.github.maropu.TestClassA', 'io.github.maropu.TestClassB']),
            ('io.github.maropu.MainClass', ['io.github.maropu.MainClassSuite']),
            ('io.github.maropu.TestClassA', [
                'io.github.maropu.MainClass',
                'io.github.maropu.TestClassC',
                'io.github.maropu.TestClassSuite'
            ]),
            ('io.github.maropu.TestClassB', ['io.github.maropu.TestClassC', 'io.github.maropu.TestClassSuite']),
            ('io.github.maropu.TestClassC', ['io.github.maropu.MainClass']),
        ])

    def test_select_subgraph(self):
        g = {'A': ['B'], 'B': ['A', 'C'], 'C': ['D'], 'D': ['E'], 'E': ['A']}
        subgraph, subnodes = depgraph.select_subgraph(['A'], g, depth=2)
        self.assertEqual(sorted(map(lambda kv: (kv[0], sorted(kv[1])), subgraph.items())),
                         [('A', ['B']), ('B', ['A', 'C'])])
        self.assertEqual(sorted(subnodes), ['A', 'B', 'C'])

    def test_generate_graph(self):
        edges = {'A': ['B'], 'B': ['A', 'C']}
        g = depgraph.generate_graph(['A', 'B', 'C'], ['A'], edges)

        def normalize(s):
            return s.replace(' ', '').replace('\n', '')

        expected = f"""
            digraph {{
                graph [pad="0.5", nodesep="0.5", ranksep="2", fontname="Helvetica"];
                node [shape=box]
                rankdir=LR;

                "A" [shape="oval"];
                "B";
                "C";
                "A" -> "B";
                "B" -> "A";
                "B" -> "C";
            }}
        """
        self.assertEqual(normalize(g), normalize(expected))


if __name__ == "__main__":
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output="target/test-reports", verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)
