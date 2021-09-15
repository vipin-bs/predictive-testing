#!/usr/bin/env bash

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

#
# Train a predictive testing model for Apache Spark

FWDIR="$(cd "`dirname $0`"/..; pwd)"

if [ ! -z "$CONDA_ENABLED" ]; then
  # Activate a conda virtual env
  . ${FWDIR}/bin/conda.sh && activate_conda_virtual_env "${FWDIR}" "predictive-testing"
fi

exec python3 -u ${FWDIR}/bin/train-ptesting-model.py \
  --output ${FWDIR}/models/spark \
  --train-log-data ${FWDIR}/models/spark/logs/github-logs.json \
  --build-deps ${FWDIR}/models/spark/indexes/spark-dep-graphs
