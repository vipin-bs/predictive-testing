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
# Predicts a preferred test set with a given predictive model

FWDIR="$(cd "`dirname $0`"/..; pwd)"

if [ -z "${SPARK_REPO}" ]; then
  echo "env SPARK_REPO not defined" 1>&2
  exit 1
fi

if [ -z "$CONDA_DISABLED" ]; then
  # Activate a conda virtual env
  . ${FWDIR}/bin/conda.sh && activate_conda_virtual_env "${FWDIR}"
fi

MODELPATH=${FWDIR}/models/spark

PYTHONPATH="${FWDIR}/python:${FWDIR}/bin" \
exec python3 -u ${FWDIR}/bin/ptesting-model.py \
  --username "<unknown>" \
  --target ${SPARK_REPO} \
  --model ${MODELPATH}/model.pkl \
  --test-files ${MODELPATH}/indexes/latest/test-files.json \
  --commits ${MODELPATH}/logs/commits.json \
  --excluded-tests ${MODELPATH}/logs/excluded-tests.json \
  --included-tests ${MODELPATH}/logs/included-tests.json \
  --failed-tests ${MODELPATH}/failed-tests.json \
  --build-dep ${MODELPATH}/indexes/latest/dep-graph.json \
  --correlated-files ${MODELPATH}/indexes/latest/correlated-files.json \
  --correlated-files-delta ${MODELPATH}/correlated-files-delta.json \
  --updated-file-stats ${MODELPATH}/logs/updated-file-stats.json \
  --contributor-stats ${MODELPATH}/logs/contributor-stats.json \
  "$@"
