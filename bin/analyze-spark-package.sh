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
# Analyze Spark repository to extract build dependency and test list

FWDIR="$(cd "`dirname $0`"/..; pwd)"

if [ ! -z "$CONDA_ENABLED" ]; then
  # Activate a conda virtual env
  . ${FWDIR}/bin/conda.sh && activate_conda_virtual_env "${FWDIR}"
fi

ROOT_PATH="$1"
if [ -z "${ROOT_PATH}" ]; then
  echo "Spark repository root path not specified and usage: ${0} <root_paths>" 1>&2
  exit 1
fi

OUTPUT_INDEX_NAME=`git -C ${ROOT_PATH} rev-parse --abbrev-ref HEAD`-`git -C ${ROOT_PATH} rev-parse --short HEAD`-`gdate '+%Y%m%d%H%M'`
OUTPUT_PATH=${FWDIR}/models/spark/indexes/${OUTPUT_INDEX_NAME}

echo "Output path is ${OUTPUT_PATH}"
PYTHONPATH="${FWDIR}/python" \
exec python3 -u ${FWDIR}/bin/analyze-spark-package.py \
  --root-path ${ROOT_PATH} \
  --output ${OUTPUT_PATH}
