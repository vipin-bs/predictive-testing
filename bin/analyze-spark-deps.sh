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
# Analyze Spark class dependencies in a coarse-grained way

FWDIR="$(cd "`dirname $0`"/..; pwd)"

if [ ! -z "$CONDA_ENABLED" ]; then
  # Activate a conda virtual env
  . ${FWDIR}/bin/conda.sh && activate_conda_virtual_env "${FWDIR}" "predictive-testing"
fi

ROOT_PATHS="$1"
if [ -z "${ROOT_PATHS}" ]; then
  echo "Root paths not specified and usage: ${0} <root_paths> <output_path>" 1>&2
  exit 1
fi

OUTPUT_PATH="$2"
if [ -z "${OUTPUT_PATH}" ]; then
  echo "Output path not specified and usage: ${0} <root_paths> <output_path>" 1>&2
  exit 1
fi

exec python3 -u ${FWDIR}/bin/build-deps.py \
  --command analyze \
  --file-type java \
  --target-package org.apache.spark \
  --root-paths ${ROOT_PATHS}  \
  --output ${OUTPUT_PATH}
