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
# Crawl Spark GitHub logs

if [ -z "${GITHUB_TOKEN}" ]; then
  echo "env GITHUB_TOKEN not defined" 1>&2
  exit 1
fi

FWDIR="$(cd "`dirname $0`"/..; pwd)"

if [ -z "$CONDA_DISABLE" ]; then
  # Activate a conda virtual env
  . ${FWDIR}/bin/conda.sh && activate_conda_virtual_env "${FWDIR}"
fi

PYTHONPATH="${FWDIR}/python" \
exec python3 -u ${FWDIR}/bin/crawl-github-logs.py \
  --github-token ${GITHUB_TOKEN} \
  --github-owner apache \
  --github-repo spark \
  "$@"
