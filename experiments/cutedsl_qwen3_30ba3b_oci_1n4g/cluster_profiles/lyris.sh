#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export CUTEDSL_PROFILE_NAME=lyris
export CUTEDSL_ACCOUNT=coreai_dlalgo_llm
export CUTEDSL_PARTITION=gb200
export CUTEDSL_GRES=
export CUTEDSL_SEGMENT=1
export CUTEDSL_COMMENT=metrics
export CUTEDSL_IMAGE=/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-nightly-stage-lyris-20260714/containers/nemo_rl_nightly_20260713_2370069.sqsh
export CUTEDSL_IMAGE_SHA256=f01387b5a9426ccd4c51822e1c48bd967329b61cd9948fa1faf3eb9c8d3e0c8c
export CUTEDSL_SHARED_HF_HOME=/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home
export CUTEDSL_FUNCTIONAL_TIME=02:00:00
export CUTEDSL_BENCHMARK_TIME=05:00:00
