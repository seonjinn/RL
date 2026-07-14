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

export CUTEDSL_PROFILE_NAME=pre_tyche
export CUTEDSL_ACCOUNT=coreai_dlalgo_llm
export CUTEDSL_PARTITION=batch
export CUTEDSL_GRES=
export CUTEDSL_SEGMENT=1
export CUTEDSL_COMMENT=metrics
export CUTEDSL_IMAGE=/lustre/fsw/coreai_dlalgo_llm/users/sna/nemo-rl-nightly-stage-20260714/containers/nemo_rl_nightly_20260713_2375815.sqsh
export CUTEDSL_IMAGE_SHA256=9fe8a6dcc0a9e3c069555cae22b15c24a7353ae396817e9c603535604bbfd368
export CUTEDSL_SHARED_HF_HOME=/lustre/fsw/coreai_dlalgo_llm/users/sna/hf_home
export CUTEDSL_FUNCTIONAL_TIME=02:00:00
export CUTEDSL_BENCHMARK_TIME=05:00:00
