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

import os

import ray


REQUIRED_ENV_VAR = "NRL_TEST_INITIALIZER_IMPORT_ENV"
REQUIRED_ENV_VALUE = "available-before-actor-import"

if os.environ.get(REQUIRED_ENV_VAR) != REQUIRED_ENV_VALUE:
    raise ImportError(f"{REQUIRED_ENV_VAR} was unavailable during actor module import")


@ray.remote
class InitializerEnvActor:
    def get_required_env(self) -> str | None:
        return os.environ.get(REQUIRED_ENV_VAR)
