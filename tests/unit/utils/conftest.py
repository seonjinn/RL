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

"""Local conftest for ``tests/unit/utils``.

Mirrors ``tests/unit/data/conftest.py`` so individual util-only tests
(e.g. multimodal-payload metrics) can run in lean containers without the
parent ``tests/unit/conftest.py`` fixtures pulling in Ray dashboard +
mlflow / swanlab dependency stacks.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def init_ray_cluster():  # noqa: D401
    yield


@pytest.fixture(scope="session", autouse=True)
def ray_gpu_monitor():  # noqa: D401
    yield


@pytest.fixture(scope="session", autouse=True)
def session_data(_unit_test_data):  # noqa: D401
    yield _unit_test_data
