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

"""Local conftest for ``tests/unit/models/megatron``.

The parent ``tests/unit/conftest.py`` defines an autouse session fixture
that starts a Ray cluster (with the dashboard) before any unit test
runs. The Megatron multimodal helper tests here only exercise pure
PyTorch and Megatron-LM imports and do not need Ray, so we override the
fixture with a no-op for this directory only. This keeps the tests
runnable in environments (e.g. the standard MCore CI container) that do
not ship the full Ray dashboard dependency stack.
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def init_ray_cluster():  # noqa: D401
    """No-op override of the parent unit-test Ray fixture for this dir."""
    yield


@pytest.fixture(scope="session", autouse=True)
def ray_gpu_monitor():  # noqa: D401
    """No-op override of the parent Ray GPU monitor fixture for this dir."""
    yield


@pytest.fixture(scope="session", autouse=True)
def session_data(_unit_test_data):  # noqa: D401
    """No-op override of the parent session_data fixture.

    The parent fixture's teardown imports ``nemo_rl.utils.logger`` which in
    turn imports a long chain of optional logging dependencies (mlflow,
    swanlab, etc.) that are not always available in test containers. We
    only need ``_unit_test_data`` so basic results-writing works.
    """
    yield _unit_test_data
