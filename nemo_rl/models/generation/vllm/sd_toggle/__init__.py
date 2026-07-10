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

"""Runtime roofline model for tail-gated speculative decoding."""

from .config import SDToggleConfig, load_config
from .predict import predict_decision, should_enable_sd
from .roofline import predict_speedup

__all__ = [
    "SDToggleConfig",
    "load_config",
    "predict_decision",
    "predict_speedup",
    "should_enable_sd",
]
