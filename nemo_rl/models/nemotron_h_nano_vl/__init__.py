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

from transformers import AutoConfig, AutoImageProcessor

from .configuration import NemotronH_Nano_VL_V2_Config
from .configuration_nemotron_h import NemotronHConfig
from .image_processing import NemotronNanoVLV2ImageProcessor


def register() -> None:
    AutoConfig.register("NemotronH_Nano_VL_V2", NemotronH_Nano_VL_V2_Config)
    AutoImageProcessor.register(
        NemotronH_Nano_VL_V2_Config,
        fast_image_processor_class=NemotronNanoVLV2ImageProcessor,
    )
    AutoConfig.register("nemotron_h", NemotronHConfig)
