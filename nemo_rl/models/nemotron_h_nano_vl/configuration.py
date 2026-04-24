# --------------------------------------------------------
# Adapted from https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B under MIT License
#     LICENSE is in incl_licenses directory.
# --------------------------------------------------------

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.utils import logging

from .configuration_nemotron_h import NemotronHConfig

logger = logging.get_logger(__name__)


class SoundConfig(PretrainedConfig):
    """Configuration for the sound/audio model (Parakeet encoder + projection)."""

    model_type = "parakeet"

    def __init__(
        self,
        hidden_size: int = 1024,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 24,
        intermediate_size: int = 4096,
        conv_kernel_size: int = 31,
        convolution_bias: bool = False,
        feat_in: int = 80,
        subsampling_factor: int = 8,
        subsampling_conv_channels: int = 256,
        subsampling_conv_kernel_size: int = 3,
        subsampling_conv_stride: int = 2,
        num_mel_bins: int = 128,
        projection_hidden_size: int = 4096,
        projection_bias: bool = True,
        sampling_rate: int = 16000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.conv_kernel_size = conv_kernel_size
        self.convolution_bias = convolution_bias
        self.feat_in = feat_in
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.subsampling_conv_kernel_size = subsampling_conv_kernel_size
        self.subsampling_conv_stride = subsampling_conv_stride
        self.num_mel_bins = num_mel_bins
        self.projection_hidden_size = projection_hidden_size
        self.projection_bias = projection_bias
        self.sampling_rate = sampling_rate


class NemotronH_Nano_VL_V2_Config(PretrainedConfig):
    model_type = "NemotronH_Nano_VL_V2"
    is_composition = True

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        llm_config=None,
        sound_config=None,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        ps_version="v1",
        image_tag_type="internvl",
        projector_hidden_size=4096,
        vit_hidden_size=1280,
        attn_implementation="flash_attention_2",
        sound_context_token_id=None,
        sound_context_token="<so_embedding>",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if vision_config is not None:
            if "auto_map" in vision_config and "AutoConfig" in vision_config["auto_map"]:
                vision_auto_config = get_class_from_dynamic_module(
                    *vision_config["auto_map"]["AutoConfig"].split("--")[::-1]
                )
                self.vision_config = vision_auto_config(**vision_config)
            else:
                self.vision_config = PretrainedConfig(**vision_config)
        else:
            self.vision_config = PretrainedConfig()

        text_cfg = llm_config or text_config
        if text_cfg is not None:
            self.text_config = NemotronHConfig(**text_cfg)
        else:
            self.text_config = NemotronHConfig()

        if sound_config is not None:
            self.sound_config = SoundConfig(**sound_config)
        else:
            self.sound_config = None

        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.ps_version = ps_version
        self.image_tag_type = image_tag_type
        self.projector_hidden_size = projector_hidden_size
        self.vit_hidden_size = vit_hidden_size

        self.sound_context_token_id = sound_context_token_id
        self.sound_context_token = sound_context_token

        self.layers_block_type = self.text_config.layers_block_type

        self._attn_implementation = attn_implementation
        self.vision_config.use_flash_attn = (
            self._attn_implementation is not None
            and "flash_attention" in self._attn_implementation
        )
        self.text_config._attn_implementation = self._attn_implementation
