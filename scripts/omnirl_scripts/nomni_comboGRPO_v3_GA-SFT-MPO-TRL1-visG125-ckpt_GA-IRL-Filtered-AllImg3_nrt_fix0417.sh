#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEMORL="${NEMORL:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

if [[ -f "${NEMORL}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${NEMORL}/.env"
  set +a
fi

CONFIG_PATH="${CONFIG_PATH:-examples/configs/vlmConv3d_grpo_mix_omnirlSDG-videorlSDG-videor1Comm-2minVidFilter-imageCommRB5-aud_nomni_32f_dedup_draco_super.yaml}"
EXP_NAME="${EXP_NAME:-nomni_comboGRPO_v3_GA-SFT-MPO-TRL1-visG125-ckpt_GA-IRL-Filtered-AllImg3_nrt_fix0417}"
RUN_ID="${RUN_ID:-20260418}"
NUM_NODES="${NUM_NODES:-32}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
MICRO_BS="${MICRO_BS:-1}"
LOGPROB_BS="${LOGPROB_BS:-1}"
JOB_CYCLES="${JOB_CYCLES:-8}"
GLOBAL_TRAIN_BATCH_SIZE="${GLOBAL_TRAIN_BATCH_SIZE:-$((NUM_NODES * GRADIENT_ACCUMULATION_STEPS * MICRO_BS * GPUS_PER_NODE))}"
JOB_NAME="${JOB_NAME:-${EXP_NAME}_n${NUM_NODES}_bs${GLOBAL_TRAIN_BATCH_SIZE}_ga${GRADIENT_ACCUMULATION_STEPS}_j${RUN_ID}}"
JOB_HASH="${JOB_HASH:-$(printf '%s' "${JOB_NAME}" | openssl dgst -sha1 -binary | od -An -tx1 | tr -d ' \n' | cut -c1-12)}"

MODEL_NAME="${OMNI_GRPO_MODEL_NAME:-${MODEL_NAME:-}}"
TRAIN_DATA_PATH="${OMNI_GRPO_TRAIN_DATA_PATH:-${TRAIN_DATA_PATH:-}}"
: "${MODEL_NAME:?Set OMNI_GRPO_MODEL_NAME or MODEL_NAME, or define it in ${NEMORL}/.env}"
: "${TRAIN_DATA_PATH:?Set OMNI_GRPO_TRAIN_DATA_PATH or TRAIN_DATA_PATH, or define it in ${NEMORL}/.env}"

RESULTS_ROOT="${RESULTS_ROOT:-${NEMORL}/../jobs}"
RESULTS_DIR="${RESULTS_DIR:-${RESULTS_ROOT}/${JOB_NAME}}"
LOGS_DIR="${LOGS_DIR:-${RESULTS_DIR}/logs}"
mkdir -p "${LOGS_DIR}" "${RESULTS_DIR}"
export BASE_LOG_DIR="${BASE_LOG_DIR:-${LOGS_DIR}}"
export OBJECT_STORE_MEMORY="${OBJECT_STORE_MEMORY:-300000000000}"

if [[ ! -f "${NEMORL}/${CONFIG_PATH}" ]]; then
  echo "Config not found: ${NEMORL}/${CONFIG_PATH}" >&2
  exit 1
fi
if [[ ! -f "${NEMORL}/ray.sub" ]]; then
  echo "ray.sub not found under NEMORL=${NEMORL}" >&2
  exit 1
fi

SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-llmservice_fm_vision}"
SBATCH_TIME="${SBATCH_TIME:-04:00:00}"
if [[ -z "${SBATCH_PARTITION:-}" ]]; then
  if [[ "$(hostname)" == *"draco-oci"* ]]; then
    SBATCH_PARTITION="batch_block1,batch_block3,batch_block4,backfill_block1,backfill_block2,backfill_block3,backfill_block4"
  elif [[ "$(hostname)" == *"cw-dfw"* ]]; then
    SBATCH_PARTITION="batch,backfill,batch_short"
  elif [[ "$(hostname)" == *"cs-oci-ord"* ]]; then
    SBATCH_PARTITION="backfill_block1,grizzly,polar,polar3,polar4"
  elif [[ "$(hostname)" == *"oci-nrt"* ]]; then
    SBATCH_PARTITION="batch_block1"
  else
    SBATCH_PARTITION="batch,batch_large,batch_large_long,batch_long"
  fi
fi

CONTAINER_ROOT="${CONTAINER_ROOT:-/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_vision/users/hanrongy/project/nemotron_omni/rl/images}"
export CONTAINER="${CONTAINER:-${CONTAINER_ROOT}/super-omni-rl-20260501-vllm0.18.sqsh}"
export MOUNTS="${MOUNTS:-/lustre:/lustre,/home}"
export NUM_NODES

export NRL_FORCE_REBUILD_VENVS="${NRL_FORCE_REBUILD_VENVS:-false}"
export NEMO_RL_VENV_DIR="${NEMO_RL_VENV_DIR:-/opt/ray_venvs}"
export NRL_VENVS_TRUST_EXISTING="${NRL_VENVS_TRUST_EXISTING:-1}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"

export CACHE_ROOT="${CACHE_ROOT:-${NEMORL}/.cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export NRL_MEGATRON_CHECKPOINT_DIR="${NRL_MEGATRON_CHECKPOINT_DIR:-${HF_HOME}/nemo_rl}"
export TMPDIR="${TMPDIR:-/tmp/nrl-${JOB_HASH}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR}/triton}"
mkdir -p "${HF_HOME}" "${HF_MODULES_CACHE}" "${NRL_MEGATRON_CHECKPOINT_DIR}" "${TMPDIR}" "${TRITON_CACHE_DIR}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NVTE_FWD_LAYERNORM_SM_MARGIN="${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}"
export NVTE_BWD_LAYERNORM_SM_MARGIN="${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}"
export NEMO_RL_LOG_GPU_MEMORY="${NEMO_RL_LOG_GPU_MEMORY:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export NRL_IGNORE_VERSION_MISMATCH="${NRL_IGNORE_VERSION_MISMATCH:-true}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800000}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-1000}"
export NRL_DEBUG="${NRL_DEBUG:-0}"
export USE_REPO_VLLM="${USE_REPO_VLLM:-0}"
export NRL_PATCH_CONTAINER_VLLM="${NRL_PATCH_CONTAINER_VLLM:-1}"
export NRL_VLLM_VIDEO_AS_IMAGES="${NRL_VLLM_VIDEO_AS_IMAGES:-0}"
export NRL_VLLM_VIDEO_FRAME_SEPARATORS="${NRL_VLLM_VIDEO_FRAME_SEPARATORS:-0}"

SEED="${SEED:-$(printf '%s' "train:${JOB_NAME}" | openssl dgst -md5 -binary | od -An -tu4 -N4 | xargs)}"
WANDB_PROJECT="${WANDB_PROJECT:-Nemotron-omni-RL}"
MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-24576}"
NUM_FRAMES="${NUM_FRAMES:-32}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
VLLM_LOAD_FORMAT="${VLLM_LOAD_FORMAT:-auto}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-false}"
OFFLOAD_OPTIMIZER_FOR_LOGPROB="${OFFLOAD_OPTIMIZER_FOR_LOGPROB:-false}"
SEQUENCE_PACKING_ENABLED="${SEQUENCE_PACKING_ENABLED:-false}"

EXTRA_OVERRIDES=""
if [[ -n "${WANDB_RUN_ID:-}" ]]; then
  EXTRA_OVERRIDES+=" +logger.wandb.id=${WANDB_RUN_ID} +logger.wandb.resume=${WANDB_RESUME:-must}"
fi
if [[ -n "${EXTRA_HYDRA_OVERRIDES:-}" ]]; then
  EXTRA_OVERRIDES+=" ${EXTRA_HYDRA_OVERRIDES}"
fi

if [[ "${NRL_PATCH_CONTAINER_VLLM}" == "1" && "${USE_REPO_VLLM}" != "1" ]]; then
  VLLM_PATCH_SETUP=$(cat <<EOS
python - <<'PY'
import os
import pathlib
import shutil
import sys

repo_root = pathlib.Path("${NEMORL}")
src_root = repo_root / "3rdparty" / "vllm" / "vllm"
target_candidates = [
    pathlib.Path("/opt/nemo-rl/3rdparty/vllm/vllm"),
]
target_candidates.extend(pathlib.Path(p) / "vllm" for p in sys.path)
target_candidates.extend(
    pathlib.Path("/opt/nemo_rl_venv").glob("lib*/python*/site-packages/vllm")
)
targets = []
seen = set()
for target in target_candidates:
    target = target.resolve()
    if target in seen or not (target / "__init__.py").exists():
        continue
    seen.add(target)
    targets.append(target)
if not targets:
    print("[NRL_PATCH_CONTAINER_VLLM] no installed vLLM package found to patch", flush=True)
else:
    for dst_root in targets:
        for rel in (
            "transformers_utils/processors/internvl.py",
            "transformers_utils/processors/nano_nemotron_vl.py",
        ):
            src = src_root / rel
            dst = dst_root / rel
            if not src.exists():
                raise FileNotFoundError(src)
            if not dst.exists():
                if rel.startswith("transformers_utils/processors/"):
                    dst.parent.mkdir(parents=True, exist_ok=True)
                else:
                    print(f"[NRL_PATCH_CONTAINER_VLLM] skip missing target {dst}", flush=True)
                    continue
            shutil.copy2(src, dst)
            print(f"[NRL_PATCH_CONTAINER_VLLM] patched {dst} from {src}", flush=True)

        if os.environ.get("NRL_COPY_REPO_VLLM_MODEL", "0") == "1":
            for rel in (
                "model_executor/models/parakeet.py",
                "multimodal/audio.py",
                "multimodal/media/audio.py",
                "transformers_utils/configs/parakeet.py",
            ):
                src = src_root / rel
                dst = dst_root / rel
                if not src.exists():
                    raise FileNotFoundError(src)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                print(f"[NRL_PATCH_CONTAINER_VLLM] patched {dst} from {src}", flush=True)

        processor_path = dst_root / "multimodal/processing/processor.py"
        if processor_path.exists():
            proc_text = processor_path.read_text()
            proc_orig = proc_text

            if "import torch" not in proc_text:
                proc_text = proc_text.replace(
                    "import regex as re\n",
                    "import regex as re\nimport torch\n",
                    1,
                )

            old_placeholder_info = """@dataclass
class PlaceholderFeaturesInfo:
    modality: str
    item_idx: int
    start_idx: int
    tokens: list[int]

    @property
"""
            new_placeholder_info = """@dataclass
class PlaceholderFeaturesInfo:
    modality: str
    item_idx: int
    start_idx: int
    tokens: list[int]
    is_embed: torch.Tensor | None = None

    @property
"""
            if old_placeholder_info in proc_text:
                proc_text = proc_text.replace(
                    old_placeholder_info, new_placeholder_info, 1
                )

            old_to_range = """        return PlaceholderRange(
            offset=self.start_idx,
            length=self.length,
        )
"""
            new_to_range = """        return PlaceholderRange(
            offset=self.start_idx,
            length=self.length,
            is_embed=self.is_embed,
        )
"""
            if old_to_range in proc_text:
                proc_text = proc_text.replace(old_to_range, new_to_range, 1)

            old_placeholder_yield = """                    yield PlaceholderFeaturesInfo(
                        modality=modality,
                        item_idx=item_idx,
                        start_idx=start_idx,
                        tokens=content_tokens_full,
                    )
"""
            new_placeholder_yield = """                    content_is_embed = content.is_embed
                    if content_is_embed is not None:
                        content_is_embed = content_is_embed(tokenizer, content.full)

                    yield PlaceholderFeaturesInfo(
                        modality=modality,
                        item_idx=item_idx,
                        start_idx=start_idx,
                        tokens=content_tokens_full,
                        is_embed=content_is_embed,
                    )
"""
            if old_placeholder_yield in proc_text:
                proc_text = proc_text.replace(
                    old_placeholder_yield, new_placeholder_yield, 1
                )

            if proc_text != proc_orig:
                processor_path.write_text(proc_text)
                print(
                    f"[NRL_PATCH_CONTAINER_VLLM] patched {processor_path} "
                    "for PromptUpdateDetails.is_embed propagation",
                    flush=True,
                )
            print(
                "[NRL_PATCH_CONTAINER_VLLM] processor is_embed support "
                f"path={processor_path} "
                f"placeholder_field={'is_embed: torch.Tensor | None' in proc_text} "
                f"range_forward={'is_embed=self.is_embed' in proc_text} "
                f"content_mask={'content_is_embed = content.is_embed' in proc_text}",
                flush=True,
            )

        model_path = dst_root / "model_executor/models/nano_nemotron_vl.py"
        if not model_path.exists():
            print(f"[NRL_PATCH_CONTAINER_VLLM] skip missing target {model_path}", flush=True)
            continue

        model_src = src_root / "model_executor/models/nano_nemotron_vl.py"
        if model_src.exists() and os.environ.get("NRL_COPY_REPO_VLLM_MODEL", "0") == "1":
            shutil.copy2(model_src, model_path)
            print(f"[NRL_PATCH_CONTAINER_VLLM] patched {model_path} from {model_src}", flush=True)
            continue

        text = model_path.read_text()

        def replace_once(old: str, new: str, marker: str) -> None:
            global text
            if old in text:
                text = text.replace(old, new, 1)
            elif marker not in text:
                raise RuntimeError(
                    f"Could not patch {model_path}: missing block for {marker}"
                )

        old_prompt_update_import = """    PromptReplacement,
    PromptUpdate,
)
"""
        new_prompt_update_import = """    PromptReplacement,
    PromptUpdateDetails,
    PromptUpdate,
)
"""
        if old_prompt_update_import in text:
            text = text.replace(old_prompt_update_import, new_prompt_update_import, 1)
        elif "PromptUpdateDetails" not in text:
            raise RuntimeError(
                f"Could not patch {model_path}: missing PromptUpdate import block"
            )

        if "import os\n" not in text:
            text = text.replace("import math\n", "import math\nimport os\n", 1)

        base_mm_processor_code = """class NanoNemotronBaseVLMultiModalProcessor(BaseMultiModalProcessor):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        dynamic_tiler = getattr(
            self,
            "is_dynamic_tiler",
            getattr(
                self.info,
                "is_dynamic_tiler",
                getattr(self, "dynamic_resolution", False),
            ),
        )
        if dynamic_tiler and "num_tokens_per_image" in hf_inputs:
            pixel_values_flat = MultiModalFieldConfig.batched("image")
        else:
            pixel_values_flat = MultiModalFieldConfig.flat_from_sizes(
                "image", hf_inputs.get("image_num_patches", torch.empty(0))
            )

        fields = dict(
            pixel_values_flat=pixel_values_flat,
            image_num_patches=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
            num_tokens_per_image=MultiModalFieldConfig.batched("image"),
            imgs_sizes=MultiModalFieldConfig.batched("image"),
        )
        for key, config in list(fields.items()):
            if isinstance(config, tuple):
                if len(config) != 1:
                    raise TypeError(
                        "Expected single field config tuple for "
                        f"{key}, got len={len(config)}"
                    )
                fields[key] = config[0]

        return fields

    def _get_prompt_repl_image(
        self,
        mm_items,
        hf_processor,
        out_mm_data,
    ):
        def get_mm_item_value(name: str, item_idx: int) -> object | None:
            values = out_mm_data.get(name)
            if values is None:
                return None
            if torch.is_tensor(values):
                if values.ndim == 0:
                    return values
                return values[item_idx]
            return values[item_idx]

        if "image_num_patches" in out_mm_data:
            image_num_patches = out_mm_data["image_num_patches"]
            assert isinstance(image_num_patches, torch.Tensor)
            image_num_patches = image_num_patches.tolist()
        elif "image_embeds" in out_mm_data:
            image_num_patches = [None] * len(out_mm_data["image_embeds"])
        else:
            image_num_patches = []

        def get_image_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                feature_size = images.get_feature_size(item_idx)
            elif "num_tokens_per_image" in out_mm_data:
                feature_size = out_mm_data["num_tokens_per_image"][item_idx]
            elif "image_num_patches" in out_mm_data or "num_patches" in out_mm_data:
                patch_count = get_mm_item_value("image_num_patches", item_idx)
                if patch_count is None:
                    patch_count = get_mm_item_value("num_patches", item_idx)
                assert patch_count is not None
                if torch.is_tensor(patch_count):
                    patch_count = int(patch_count.item())
                else:
                    patch_count = int(patch_count)

                if patch_count == 0:
                    feature_size = 0
                elif "imgs_sizes" in out_mm_data:
                    imgs_size = get_mm_item_value("imgs_sizes", item_idx)
                    assert imgs_size is not None
                    target_h, target_w = imgs_size
                    patch_size = int(getattr(hf_processor.config, "patch_size", 16))
                    downsample_ratio = float(
                        getattr(hf_processor.config, "downsample_ratio", 0.5)
                    )
                    feature_size = int(
                        (int(target_h) // patch_size)
                        * downsample_ratio
                        * (int(target_w) // patch_size)
                        * downsample_ratio
                    )
                else:
                    image_size = images.get_image_size(item_idx)
                    max_num_tiles = hf_processor.max_num_tiles
                    tokens_per_patch = hf_processor.get_num_image_tokens(
                        image_width=image_size.width,
                        image_height=image_size.height,
                        max_num_tiles=max_num_tiles,
                    )
                    feature_size = int(patch_count * tokens_per_patch)
            else:
                image_size = images.get_image_size(item_idx)
                max_num_tiles = hf_processor.max_num_tiles
                feature_size = hf_processor.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                    max_num_tiles=max_num_tiles,
                )

            num_patches = None
            local_image_num_patches = image_num_patches
            if isinstance(local_image_num_patches, torch.Tensor):
                local_image_num_patches = local_image_num_patches.tolist()
            if isinstance(local_image_num_patches, (list, tuple)) and item_idx < len(
                local_image_num_patches
            ):
                num_patches = int(local_image_num_patches[item_idx])

            if torch.is_tensor(feature_size):
                feature_size = int(feature_size.item())
            else:
                feature_size = int(feature_size)

            return hf_processor.get_image_repl(feature_size, num_patches)

        return PromptReplacement(
            modality="image",
            target="<image>",
            replacement=get_image_replacement,
        )

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs,
        out_mm_kwargs,
    ):
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        out_mm_data = out_mm_kwargs.get_data()
        return [
            self._get_prompt_repl_image(mm_items, hf_processor, out_mm_data),
        ]

"""
        patch_base_processor = (
            os.environ.get("NRL_PATCH_NANO_BASE_PROCESSOR", "0")
            .strip()
            .lower()
            in ("1", "true", "yes", "on")
        )
        if (
            patch_base_processor
            and "NanoNemotronBaseVLMultiModalProcessor = BaseMultiModalProcessor"
            in text
        ):
            text = text.replace(
                "NanoNemotronBaseVLMultiModalProcessor = BaseMultiModalProcessor\n",
                base_mm_processor_code,
                1,
            )
        elif (
            patch_base_processor
            and
            "NanoNemotronBaseVLMultiModalProcessor" in text
            and (
                "class NanoNemotronBaseVLMultiModalProcessor(" not in text
                or (
                    text.find("class NanoNemotronVLMultiModalProcessor") != -1
                    and text.find("class NanoNemotronBaseVLMultiModalProcessor(")
                    > text.find("class NanoNemotronVLMultiModalProcessor")
                )
            )
        ):
            subclass_pos = text.find("class NanoNemotronVLMultiModalProcessor")
            if subclass_pos != -1:
                text = text[:subclass_pos] + base_mm_processor_code + text[subclass_pos:]
            else:
                text = text.replace(
                    "logger = init_logger(__name__)\n",
                    "logger = init_logger(__name__)\n\n" + base_mm_processor_code,
                    1,
                )
        base_class_pos = text.find("class NanoNemotronBaseVLMultiModalProcessor(")
        base_alias_pos = text.find(
            "NanoNemotronBaseVLMultiModalProcessor = BaseMultiModalProcessor\n"
        )
        subclass_pos = text.find("class NanoNemotronVLMultiModalProcessor")
        if (
            patch_base_processor
            and subclass_pos != -1
            and (base_class_pos == -1 or base_class_pos > subclass_pos)
        ):
            text = text[:subclass_pos] + base_mm_processor_code + text[subclass_pos:]
        elif (
            not patch_base_processor
            and "NanoNemotronBaseVLMultiModalProcessor" in text
            and subclass_pos != -1
            and (base_class_pos == -1 or base_class_pos > subclass_pos)
            and (base_alias_pos == -1 or base_alias_pos > subclass_pos)
        ):
            text = (
                text[:subclass_pos]
                + "NanoNemotronBaseVLMultiModalProcessor = BaseMultiModalProcessor\n\n"
                + text[subclass_pos:]
            )
        if os.environ.get("NRL_DEBUG", "0") == "1":
            base_class_pos = text.find("class NanoNemotronBaseVLMultiModalProcessor(")
            base_alias_pos = text.find(
                "NanoNemotronBaseVLMultiModalProcessor = BaseMultiModalProcessor\n"
            )
            subclass_pos = text.find("class NanoNemotronVLMultiModalProcessor")
            print(
                "[NRL_PATCH_CONTAINER_VLLM] "
                "base multimodal processor "
                f"real_class={'class NanoNemotronBaseVLMultiModalProcessor(' in text} "
                f"alias={'NanoNemotronBaseVLMultiModalProcessor = BaseMultiModalProcessor' in text} "
                f"base_pos={base_class_pos} alias_pos={base_alias_pos} "
                f"subclass_pos={subclass_pos}",
                flush=True,
            )

        new_get_hf_processor = """    def get_hf_processor(self, **kwargs: object) -> NanoNemotronVLProcessor:
        processor = self.ctx.init_processor(
            NanoNemotronVLProcessor,
            config=self.get_hf_config(),
            tokenizer=self.get_tokenizer(),
            video_token=self.get_video_token(),
            video_pruning_rate=self.get_video_pruning_rate(),
            max_model_len=self.ctx.model_config.max_model_len,
            **kwargs,
        )
        use_frame_separators = (
            os.environ.get("NRL_VLLM_VIDEO_FRAME_SEPARATORS", "0")
            .strip()
            .lower()
            in ("1", "true", "yes", "on")
        )
        if not use_frame_separators:

            def get_video_repl_plain(
                *,
                tokens_per_frame: list[int],
                frames_indices: list[int],
                frame_duration_ms: int,
                tokenizer,
                img_start_token_ids: list[int],
                img_end_token_ids: list[int],
                img_context_token_ids: list[int],
                video_temporal_patch_size: int = 1,
            ) -> PromptUpdateDetails[list[int]]:
                del frames_indices, frame_duration_ms
                all_token_ids: list[int] = []
                for num_tokens in tokens_per_frame:
                    all_token_ids.extend(img_start_token_ids)
                    all_token_ids.extend(img_context_token_ids * num_tokens)
                    all_token_ids.extend(img_end_token_ids)

                def is_embed(tokenizer, full):
                    token_ids = (
                        full
                        if isinstance(full, list)
                        else tokenizer.encode(full, add_special_tokens=False)
                    )
                    return torch.isin(
                        torch.tensor(token_ids),
                        torch.tensor(img_context_token_ids),
                    )

                embed_token_count = sum(tokens_per_frame) * len(img_context_token_ids)
                if os.environ.get("NRL_DEBUG", "0") == "1":
                    print(
                        "[VLLM_NATIVE_VIDEO_REPL_PROCESSOR_PLAIN] "
                        "frame_separators=0 "
                        f"full_tokens={len(all_token_ids)} "
                        f"embed_tokens={embed_token_count} "
                        f"text_tokens={len(all_token_ids) - embed_token_count} "
                        f"tubelets={len(tokens_per_frame)} "
                        f"T={video_temporal_patch_size}",
                        flush=True,
                    )

                return PromptUpdateDetails(
                    full=all_token_ids,
                    is_embed=is_embed,
                )

            processor.get_video_repl = get_video_repl_plain

        return processor
"""
        if "VLLM_NATIVE_VIDEO_REPL_PROCESSOR_PLAIN" not in text:
            class_pos = text.find("class NanoNemotronVLProcessingInfo")
            method_start = text.find("    def get_hf_processor(", class_pos)
            method_end = text.find("\n    @cached_property", method_start)
            if class_pos == -1 or method_start == -1 or method_end == -1:
                raise RuntimeError(
                    f"Could not patch {model_path}: missing get_hf_processor block"
                )
            text = (
                text[:method_start]
                + new_get_hf_processor
                + text[method_end:]
            )

        replace_once(
            """    @classmethod
    def get_cached_feature_size(cls, image: Image.Image) -> int:
        feature_size = cls.feature_size_cache[id(image)]
        # hard assert that we only use the feature size once
        del cls.feature_size_cache[id(image)]
        return feature_size
""",
            """    @classmethod
    def get_cached_feature_size(cls, image: Image.Image) -> int:
        image_id = id(image)
        if image_id in cls.feature_size_cache:
            feature_size = cls.feature_size_cache[image_id]
            # hard assert that we only use the feature size once
            del cls.feature_size_cache[image_id]
            return feature_size

        by_size = getattr(cls, "feature_size_cache_by_size", {})
        image_size = getattr(image, "size", None)
        if image_size in by_size:
            return int(by_size[image_size])

        unique_feature_sizes = set(by_size.values())
        if len(unique_feature_sizes) == 1:
            feature_size = int(next(iter(unique_feature_sizes)))
            if os.environ.get("NRL_DEBUG", "0") == "1":
                print(
                    "[VLLM_PRECOMPUTED_PATH] "
                    "falling back to the sole cached feature size "
                    f"{feature_size} for image_id={image_id} image_size={image_size}",
                    flush=True,
                )
            return feature_size

        last_feature_size = getattr(cls, "last_precomputed_feature_size", None)
        if last_feature_size is not None:
            feature_size = int(last_feature_size)
            if os.environ.get("NRL_DEBUG", "0") == "1":
                print(
                    "[VLLM_PRECOMPUTED_PATH] "
                    "falling back to the last precomputed feature size "
                    f"{feature_size} for image_id={image_id} image_size={image_size}",
                    flush=True,
                )
            return feature_size

        raise KeyError(image_id)
""",
            "feature_size_cache_by_size",
        )

        cls_pos = text.find("class NanoNemotronVLProcessor")
        init_pos = text.find("    def __init__(", cls_pos)
        call_pos = text.find("    def __call__(", init_pos)
        init_sig_end = text.find("    ) -> None:", init_pos)
        if init_pos != -1 and init_sig_end != -1 and (call_pos == -1 or init_sig_end < call_pos):
            init_sig = text[init_pos:init_sig_end]
            missing_init_kwargs = []
            if "max_num_patches" not in init_sig:
                missing_init_kwargs.append(
                    "        max_num_patches: int | None = None,"
                )
            if "precomputed_imgs_sizes" not in init_sig:
                missing_init_kwargs.append(
                    "        precomputed_imgs_sizes: list[list[int]] | None = None,"
                )
            if "video_as_images" not in init_sig:
                missing_init_kwargs.append(
                    "        video_as_images: bool | None = None,"
                )
            if "**kwargs" not in init_sig:
                missing_init_kwargs.append("        **kwargs: object,")
            if missing_init_kwargs:
                text = (
                    text[:init_sig_end]
                    + "\n"
                    + "\n".join(missing_init_kwargs)
                    + text[init_sig_end:]
                )
        elif "class NanoNemotronVLProcessor" in text:
            raise RuntimeError(
                f"Could not patch {model_path}: missing NanoNemotronVLProcessor.__init__ signature"
            )

        replace_once(
            """    def _preprocess_image(
        self,
        text: list[str],
        images: list[Image.Image],
        max_num_tiles: int,
    ) -> tuple[list[str], dict[str, Any]]:
""",
            """    def _preprocess_image(
        self,
        text: list[str],
        images: list[Image.Image],
        max_num_tiles: int | None,
        use_fast_preprocessing: bool | None = None,
        max_num_patches: int | None = None,
        precomputed_imgs_sizes: list[list[int]] | None = None,
        video_as_images: bool | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
""",
            "precomputed_imgs_sizes",
        )

        replace_once(
            """        if tiler := self.dynamic_tiler:
            sans_images = text[0].replace("<image>", "")
            text_prompt_length = len(
                self.tokenizer(sans_images, add_special_tokens=False).input_ids
            )
            pixel_values_lst, num_tokens_per_image = tiler._images_to_pixel_values_lst(
                text_prompt_length=text_prompt_length,
                images=images,
            )
            imgs_sizes = [(pv.shape[-2], pv.shape[-1]) for pv in pixel_values_lst]
            normalized = [
                input_conditioner(img, tiler.norm_mean, tiler.norm_std)
                for img in pixel_values_lst
            ]
            image_num_patches = torch.tensor([1] * len(num_tokens_per_image))
            image_inputs = {
                "pixel_values_flat": normalized,
                "imgs_sizes": imgs_sizes,
                "num_tokens_per_image": num_tokens_per_image,
            }
        else:
""",
            """        tiler = None if max_num_tiles is not None else self.dynamic_tiler
        if tiler:
            sans_images = text[0].replace("<image>", "")
            text_prompt_length = len(
                self.tokenizer(sans_images, add_special_tokens=False).input_ids
            )
            if precomputed_imgs_sizes is not None and len(precomputed_imgs_sizes) == len(images):
                normalized = []
                imgs_sizes = []
                num_tokens_per_image = []
                DynamicResolutionImageTiler.feature_size_cache_by_size = {}
                DynamicResolutionImageTiler.last_precomputed_feature_size = None
                for image, (target_h, target_w) in zip(images, precomputed_imgs_sizes, strict=True):
                    target_h, target_w = int(target_h), int(target_w)
                    image_arr = np.asarray(
                        image.convert("RGB") if image.mode != "RGB" else image,
                        dtype=np.uint8,
                    )
                    image_tensor = torch.from_numpy(image_arr).unsqueeze(0).permute(0, 3, 1, 2)
                    resized = torch.nn.functional.interpolate(
                        image_tensor,
                        size=(target_h, target_w),
                        mode="bicubic",
                        align_corners=False,
                        antialias=True,
                    ) / 255.0
                    normalized.append(input_conditioner(resized[0], tiler.norm_mean, tiler.norm_std))
                    imgs_sizes.append((target_h, target_w))
                    num_tokens = tiler._get_num_embeddings(target_w, target_h)
                    num_tokens_per_image.append(num_tokens)
                    DynamicResolutionImageTiler.feature_size_cache[id(image)] = num_tokens
                    DynamicResolutionImageTiler.last_precomputed_feature_size = num_tokens
                    if not hasattr(DynamicResolutionImageTiler, "feature_size_cache_by_size"):
                        DynamicResolutionImageTiler.feature_size_cache_by_size = {}
                    DynamicResolutionImageTiler.feature_size_cache_by_size[(image.width, image.height)] = num_tokens
                    DynamicResolutionImageTiler.feature_size_cache_by_size[(target_w, target_h)] = num_tokens
                if os.environ.get("NRL_DEBUG", "0") == "1":
                    print(
                        "[VLLM_PRECOMPUTED_PATH] "
                        f"using {len(imgs_sizes)} HF-resolved image sizes; "
                        f"first_sizes={imgs_sizes[:3]}",
                        flush=True,
                    )
            else:
                old_max_num_patches = tiler._max_num_patches
                if max_num_patches is not None:
                    tiler._max_num_patches = (
                        max_num_patches if max_num_patches > 0 else float("inf")
                    )
                try:
                    pixel_values_lst, num_tokens_per_image = tiler._images_to_pixel_values_lst(
                        text_prompt_length=text_prompt_length,
                        images=images,
                    )
                finally:
                    tiler._max_num_patches = old_max_num_patches
                imgs_sizes = [(pv.shape[-2], pv.shape[-1]) for pv in pixel_values_lst]
                normalized = [
                    input_conditioner(img, tiler.norm_mean, tiler.norm_std)
                    for img in pixel_values_lst
                ]
            image_num_patches = torch.tensor([1] * len(num_tokens_per_image))
            image_inputs = {
                "pixel_values_flat": normalized,
                "imgs_sizes": imgs_sizes,
                "num_tokens_per_image": num_tokens_per_image,
            }
        elif (
            max_num_tiles == 1
            and getattr(self, "video_target_num_patches", None) is not None
            and len(images) > 1
            and bool(video_as_images)
        ):
            temporal_patch_size = self.video_temporal_patch_size
            num_images = len(images)
            num_tubelets = math.ceil(num_images / temporal_patch_size)
            num_padded = num_tubelets * temporal_patch_size
            if num_padded > num_images:
                images = list(images) + [images[-1]] * (num_padded - num_images)
            patch_size = self.config.patch_size
            downsample_ratio = self.config.downsample_ratio
            target_patches = self.video_target_num_patches
            orig_w, orig_h = images[0].width, images[0].height

            if precomputed_imgs_sizes is not None and len(precomputed_imgs_sizes) > 0:
                if len(precomputed_imgs_sizes) != num_images:
                    raise ValueError(
                        "video_as_images expected one precomputed target per frame, "
                        f"got {len(precomputed_imgs_sizes)} for {num_images}"
                    )
                unique_sizes = {
                    (int(height), int(width))
                    for height, width in precomputed_imgs_sizes
                }
                if len(unique_sizes) != 1:
                    raise ValueError(
                        "video_as_images expects one resize target across frames, "
                        f"got {sorted(unique_sizes)}"
                    )
                target_h, target_w = next(iter(unique_sizes))
            elif self.video_maintain_aspect_ratio:
                target_w, target_h = _compute_aspect_preserving_size(
                    orig_w=orig_w,
                    orig_h=orig_h,
                    target_num_patches=target_patches,
                    patch_size=patch_size,
                    downsample_ratio=downsample_ratio,
                )
            else:
                reduction_factor = int(round(1 / downsample_ratio))
                side = int(math.sqrt(target_patches))
                side = max(reduction_factor, (side // reduction_factor) * reduction_factor)
                target_w = side * patch_size
                target_h = side * patch_size

            frame_tensors = []
            for image in images:
                image_arr = np.asarray(
                    image.convert("RGB") if image.mode != "RGB" else image,
                    dtype=np.uint8,
                )
                image_tensor = torch.from_numpy(image_arr).unsqueeze(0).permute(0, 3, 1, 2)
                resized = torch.nn.functional.interpolate(
                    image_tensor,
                    size=(target_h, target_w),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                ) / 255.0
                frame_tensors.append(resized[0])
            pixel_values_flat = input_conditioner(
                torch.stack(frame_tensors), self.norm_mean, self.norm_std
            )
            image_num_patches = torch.tensor([
                temporal_patch_size if (i % temporal_patch_size == 0) else 0
                for i in range(num_images)
            ])
            tokens_per_tubelet = int(
                (target_h // patch_size)
                * downsample_ratio
                * (target_w // patch_size)
                * downsample_ratio
            )
            num_tokens_per_image = [
                tokens_per_tubelet if (i % temporal_patch_size == 0) else 0
                for i in range(num_images)
            ]
            image_inputs = {
                "pixel_values_flat": pixel_values_flat,
                "image_num_patches": image_num_patches,
                "imgs_sizes": [(target_h, target_w)] * num_images,
            }
            if os.environ.get("NRL_DEBUG", "0") == "1":
                print(
                    "[VLLM_VIDEO_AS_IMAGES] "
                    f"orig=({orig_w}x{orig_h}) target=({target_w}x{target_h}) "
                    f"frames={num_images} T={temporal_patch_size} "
                    f"tokens_per_tubelet={tokens_per_tubelet} "
                    f"num_tokens_per_image={num_tokens_per_image[:8]}",
                    flush=True,
                )
        else:
            max_num_tiles = max_num_tiles or self.max_num_tiles
""",
            "VLLM_PRECOMPUTED_PATH",
        )

        replace_once(
            """        for i, (feature_size, num_patches) in enumerate(
            zip(num_tokens_per_image, image_num_patches, strict=True)
        ):
            image_repl = self.get_image_repl(feature_size, num_patches)
            parts[i] = parts[i].replace("<image>", image_repl.full)
""",
            """        image_idx = 0
        for part_idx, part in enumerate(parts):
            if part != "<image>":
                continue
            feature_size = num_tokens_per_image[image_idx]
            num_patches = image_num_patches[image_idx]
            image_repl = self.get_image_repl(feature_size, num_patches)
            parts[part_idx] = image_repl.full
            image_idx += 1
""",
            "image_idx = 0",
        )

        text = text.replace(
            'return PromptUpdateDetails.select_text("", IMG_CONTEXT)',
            'return PromptUpdateDetails.select_text(IMG_START + IMG_END, IMG_CONTEXT)',
        )
        text = text.replace(
            'return PromptUpdateDetails.from_seq("")',
            'return PromptUpdateDetails.select_text(IMG_START + IMG_END, IMG_CONTEXT)',
        )
        if "VLLM_ZERO_IMAGE_REPL_WRAPPER" not in text:
            image_repl_anchor = "        repl_features = IMG_CONTEXT * feature_size\n"
            if image_repl_anchor not in text:
                raise RuntimeError(
                    f"Could not patch {processor_path}: missing get_image_repl anchor"
                )
            text = text.replace(
                image_repl_anchor,
                """        # VLLM_ZERO_IMAGE_REPL_WRAPPER: keep a wrapper-only placeholder so
        # vLLM's multimodal item accounting still sees secondary video frames.
        if feature_size == 0:
            patch_count = 0 if num_patches is None else int(num_patches)
            if patch_count == 0:
                return PromptUpdateDetails.select_text(IMG_START + IMG_END, IMG_CONTEXT)

"""
                + image_repl_anchor,
                1,
            )

        old_prompt_feature_block = """            tiler = self.use_dynamic_tiler(hf_processor_mm_kwargs)
            if isinstance(images, ImageEmbeddingItems):
                feature_size = images.get_feature_size(item_idx)
            elif tiler:
                image = images.get(item_idx)
                feature_size = tiler.get_cached_feature_size(image)
            elif (
"""
        new_prompt_feature_block = """            tiler = self.use_dynamic_tiler(hf_processor_mm_kwargs)
            if isinstance(images, ImageEmbeddingItems):
                feature_size = images.get_feature_size(item_idx)
            elif "num_tokens_per_image" in out_mm_data:
                # VLLM_PRECOMPUTED_REPL_FEATURE_SIZE: precomputed/dynamic
                # image paths do not populate DynamicResolutionImageTiler's
                # side cache, so use the HF processor output directly.
                tokens_per_image = out_mm_data["num_tokens_per_image"]
                if torch.is_tensor(tokens_per_image):
                    feature_size = int(tokens_per_image[item_idx].item())
                else:
                    feature_size = int(tokens_per_image[item_idx])
            elif tiler:
                image = images.get(item_idx)
                feature_size = tiler.get_cached_feature_size(image)
            elif (
"""
        if (
            "VLLM_PRECOMPUTED_REPL_FEATURE_SIZE" not in text
            and "num_tokens_per_image_data is not None" not in text
        ):
            if old_prompt_feature_block in text:
                text = text.replace(
                    old_prompt_feature_block,
                    new_prompt_feature_block,
                    1,
                )
            elif """            elif tiler:
                image = images.get(item_idx)
                feature_size = tiler.get_cached_feature_size(image)
            elif (
""" in text:
                text = text.replace(
                    """            elif tiler:
                image = images.get(item_idx)
                feature_size = tiler.get_cached_feature_size(image)
            elif (
""",
                    """            elif "num_tokens_per_image" in out_mm_data:
                # VLLM_PRECOMPUTED_REPL_FEATURE_SIZE: precomputed/dynamic
                # image paths do not populate DynamicResolutionImageTiler's
                # side cache, so use the HF processor output directly.
                tokens_per_image = out_mm_data["num_tokens_per_image"]
                if torch.is_tensor(tokens_per_image):
                    feature_size = int(tokens_per_image[item_idx].item())
                else:
                    feature_size = int(tokens_per_image[item_idx])
            elif tiler:
                image = images.get(item_idx)
                feature_size = tiler.get_cached_feature_size(image)
            elif (
""",
                    1,
                )
            elif """                image = images.get(item_idx)
                feature_size = tiler.get_cached_feature_size(image)
""" in text:
                text = text.replace(
                    """                image = images.get(item_idx)
                feature_size = tiler.get_cached_feature_size(image)
""",
                    """                if "num_tokens_per_image" in out_mm_data:
                    # VLLM_PRECOMPUTED_REPL_FEATURE_SIZE: precomputed/dynamic
                    # image paths do not populate DynamicResolutionImageTiler's
                    # side cache, so use the HF processor output directly.
                    tokens_per_image = out_mm_data["num_tokens_per_image"]
                    if torch.is_tensor(tokens_per_image):
                        feature_size = int(tokens_per_image[item_idx].item())
                    else:
                        feature_size = int(tokens_per_image[item_idx])
                else:
                    image = images.get(item_idx)
                    feature_size = tiler.get_cached_feature_size(image)
""",
                    1,
                )
            else:
                print(
                    "[NRL_PATCH_CONTAINER_VLLM] warning: prompt feature-size "
                    f"cache block not found in {model_path}; continuing",
                    flush=True,
                )

        raw_prompt_cache_block = """                image = images.get(item_idx)
                feature_size = tiler.get_cached_feature_size(image)
"""
        patched_prompt_cache_block = """                if "num_tokens_per_image" in out_mm_data:
                    # VLLM_PRECOMPUTED_REPL_FEATURE_SIZE_DIRECT: use HF
                    # processor output when vLLM prompt items are not the
                    # same PIL objects that populated the tiler side cache.
                    tokens_per_image = out_mm_data["num_tokens_per_image"]
                    if torch.is_tensor(tokens_per_image):
                        feature_size = int(tokens_per_image[item_idx].item())
                    else:
                        feature_size = int(tokens_per_image[item_idx])
                elif (
                    "image_num_patches" in out_mm_data
                    or "num_patches" in out_mm_data
                ):
                    # VLLM_VIDEO_AS_IMAGES_REPL_FEATURE_SIZE: flattened videos
                    # use the static image path, while dynamic-resolution model
                    # configs still make the tiler branch truthy. Compute the static
                    # replacement size from the same precomputed frame target.
                    nrl_image_num_patches = out_mm_data.get(
                        "image_num_patches", out_mm_data.get("num_patches")
                    )
                    if torch.is_tensor(nrl_image_num_patches):
                        patch_count = int(nrl_image_num_patches[item_idx].item())
                    else:
                        patch_count = int(nrl_image_num_patches[item_idx])
                    if patch_count == 0:
                        feature_size = 0
                    else:
                        precomputed_sizes = hf_processor_mm_kwargs.get(
                            "precomputed_imgs_sizes"
                        )
                        if precomputed_sizes:
                            target_h, target_w = precomputed_sizes[
                                min(item_idx, len(precomputed_sizes) - 1)
                            ]
                            patch_size = int(getattr(hf_processor, "patch_size", 16))
                            downsample_ratio = float(
                                getattr(hf_processor, "downsample_ratio", 0.5)
                            )
                            tokens_per_patch = int(
                                (int(target_h) // patch_size)
                                * downsample_ratio
                                * (int(target_w) // patch_size)
                                * downsample_ratio
                            )
                        else:
                            image = images.get(item_idx)
                            image_size = getattr(image, "size", None)
                            max_num_tiles = hf_processor_mm_kwargs.get(
                                "max_num_tiles"
                            ) or getattr(hf_processor, "max_num_tiles", 1)
                            if image_size is not None and hasattr(
                                hf_processor, "get_num_image_tokens"
                            ):
                                tokens_per_patch = int(
                                    hf_processor.get_num_image_tokens(
                                        image_width=image_size[0],
                                        image_height=image_size[1],
                                        max_num_tiles=max_num_tiles,
                                    )
                                )
                            else:
                                tokens_per_patch = int(
                                    getattr(hf_processor, "num_image_token", 256)
                                )
                            if patch_count not in (0, 1):
                                tokens_per_patch = max(
                                    1, tokens_per_patch // patch_count
                                )
                        if bool(hf_processor_mm_kwargs.get("video_as_images")):
                            feature_size = int(tokens_per_patch)
                        else:
                            feature_size = int(patch_count * tokens_per_patch)
                    if (
                        os.environ.get("NRL_DEBUG", "0") == "1"
                        and item_idx < 2
                    ):
                        print(
                            "[VLLM_STATIC_IMAGE_REPL] "
                            f"item_idx={item_idx} patch_count={patch_count} "
                            f"feature_size={feature_size} "
                            f"out_keys={sorted(out_mm_data.keys())} "
                            f"mm_kw_keys={sorted(hf_processor_mm_kwargs.keys())}",
                            flush=True,
                        )
                else:
                    precomputed_sizes = hf_processor_mm_kwargs.get(
                        "precomputed_imgs_sizes"
                    )
                    if precomputed_sizes:
                        target_h, target_w = precomputed_sizes[
                            min(item_idx, len(precomputed_sizes) - 1)
                        ]
                        patch_size = int(getattr(hf_processor, "patch_size", 16))
                        downsample_ratio = float(
                            getattr(hf_processor, "downsample_ratio", 0.5)
                        )
                        feature_size = int(
                            (int(target_h) // patch_size)
                            * downsample_ratio
                            * (int(target_w) // patch_size)
                            * downsample_ratio
                        )
                        if (
                            os.environ.get("NRL_DEBUG", "0") == "1"
                            and item_idx < 2
                        ):
                            print(
                                "[VLLM_PRECOMPUTED_REPL_FALLBACK] "
                                f"item_idx={item_idx} feature_size={feature_size} "
                                f"out_keys={sorted(out_mm_data.keys())} "
                                f"mm_kw_keys={sorted(hf_processor_mm_kwargs.keys())}",
                                flush=True,
                            )
                    else:
                        image = images.get(item_idx)
                        if (
                            os.environ.get("NRL_DEBUG", "0") == "1"
                            and item_idx < 2
                        ):
                            print(
                                "[VLLM_TILER_CACHE_REPL_FALLBACK] "
                                f"item_idx={item_idx} "
                                f"out_keys={sorted(out_mm_data.keys())} "
                                f"mm_kw_keys={sorted(hf_processor_mm_kwargs.keys())}",
                                flush=True,
                            )
                        feature_size = tiler.get_cached_feature_size(image)
"""
        if raw_prompt_cache_block in text:
            text = text.replace(raw_prompt_cache_block, patched_prompt_cache_block)

        line_patch_marker = "VLLM_PROMPT_CACHE_LINE_PATCH"
        if line_patch_marker not in text:
            patched_lines = []
            prompt_cache_line_patches = 0
            for line in text.splitlines():
                if line.strip() != "feature_size = tiler.get_cached_feature_size(image)":
                    patched_lines.append(line)
                    continue

                indent = line[: len(line) - len(line.lstrip())]
                patched_lines.extend(
                    [
                        f"{indent}# {line_patch_marker}: tolerate precomputed/static image paths.",
                        f"{indent}if \"num_tokens_per_image\" in out_mm_data:",
                        f"{indent}    tokens_per_image = out_mm_data[\"num_tokens_per_image\"]",
                        f"{indent}    if torch.is_tensor(tokens_per_image):",
                        f"{indent}        feature_size = int(tokens_per_image[item_idx].item())",
                        f"{indent}    else:",
                        f"{indent}        feature_size = int(tokens_per_image[item_idx])",
                        f"{indent}elif \"image_num_patches\" in out_mm_data or \"num_patches\" in out_mm_data:",
                        f"{indent}    nrl_image_num_patches = out_mm_data.get(",
                        f"{indent}        \"image_num_patches\", out_mm_data.get(\"num_patches\")",
                        f"{indent}    )",
                        f"{indent}    if torch.is_tensor(nrl_image_num_patches):",
                        f"{indent}        patch_count = int(nrl_image_num_patches[item_idx].item())",
                        f"{indent}    else:",
                        f"{indent}        patch_count = int(nrl_image_num_patches[item_idx])",
                        f"{indent}    if patch_count == 0:",
                        f"{indent}        feature_size = 0",
                        f"{indent}    else:",
                        f"{indent}        precomputed_sizes = hf_processor_mm_kwargs.get(\"precomputed_imgs_sizes\")",
                        f"{indent}        if precomputed_sizes:",
                        f"{indent}            target_h, target_w = precomputed_sizes[min(item_idx, len(precomputed_sizes) - 1)]",
                        f"{indent}            patch_size = int(getattr(hf_processor, \"patch_size\", 16))",
                        f"{indent}            downsample_ratio = float(getattr(hf_processor, \"downsample_ratio\", 0.5))",
                        f"{indent}            tokens_per_patch = int(",
                        f"{indent}                (int(target_h) // patch_size)",
                        f"{indent}                * downsample_ratio",
                        f"{indent}                * (int(target_w) // patch_size)",
                        f"{indent}                * downsample_ratio",
                        f"{indent}            )",
                        f"{indent}        else:",
                        f"{indent}            image_size = getattr(image, \"size\", None)",
                        f"{indent}            max_num_tiles = hf_processor_mm_kwargs.get(\"max_num_tiles\") or getattr(hf_processor, \"max_num_tiles\", 1)",
                        f"{indent}            if image_size is not None and hasattr(hf_processor, \"get_num_image_tokens\"):",
                        f"{indent}                tokens_per_patch = int(",
                        f"{indent}                    hf_processor.get_num_image_tokens(",
                        f"{indent}                        image_width=image_size[0],",
                        f"{indent}                        image_height=image_size[1],",
                        f"{indent}                        max_num_tiles=max_num_tiles,",
                        f"{indent}                    )",
                        f"{indent}                )",
                        f"{indent}            else:",
                        f"{indent}                tokens_per_patch = int(getattr(hf_processor, \"num_image_token\", 256))",
                        f"{indent}            if patch_count not in (0, 1):",
                        f"{indent}                tokens_per_patch = max(1, tokens_per_patch // patch_count)",
                        f"{indent}        if bool(hf_processor_mm_kwargs.get(\"video_as_images\")):",
                        f"{indent}            feature_size = int(tokens_per_patch)",
                        f"{indent}        else:",
                        f"{indent}            feature_size = int(patch_count * tokens_per_patch)",
                        f"{indent}else:",
                        f"{indent}    precomputed_sizes = hf_processor_mm_kwargs.get(\"precomputed_imgs_sizes\")",
                        f"{indent}    if precomputed_sizes:",
                        f"{indent}        target_h, target_w = precomputed_sizes[min(item_idx, len(precomputed_sizes) - 1)]",
                        f"{indent}        patch_size = int(getattr(hf_processor, \"patch_size\", 16))",
                        f"{indent}        downsample_ratio = float(getattr(hf_processor, \"downsample_ratio\", 0.5))",
                        f"{indent}        feature_size = int(",
                        f"{indent}            (int(target_h) // patch_size)",
                        f"{indent}            * downsample_ratio",
                        f"{indent}            * (int(target_w) // patch_size)",
                        f"{indent}            * downsample_ratio",
                        f"{indent}        )",
                        f"{indent}    else:",
                        f"{indent}        feature_size = tiler.get_cached_feature_size(image)",
                    ]
                )
                prompt_cache_line_patches += 1

            if prompt_cache_line_patches:
                text = "\n".join(patched_lines) + ("\n" if text.endswith("\n") else "")
                print(
                    "[NRL_PATCH_CONTAINER_VLLM] "
                    f"line-patched prompt cache calls={prompt_cache_line_patches}",
                    flush=True,
                )

        remaining_raw_prompt_cache = text.count(raw_prompt_cache_block)
        if remaining_raw_prompt_cache:
            raise RuntimeError(
                f"Could not patch all prompt feature-size cache blocks in "
                f"{model_path}; remaining={remaining_raw_prompt_cache}"
            )
        if os.environ.get("NRL_DEBUG", "0") == "1":
            prompt_feature_hits = text.count("VLLM_PRECOMPUTED_REPL_FEATURE_SIZE")
            raw_feature_hits = text.count("feature_size = tiler.get_cached_feature_size(image)")
            print(
                "[NRL_PATCH_CONTAINER_VLLM] "
                f"prompt feature-size patches={prompt_feature_hits} "
                f"raw cache calls={raw_feature_hits}",
                flush=True,
            )

        replace_once(
            """        max_num_tiles: int | None = None,
    ) -> BatchFeature:
        # Use default if not provided
        if max_num_tiles is None:
            max_num_tiles = self.max_num_tiles

        text, images, videos, audios = [
            self._make_batch_input(x) for x in (text, images, videos, audios)
        ]
""",
            """        max_num_tiles: int | None = None,
        use_fast_preprocessing: bool | None = None,
        max_num_patches: int | None = None,
        precomputed_imgs_sizes: list[list[int]] | None = None,
        video_as_images: bool | None = None,
    ) -> BatchFeature:
        text, images, videos, audios = [
            self._make_batch_input(x) for x in (text, images, videos, audios)
        ]
""",
            "video_as_images",
        )

        replace_once(
            """        text, image_inputs = self._preprocess_image(
            text=text,
            images=images,
            max_num_tiles=max_num_tiles,
        )
""",
            """        text, image_inputs = self._preprocess_image(
            text=text,
            images=images,
            max_num_tiles=max_num_tiles,
            use_fast_preprocessing=use_fast_preprocessing,
            max_num_patches=max_num_patches,
            precomputed_imgs_sizes=precomputed_imgs_sizes,
            video_as_images=video_as_images,
        )
""",
            "precomputed_imgs_sizes=precomputed_imgs_sizes",
        )

        replace_once(
            """        if self.dynamic_resolution:
            pixel_values_flat = DynamicResolutionImageTiler.stack(
                kwargs.pop("pixel_values_flat"), self.patch_size
            )
            return NanoNemotronVLImagePixelInputsDynamic(
                pixel_values_flat=pixel_values_flat, **kwargs
            )
        else:
            return NanoNemotronVLImagePixelInputs(
                num_patches=kwargs.pop("image_num_patches"), **kwargs
            )
""",
            """        pixel_values_flat = kwargs.pop("pixel_values_flat", None)
        if pixel_values_flat is None:
            return None

        has_dynamic = "imgs_sizes" in kwargs and "num_tokens_per_image" in kwargs
        has_static = "image_num_patches" in kwargs
        if has_dynamic:
            if not torch.is_tensor(pixel_values_flat) or pixel_values_flat.ndim == 4:
                pixel_values_flat = DynamicResolutionImageTiler.stack(
                    pixel_values_flat, self.patch_size
                )
            return NanoNemotronVLImagePixelInputsDynamic(
                pixel_values_flat=pixel_values_flat, **kwargs
            )

        if not has_static:
            raise ValueError(
                "Expected either dynamic image keys "
                "(imgs_sizes, num_tokens_per_image) or static image_num_patches, "
                f"got keys={sorted(kwargs.keys())}"
            )

        kwargs.pop("imgs_sizes", None)
        return NanoNemotronVLImagePixelInputs(
            pixel_values_flat=pixel_values_flat,
            num_patches=kwargs.pop("image_num_patches"),
            **kwargs,
        )
""",
            "has_dynamic",
        )

        field_config_marker = "VLLM_STATIC_VIDEO_AS_IMAGES_FIELD_CONFIG"
        inline_field_config_marker = (
            "VLLM_STATIC_VIDEO_AS_IMAGES_INLINE_FIELD_CONFIG"
        )
        field_pos = text.find(
            "    def _get_image_fields_config(self, hf_inputs: BatchFeature):"
        )
        field_end_candidates = [
            pos
            for pos in (
                text.find("    def _get_video_fields_config", field_pos),
                text.find("    def _get_audio_fields_config", field_pos),
                text.find("    def _get_mm_fields_config", field_pos),
            )
            if pos != -1
        ]
        if field_pos != -1 and field_end_candidates:
            field_end = min(field_end_candidates)
            field_block = text[field_pos:field_end]
            if field_config_marker not in field_block:
                if "pixel_values_flat" not in field_block or "MultiModalFieldConfig" not in field_block:
                    raise RuntimeError(
                        f"Could not patch {model_path}: unexpected _get_image_fields_config block"
                    )
                text = (
                    text[:field_pos]
                    + """    def _get_image_fields_config(self, hf_inputs: BatchFeature):
        dynamic_tiler = getattr(
            self,
            "is_dynamic_tiler",
            getattr(
                self.info,
                "is_dynamic_tiler",
                getattr(self, "dynamic_resolution", False),
            ),
        )
        if dynamic_tiler and "num_tokens_per_image" in hf_inputs:
            pixel_values_flat = MultiModalFieldConfig.batched("image")
        else:
            image_num_patches = hf_inputs.get("image_num_patches", torch.empty(0))
            # VLLM_STATIC_VIDEO_AS_IMAGES_FIELD_CONFIG: static video-as-images
            # uses zero patch counts for secondary frame placeholders, so
            # flat_from_sizes groups each primary frame with its temporal partner.
            pixel_values_flat = MultiModalFieldConfig.flat_from_sizes(
                "image", image_num_patches
            )

        return dict(
            pixel_values_flat=pixel_values_flat,
            image_num_patches=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
            num_tokens_per_image=MultiModalFieldConfig.batched("image"),
            imgs_sizes=MultiModalFieldConfig.batched("image"),
        )

"""
                    + text[field_end:]
                )
        elif field_config_marker not in text:
            if inline_field_config_marker not in text:
                lines = text.splitlines()
                patched_lines = []
                inline_field_patches = 0
                i = 0
                while i < len(lines):
                    line = lines[i]
                    stripped = line.strip()
                    indent = line[: len(line) - len(line.lstrip())]

                    if (
                        "pixel_values_flat" in stripped
                        and stripped.endswith("(")
                        and i + 1 < len(lines)
                        and 'MultiModalFieldConfig.batched("image")'
                        in lines[i + 1]
                    ):
                        prefix = stripped[:-1]
                        close_expr = (
                            ")"
                            if prefix.strip().startswith("pixel_values_flat =")
                            else "),"
                        )
                        patched_lines.extend(
                            [
                                f"{indent}{prefix}(",
                                f'{indent}    MultiModalFieldConfig.batched("image")',
                                f"{indent}    if getattr(",
                                f'{indent}        self, "is_dynamic_tiler",',
                                f"{indent}        getattr(",
                                f'{indent}            self.info, "is_dynamic_tiler",',
                                f'{indent}            getattr(self, "dynamic_resolution", False),',
                                f"{indent}        ),",
                                f"{indent}    )",
                                f'{indent}    and "num_tokens_per_image" in hf_inputs',
                                f"{indent}    else MultiModalFieldConfig.flat_from_sizes(",
                                f'{indent}        "image",',
                                f'{indent}        hf_inputs.get("image_num_patches", torch.empty(0)),',
                                f"{indent}    )",
                                f"{indent}{close_expr}  # {inline_field_config_marker}",
                            ]
                        )
                        inline_field_patches += 1
                        i += 1
                        while i < len(lines):
                            tail = lines[i].strip()
                            i += 1
                            if tail == ")," or tail.endswith("),"):
                                break
                        continue

                    if (
                        "pixel_values_flat" in stripped
                        and 'MultiModalFieldConfig.batched("image")' in stripped
                    ):
                        prefix = stripped.split("MultiModalFieldConfig.batched", 1)[0]
                        close_expr = (
                            ")"
                            if prefix.strip().startswith("pixel_values_flat =")
                            else "),"
                        )
                        patched_lines.extend(
                            [
                                f"{indent}{prefix}(",
                                f'{indent}    MultiModalFieldConfig.batched("image")',
                                f"{indent}    if getattr(",
                                f'{indent}        self, "is_dynamic_tiler",',
                                f"{indent}        getattr(",
                                f'{indent}            self.info, "is_dynamic_tiler",',
                                f'{indent}            getattr(self, "dynamic_resolution", False),',
                                f"{indent}        ),",
                                f"{indent}    )",
                                f'{indent}    and "num_tokens_per_image" in hf_inputs',
                                f"{indent}    else MultiModalFieldConfig.flat_from_sizes(",
                                f'{indent}        "image",',
                                f'{indent}        hf_inputs.get("image_num_patches", torch.empty(0)),',
                                f"{indent}    )",
                                f"{indent}{close_expr}  # {inline_field_config_marker}",
                            ]
                        )
                        inline_field_patches += 1
                        i += 1
                        if (
                            i < len(lines)
                            and lines[i].strip() == "if self.info.is_dynamic_tiler"
                        ):
                            i += 1
                            while i < len(lines):
                                tail = lines[i].strip()
                                i += 1
                                if tail.endswith("),") or tail == "),":
                                    break
                        continue

                    patched_lines.append(line)
                    i += 1

                if inline_field_patches == 0:
                    candidates = [
                        item.strip()
                        for item in lines
                        if "pixel_values" in item and "MultiModalFieldConfig" in item
                    ][:20]
                    raise RuntimeError(
                        f"Could not patch {model_path}: missing image field config "
                        f"block; candidates={candidates}"
                    )
                text = "\n".join(patched_lines) + (
                    "\n" if text.endswith("\n") else ""
                )
                if os.environ.get("NRL_DEBUG", "0") == "1":
                    print(
                        "[NRL_PATCH_CONTAINER_VLLM] "
                        f"inline image field-config patches={inline_field_patches}",
                        flush=True,
                    )

        unwrap_field_config_marker = (
            "VLLM_STATIC_VIDEO_AS_IMAGES_UNWRAP_FIELD_CONFIG"
        )
        mm_fields_pos = text.find("    def _get_mm_fields_config")
        mm_field_end_candidates = [
            pos
            for pos in (
                text.find("\n    def ", mm_fields_pos + 1),
                text.find("\nclass ", mm_fields_pos + 1),
            )
            if pos != -1
        ]
        super_field_config_marker = (
            "VLLM_STATIC_VIDEO_AS_IMAGES_SUPER_FIELD_CONFIG"
        )
        if (
            mm_fields_pos != -1
            and "super()._get_mm_fields_config" in text[
                mm_fields_pos : (
                    min(mm_field_end_candidates)
                    if mm_field_end_candidates
                    else len(text)
                )
            ]
            and super_field_config_marker not in text
        ):
            mm_fields_end = (
                min(mm_field_end_candidates)
                if mm_field_end_candidates
                else len(text)
            )
            text = (
                text[:mm_fields_pos]
                + """    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # VLLM_STATIC_VIDEO_AS_IMAGES_SUPER_FIELD_CONFIG: the container build
        # references a missing base processor for image field config. Build the
        # per-modality field config directly so native video dummy inputs work.
        if hasattr(self, "_get_image_fields_config"):
            fields = self._get_image_fields_config(hf_inputs)
        else:
            dynamic_tiler = getattr(
                self,
                "is_dynamic_tiler",
                getattr(
                    self.info,
                    "is_dynamic_tiler",
                    getattr(self, "dynamic_resolution", False),
                ),
            )
            if dynamic_tiler and "num_tokens_per_image" in hf_inputs:
                pixel_values_flat = MultiModalFieldConfig.batched("image")
            else:
                pixel_values_flat = MultiModalFieldConfig.flat_from_sizes(
                    "image", hf_inputs.get("image_num_patches", torch.empty(0))
                )
            fields = dict(
                pixel_values_flat=pixel_values_flat,
                image_num_patches=MultiModalFieldConfig.batched("image"),
                image_embeds=MultiModalFieldConfig.batched("image"),
                num_tokens_per_image=MultiModalFieldConfig.batched("image"),
                imgs_sizes=MultiModalFieldConfig.batched("image"),
            )
        if self.info.supports_video:
            if hasattr(self, "_get_video_fields_config"):
                fields |= self._get_video_fields_config(hf_inputs)
            else:
                video_num_patches = hf_inputs.get("video_num_patches", torch.empty(0))
                fields |= dict(
                    pixel_values_flat_video=MultiModalFieldConfig.flat_from_sizes(
                        "video", video_num_patches
                    ),
                    video_num_patches=MultiModalFieldConfig.batched("video"),
                    frames_indices=MultiModalFieldConfig.batched("video"),
                    frame_duration_ms=MultiModalFieldConfig.batched("video"),
                )
        if self.info.supports_audio:
            if hasattr(self, "_get_audio_fields_config"):
                fields |= self._get_audio_fields_config(hf_inputs)
            else:
                audio_num_clips = torch.as_tensor(hf_inputs["audio_num_clips"])
                fields |= dict(
                    input_audio_features=MultiModalFieldConfig.flat_from_sizes(
                        "audio", audio_num_clips
                    ),
                    feature_attention_mask=MultiModalFieldConfig.flat_from_sizes(
                        "audio", audio_num_clips
                    ),
                    audio_num_clips=MultiModalFieldConfig.batched(
                        "audio", keep_on_cpu=True
                    ),
                )

        return fields

"""
                + text[mm_fields_end:]
            )
            mm_fields_pos = text.find("    def _get_mm_fields_config")
            mm_field_end_candidates = [
                pos
                for pos in (
                    text.find("\n    def ", mm_fields_pos + 1),
                    text.find("\nclass ", mm_fields_pos + 1),
                )
                if pos != -1
            ]
        if mm_fields_pos != -1 and unwrap_field_config_marker not in text:
            mm_fields_end = (
                min(mm_field_end_candidates)
                if mm_field_end_candidates
                else len(text)
            )
            mm_fields_block = text[mm_fields_pos:mm_fields_end]
            unwrap_field_config_code = """        for key, config in list(fields.items()):
            if isinstance(config, tuple):
                if len(config) != 1:
                    raise TypeError(
                        "Expected single field config tuple for "
                        f"{key}, got len={len(config)}"
                    )
                if os.environ.get("NRL_DEBUG", "0") == "1":
                    print(
                        "[NRL_PATCH_CONTAINER_VLLM_FIELD_CONFIG] "
                        f"unwrapped tuple config for {key}",
                        flush=True,
                    )
                fields[key] = config[0]

        # VLLM_STATIC_VIDEO_AS_IMAGES_UNWRAP_FIELD_CONFIG: older container builds
        # may leave a one-element tuple around inline conditional field configs.
        return fields
"""
            return_fields = "        return fields\n"
            if return_fields in mm_fields_block:
                mm_fields_block = mm_fields_block.replace(
                    return_fields,
                    unwrap_field_config_code,
                    1,
                )
            else:
                mm_lines = mm_fields_block.splitlines()
                rewritten_return_dict = False
                for return_idx, return_line in enumerate(mm_lines):
                    stripped_return = return_line.strip()
                    if not stripped_return.startswith("return dict("):
                        continue

                    return_indent = return_line[
                        : len(return_line) - len(return_line.lstrip())
                    ]
                    mm_lines[return_idx] = return_line.replace(
                        "return dict(", "fields = dict(", 1
                    )
                    paren_depth = 0
                    end_idx = None
                    for line_idx in range(return_idx, len(mm_lines)):
                        paren_depth += mm_lines[line_idx].count("(")
                        paren_depth -= mm_lines[line_idx].count(")")
                        if paren_depth == 0:
                            end_idx = line_idx
                            break
                    if end_idx is None:
                        raise RuntimeError(
                            f"Could not patch {model_path}: unterminated "
                            "_get_mm_fields_config return dict"
                        )
                    unwrap_lines = unwrap_field_config_code.rstrip("\n").splitlines()
                    if return_indent != "        ":
                        unwrap_lines = [
                            return_indent + line[8:] if line.startswith("        ") else line
                            for line in unwrap_lines
                        ]
                    mm_lines[end_idx + 1 : end_idx + 1] = unwrap_lines
                    mm_fields_block = "\n".join(mm_lines) + (
                        "\n" if mm_fields_block.endswith("\n") else ""
                    )
                    rewritten_return_dict = True
                    break

                if not rewritten_return_dict:
                    snippet = mm_fields_block[:1200]
                    raise RuntimeError(
                        f"Could not patch {model_path}: missing _get_mm_fields_config "
                        f"return; block head={snippet!r}"
                    )
            text = text[:mm_fields_pos] + mm_fields_block + text[mm_fields_end:]

        if os.environ.get("NRL_DEBUG", "0") == "1":
            for marker in (
                field_config_marker,
                inline_field_config_marker,
                unwrap_field_config_marker,
            ):
                marker_pos = text.find(marker)
                if marker_pos == -1:
                    continue
                snippet_start = max(0, marker_pos - 700)
                snippet_start = text.rfind("\n", 0, snippet_start) + 1
                snippet_end = text.find("\n", marker_pos + 700)
                if snippet_end == -1:
                    snippet_end = len(text)
                print(
                    "[NRL_PATCH_CONTAINER_VLLM_FIELD_SNIPPET] "
                    f"marker={marker}\n{text[snippet_start:snippet_end]}",
                    flush=True,
                )

        old_process_image_input = """    def _process_image_input(
        self, image_input: NanoNemotronVLImagePixelInputs
    ) -> tuple[torch.Tensor, ...]:
        image_embeds = self.extract_feature(image_input["pixel_values_flat"])
        num_patches = image_input["num_patches"]

        # Only one image in the current batch
        if len(num_patches) == 1:
            return (image_embeds.view(-1, self.config.text_config.hidden_size),)

        # NOTE: Image embeddings are split into separate tensors for each image
        # by the size of each embedding.
        feature_size = image_embeds.shape[1]
        image_embeds = image_embeds.view(-1, self.config.text_config.hidden_size)
        image_feature_sizes = [
            num_patches * feature_size for num_patches in num_patches
        ]
        return image_embeds.split(image_feature_sizes)
"""
        old_process_image_input_static_video = """    def _process_image_input(
        self, image_input: NanoNemotronVLImagePixelInputs
    ) -> tuple[torch.Tensor, ...]:
        pixel_values = image_input["pixel_values_flat"]
        num_patches = image_input["num_patches"]
        hidden_size = self.config.text_config.hidden_size
        temporal_patch_size = self.video_temporal_patch_size

        if (
            temporal_patch_size > 1
            and num_patches.numel() > 0
            and ((num_patches == 0).any() or (num_patches == temporal_patch_size).all())
        ):
            nonzero_items = int((num_patches > 0).sum().item())
            video_embeds = self.extract_feature(
                pixel_values[:total_requested_frames],
                num_frames=total_requested_frames,
            ).view(-1, hidden_size)
            if nonzero_items == 0:
                feature_size = 0
            elif video_embeds.shape[0] % nonzero_items != 0:
                raise ValueError(
                    "Static video-as-images produced an embedding count that "
                    "cannot be split across temporal tubelets: "
                    f"embeds={video_embeds.shape[0]}, items={nonzero_items}, "
                    f"num_patches={num_patches.tolist()}"
                )
            else:
                feature_size = video_embeds.shape[0] // nonzero_items

            results: list[torch.Tensor] = []
            embed_offset = 0
            for patch_count_tensor in num_patches:
                patch_count = int(patch_count_tensor.item())
                if patch_count == 0:
                    results.append(
                        torch.empty(
                            0,
                            hidden_size,
                            device=pixel_values.device,
                            dtype=torch.bfloat16,
                        )
                    )
                    continue

                item_embeds = video_embeds[embed_offset : embed_offset + feature_size]
                embed_offset += feature_size
                results.append(item_embeds)

            return tuple(results)

        image_embeds = self.extract_feature(pixel_values)

        # Only one image in the current batch
        if len(num_patches) == 1:
            return (image_embeds.view(-1, hidden_size),)

        # NOTE: Image embeddings are split into separate tensors for each image
        # by the size of each embedding.
        feature_size = image_embeds.shape[1]
        image_embeds = image_embeds.view(-1, hidden_size)
        image_feature_sizes = [
            num_patches * feature_size for num_patches in num_patches
        ]
        return image_embeds.split(image_feature_sizes)
"""
        new_process_image_input = """    def _process_image_input(
        self, image_input: NanoNemotronVLImagePixelInputs
    ) -> tuple[torch.Tensor, ...]:
        pixel_values = image_input["pixel_values_flat"]
        num_patches = image_input["num_patches"]
        hidden_size = self.config.text_config.hidden_size
        temporal_patch_size = self.video_temporal_patch_size

        if (
            temporal_patch_size > 1
            and num_patches.numel() > 0
            and ((num_patches == 0).any() or (num_patches == temporal_patch_size).all())
        ):
            if not torch.is_tensor(pixel_values):
                flattened_pixels: list[torch.Tensor] = []

                def append_pixels(item: object) -> None:
                    if torch.is_tensor(item):
                        if item.numel() == 0:
                            return
                        flattened_pixels.append(
                            item if item.ndim == 4 else item.unsqueeze(0)
                        )
                        return
                    if isinstance(item, (list, tuple)):
                        for child in item:
                            append_pixels(child)
                        return
                    raise TypeError(
                        "Expected tensor or nested tensor list for "
                        f"pixel_values_flat, got {type(item)}"
                    )

                append_pixels(pixel_values)
                if not flattened_pixels:
                    device = num_patches.device
                    pixel_values = torch.empty(
                        0, 3, 0, 0, device=device, dtype=torch.bfloat16
                    )
                else:
                    pixel_values = torch.cat(flattened_pixels, dim=0)

            total_requested_frames = int(num_patches.sum().item())
            available_frames = int(pixel_values.shape[0])
            if available_frames < total_requested_frames:
                raise ValueError(
                    "Static video-as-images received too few frame tensors for "
                    "temporal tubelets: "
                    f"available={available_frames}, requested={total_requested_frames}, "
                    f"num_patches={num_patches.tolist()}. "
                    "Check that pixel_values_flat uses flat_from_sizes with "
                    "image_num_patches."
                )

            if os.environ.get("NRL_DEBUG", "0") == "1":
                print(
                    "[VLLM_VIDEO_AS_IMAGES_EMBED] "
                    f"items={num_patches.numel()} "
                    f"available_frames={available_frames} "
                    f"requested_frames={total_requested_frames} "
                    f"num_patches_head={num_patches[:8].tolist()}",
                    flush=True,
                )

            nonzero_items = int((num_patches > 0).sum().item())
            video_embeds = self.extract_feature(
                pixel_values[:total_requested_frames],
                num_frames=total_requested_frames,
            ).view(-1, hidden_size)
            if nonzero_items == 0:
                feature_size = 0
            elif video_embeds.shape[0] % nonzero_items != 0:
                raise ValueError(
                    "Static video-as-images produced an embedding count that "
                    "cannot be split across temporal tubelets: "
                    f"embeds={video_embeds.shape[0]}, items={nonzero_items}, "
                    f"num_patches={num_patches.tolist()}"
                )
            else:
                feature_size = video_embeds.shape[0] // nonzero_items

            results: list[torch.Tensor] = []
            embed_offset = 0
            for patch_count_tensor in num_patches:
                patch_count = int(patch_count_tensor.item())
                if patch_count == 0:
                    results.append(
                        torch.empty(
                            0,
                            hidden_size,
                            device=pixel_values.device,
                            dtype=torch.bfloat16,
                        )
                    )
                    continue

                item_embeds = video_embeds[embed_offset : embed_offset + feature_size]
                embed_offset += feature_size
                results.append(item_embeds)

            return tuple(results)

        image_embeds = self.extract_feature(pixel_values)

        # Only one image in the current batch
        if len(num_patches) == 1:
            return (image_embeds.view(-1, hidden_size),)

        # NOTE: Image embeddings are split into separate tensors for each image
        # by the size of each embedding.
        feature_size = image_embeds.shape[1]
        image_embeds = image_embeds.view(-1, hidden_size)
        image_feature_sizes = [
            num_patches * feature_size for num_patches in num_patches
        ]
        return image_embeds.split(image_feature_sizes)
"""
        if old_process_image_input in text:
            text = text.replace(old_process_image_input, new_process_image_input, 1)
        elif old_process_image_input_static_video in text:
            text = text.replace(
                old_process_image_input_static_video, new_process_image_input, 1
            )
        elif "Static video-as-images received too few frame tensors" not in text:
            raise RuntimeError(
                f"Could not patch {model_path}: missing _process_image_input block"
            )

        old_image_dispatch = """                elif self.dynamic_resolution:
                    assert image_input["type"] == "pixel_values_dynamic"
                    image_embeddings = self._process_image_input_dynamic(image_input)
                else:
                    image_embeddings = self._process_image_input(image_input)
"""
        new_image_dispatch = """                elif image_input["type"] == "pixel_values_dynamic":
                    image_embeddings = self._process_image_input_dynamic(image_input)
                else:
                    assert image_input["type"] == "pixel_values"
                    image_embeddings = self._process_image_input(image_input)
"""
        if old_image_dispatch in text:
            text = text.replace(old_image_dispatch, new_image_dispatch, 1)
        elif 'elif image_input["type"] == "pixel_values_dynamic"' not in text:
            raise RuntimeError(
                f"Could not patch {model_path}: missing image dispatch block"
            )

        old_video_repl_return = """            frame_duration_ms = int(1000 / metadata["fps"])
            return hf_processor.get_video_repl(
                tokens_per_frame=tokens_per_frame,
                frames_indices=metadata["frames_indices"],
                frame_duration_ms=frame_duration_ms,
                tokenizer=hf_processor.tokenizer,
                img_start_token_ids=hf_processor._img_start_token_ids,
                img_end_token_ids=hf_processor._img_end_token_ids,
                img_context_token_ids=hf_processor._img_context_token_ids,
                video_temporal_patch_size=T,
            )
"""
        new_video_repl_return = """            frame_duration_ms = int(1000 / metadata["fps"])
            use_frame_separators = (
                os.environ.get("NRL_VLLM_VIDEO_FRAME_SEPARATORS", "0")
                .strip()
                .lower()
                in ("1", "true", "yes", "on")
            )
            if not use_frame_separators:
                img_context_token_ids = list(hf_processor._img_context_token_ids)
                all_token_ids: list[int] = []
                for num_tokens in tokens_per_frame:
                    all_token_ids.extend(hf_processor._img_start_token_ids)
                    all_token_ids.extend(img_context_token_ids * num_tokens)
                    all_token_ids.extend(hf_processor._img_end_token_ids)

                def is_embed(tokenizer, full):
                    token_ids = (
                        full
                        if isinstance(full, list)
                        else tokenizer.encode(full, add_special_tokens=False)
                    )
                    return torch.isin(
                        torch.tensor(token_ids),
                        torch.tensor(img_context_token_ids),
                    )

                embed_token_count = sum(tokens_per_frame) * len(img_context_token_ids)
                if os.environ.get("NRL_DEBUG", "0") == "1":
                    print(
                        "[VLLM_NATIVE_VIDEO_REPL_MODEL_PLAIN] "
                        "frame_separators=0 "
                        f"full_tokens={len(all_token_ids)} "
                        f"embed_tokens={embed_token_count} "
                        f"text_tokens={len(all_token_ids) - embed_token_count} "
                        f"tubelets={len(tokens_per_frame)} T={T}",
                        flush=True,
                    )

                return PromptUpdateDetails(
                    full=all_token_ids,
                    is_embed=is_embed,
                )

            video_repl = hf_processor.get_video_repl(
                tokens_per_frame=tokens_per_frame,
                frames_indices=metadata["frames_indices"],
                frame_duration_ms=frame_duration_ms,
                tokenizer=hf_processor.tokenizer,
                img_start_token_ids=hf_processor._img_start_token_ids,
                img_end_token_ids=hf_processor._img_end_token_ids,
                img_context_token_ids=hf_processor._img_context_token_ids,
                video_temporal_patch_size=T,
            )
            if video_repl.is_embed is None:
                img_context_token_ids = list(hf_processor._img_context_token_ids)

                def is_embed(tokenizer, full):
                    token_ids = (
                        full
                        if isinstance(full, list)
                        else tokenizer.encode(full, add_special_tokens=False)
                    )
                    return torch.isin(
                        torch.tensor(token_ids),
                        torch.tensor(img_context_token_ids),
                    )

                video_repl = PromptUpdateDetails(
                    full=video_repl.full,
                    is_embed=is_embed,
                )
                if os.environ.get("NRL_DEBUG", "0") == "1":
                    print(
                        "[VLLM_NATIVE_VIDEO_REPL_GUARD] "
                        "added context-token embed mask "
                        f"full_tokens={len(video_repl.full)} "
                        f"embed_token_ids={img_context_token_ids}",
                        flush=True,
                    )

            return video_repl
"""
        if old_video_repl_return in text:
            text = text.replace(old_video_repl_return, new_video_repl_return, 1)
        elif "VLLM_NATIVE_VIDEO_REPL_GUARD" not in text:
            raise RuntimeError(
                f"Could not patch {model_path}: missing video replacement block"
            )
        if (
            "VLLM_NATIVE_VIDEO_REPL_GUARD" in text
            and "VLLM_NATIVE_VIDEO_REPL_MODEL_PLAIN" not in text
        ):
            old_video_repl_call = """            frame_duration_ms = int(1000 / metadata["fps"])
            video_repl = hf_processor.get_video_repl(
"""
            new_video_repl_call = """            frame_duration_ms = int(1000 / metadata["fps"])
            use_frame_separators = (
                os.environ.get("NRL_VLLM_VIDEO_FRAME_SEPARATORS", "0")
                .strip()
                .lower()
                in ("1", "true", "yes", "on")
            )
            if not use_frame_separators:
                img_context_token_ids = list(hf_processor._img_context_token_ids)
                all_token_ids: list[int] = []
                for num_tokens in tokens_per_frame:
                    all_token_ids.extend(hf_processor._img_start_token_ids)
                    all_token_ids.extend(img_context_token_ids * num_tokens)
                    all_token_ids.extend(hf_processor._img_end_token_ids)

                def is_embed(tokenizer, full):
                    token_ids = (
                        full
                        if isinstance(full, list)
                        else tokenizer.encode(full, add_special_tokens=False)
                    )
                    return torch.isin(
                        torch.tensor(token_ids),
                        torch.tensor(img_context_token_ids),
                    )

                embed_token_count = sum(tokens_per_frame) * len(img_context_token_ids)
                if os.environ.get("NRL_DEBUG", "0") == "1":
                    print(
                        "[VLLM_NATIVE_VIDEO_REPL_MODEL_PLAIN] "
                        "frame_separators=0 "
                        f"full_tokens={len(all_token_ids)} "
                        f"embed_tokens={embed_token_count} "
                        f"text_tokens={len(all_token_ids) - embed_token_count} "
                        f"tubelets={len(tokens_per_frame)} T={T}",
                        flush=True,
                    )

                return PromptUpdateDetails(
                    full=all_token_ids,
                    is_embed=is_embed,
                )

            video_repl = hf_processor.get_video_repl(
"""
            if old_video_repl_call not in text:
                raise RuntimeError(
                    f"Could not patch {model_path}: missing video replacement call"
                )
            text = text.replace(old_video_repl_call, new_video_repl_call, 1)
        print(
            "[NRL_PATCH_CONTAINER_VLLM] video replacement guard "
            f"processor_plain={'VLLM_NATIVE_VIDEO_REPL_PROCESSOR_PLAIN' in text} "
            f"present={'VLLM_NATIVE_VIDEO_REPL_GUARD' in text} "
            f"model_plain={'VLLM_NATIVE_VIDEO_REPL_MODEL_PLAIN' in text}",
            flush=True,
        )

        old_process_video_input = """    def _process_video_input(
        self, video_input: NanoNemotronVLVideoPixelInputs
    ) -> tuple[torch.Tensor, ...]:
        \"\"\"Process video input and create final embeddings with video content
        and indicator tokens.\"\"\"
        T = self.video_temporal_patch_size

        if T > 1:
            video_embeddings = self._extract_video_embeddings_temporal(video_input)
        else:
            video_embeddings = self._process_image_input(video_input)

        final_video_embeddings: tuple[torch.Tensor, ...] = ()

        downsample_ratio = self.config.downsample_ratio
        patch_size = self.config.patch_size
        pixel_values = video_input["pixel_values_flat"]
        frame_h, frame_w = pixel_values.shape[-2], pixel_values.shape[-1]
        rows = int(frame_h * downsample_ratio // patch_size)
        cols = int(frame_w * downsample_ratio // patch_size)
        video_pruning_rate = self.video_pruning_rate
        video_num_frames = video_input["num_patches"].tolist()
        video_frames_indices = video_input["frames_indices"].split(video_num_frames)
        # Calculate video feature dimensions (number of frames and
        # their feature size (AKA tokens per frame))
        # TODO: Maybe this can be optimized to avoid the loop?
        for i, single_video_embeddings in enumerate(video_embeddings):
            num_frames = video_num_frames[i]
            frames_indices = video_frames_indices[i].tolist()
            frame_duration_ms = video_input["frame_duration_ms"][i].item()
            num_tubelets = math.ceil(num_frames / T) if T > 1 else num_frames
            assert single_video_embeddings.shape[0] % num_tubelets == 0

            if video_pruning_rate is not None and video_pruning_rate > 0.0:
                # Start of EVS-specific code
                retention_mask = compute_retention_mask(
                    single_video_embeddings,
                    video_size_thw=(num_tubelets, rows, cols),
                    spatial_merge_size=1,
                    q=video_pruning_rate,
                )

                # apply retention mask
                single_video_embeddings = single_video_embeddings[retention_mask]

                # calculate the actual number of retained tokens per frame
                retention_mask_thw = retention_mask.reshape(num_tubelets, rows, cols)
                num_tokens_per_frame = (
                    retention_mask_thw.sum(dim=(1, 2)).long().tolist()
                )
                # End of EVS-specific code
            else:
                feature_size = single_video_embeddings.shape[0] // num_tubelets
                num_tokens_per_frame = [feature_size] * num_tubelets

            final_video_embeddings += (
                self._create_final_video_embeddings(
                    single_video_embeddings,
                    num_tokens_per_frame,
                    frames_indices,
                    frame_duration_ms,
                    video_temporal_patch_size=T,
                ),
            )

        return final_video_embeddings
"""
        new_process_video_input = """    def _process_video_input(
        self, video_input: NanoNemotronVLVideoPixelInputs
    ) -> tuple[torch.Tensor, ...]:
        \"\"\"Process video input into embeddings for video context tokens.\"\"\"
        T = self.video_temporal_patch_size

        if T > 1:
            video_embeddings = self._extract_video_embeddings_temporal(video_input)
        else:
            video_embeddings = self._process_image_input(video_input)

        video_pruning_rate = self.video_pruning_rate
        if video_pruning_rate is None or video_pruning_rate <= 0.0:
            return video_embeddings

        downsample_ratio = self.config.downsample_ratio
        patch_size = self.config.patch_size
        pixel_values = video_input["pixel_values_flat"]
        frame_h, frame_w = pixel_values.shape[-2], pixel_values.shape[-1]
        rows = int(frame_h * downsample_ratio // patch_size)
        cols = int(frame_w * downsample_ratio // patch_size)
        video_num_frames = video_input["num_patches"].tolist()

        pruned_video_embeddings: list[torch.Tensor] = []
        for i, single_video_embeddings in enumerate(video_embeddings):
            num_frames = video_num_frames[i]
            num_tubelets = math.ceil(num_frames / T) if T > 1 else num_frames
            assert single_video_embeddings.shape[0] % num_tubelets == 0

            retention_mask = compute_retention_mask(
                single_video_embeddings,
                video_size_thw=(num_tubelets, rows, cols),
                spatial_merge_size=1,
                q=video_pruning_rate,
            )
            pruned_video_embeddings.append(single_video_embeddings[retention_mask])

        return tuple(pruned_video_embeddings)
"""
        if old_process_video_input in text:
            text = text.replace(old_process_video_input, new_process_video_input, 1)
        elif "Process video input into embeddings for video context tokens." not in text:
            raise RuntimeError(
                f"Could not patch {model_path}: missing _process_video_input block"
            )

        model_path.write_text(text)
        print(f"[NRL_PATCH_CONTAINER_VLLM] patched {model_path} in-place", flush=True)
PY
EOS
)
  SETUP_COMMAND="${SETUP_COMMAND:+${SETUP_COMMAND}
}${VLLM_PATCH_SETUP}"
fi
export SETUP_COMMAND

PYTHONPATH_ROOTS="${NEMORL}:${NEMORL}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src:${NEMORL}/3rdparty/Megatron-LM-workspace/Megatron-LM"
if [[ "${USE_REPO_VLLM:-0}" == "1" ]]; then
  PYTHONPATH_ROOTS="${NEMORL}/3rdparty/vllm:${PYTHONPATH_ROOTS}"
fi

export COMMAND="\
mkdir -p '${HF_HOME}' '${HF_MODULES_CACHE}' '${NRL_MEGATRON_CHECKPOINT_DIR}' '${TRITON_CACHE_DIR}' '${TMPDIR}' '${RESULTS_DIR}' && \
export PYTHONPATH=${PYTHONPATH_ROOTS}\${PYTHONPATH:+:\$PYTHONPATH} && \
uv run --no-sync examples/run_vlm_grpo.py --config '${CONFIG_PATH}' \
cluster.num_nodes=${NUM_NODES} \
cluster.gpus_per_node=${GPUS_PER_NODE} \
grpo.seed=${SEED} \
grpo.num_prompts_per_step=256 \
grpo.seq_logprob_error_threshold=50 \
grpo.zero_variance_prompt_filtering=true \
grpo.val_at_end=false \
policy.train_global_batch_size=${GLOBAL_TRAIN_BATCH_SIZE} \
policy.train_micro_batch_size=${MICRO_BS} \
policy.logprob_batch_size=${LOGPROB_BS} \
policy.offload_optimizer_for_logprob=${OFFLOAD_OPTIMIZER_FOR_LOGPROB} \
policy.sequence_packing.enabled=${SEQUENCE_PACKING_ENABLED} \
	policy.max_total_sequence_length=${MAX_TOTAL_SEQUENCE_LENGTH} \
	policy.model_name='${MODEL_NAME}' \
policy.megatron_cfg.freeze_vision_model=true \
policy.megatron_cfg.freeze_vision_projection=true \
policy.megatron_cfg.freeze_sound_encoder=true \
	policy.megatron_cfg.freeze_sound_projection=true \
	policy.megatron_cfg.scheduler.lr_warmup_iters=3 \
		policy.generation.vllm_cfg.gpu_memory_utilization=${VLLM_GPU_MEMORY_UTILIZATION} \
		+policy.generation.vllm_cfg.load_format=${VLLM_LOAD_FORMAT} \
		+policy.generation.vllm_cfg.enable_prefix_caching=${VLLM_ENABLE_PREFIX_CACHING} \
		data.train.train_data_path='${TRAIN_DATA_PATH}' \
	data.default.num_frames=${NUM_FRAMES} \
	data.default.max_images_per_prompt=${NUM_FRAMES} \
	policy.generation.vllm_kwargs.limit_mm_per_prompt.image=${NUM_FRAMES} \
	checkpointing.checkpoint_dir='${RESULTS_DIR}/checkpoints' \
checkpointing.save_period=5 \
logger.log_dir='${RESULTS_DIR}/nemorl_logs' \
logger.wandb_enabled=true \
logger.wandb.project='${WANDB_PROJECT}' \
logger.wandb.name='${JOB_NAME}' \
+policy.megatron_cfg.checkpoint.async_save=false \
+policy.megatron_cfg.freeze_embedding=false\
${EXTRA_OVERRIDES}"

cd "${NEMORL}"

SBATCH_ARRAY_ARGS=()
if [[ "${JOB_CYCLES}" -gt 0 ]]; then
  SBATCH_ARRAY_ARGS+=(--array="0-${JOB_CYCLES}%1" --dependency=singleton)
fi
SBATCH_MEMORY_ARGS=()
if [[ -n "${SBATCH_MEM:-}" ]]; then
  SBATCH_MEMORY_ARGS+=(--mem="${SBATCH_MEM}")
fi
if [[ -n "${SBATCH_MEM_PER_GPU:-}" ]]; then
  SBATCH_MEMORY_ARGS+=(--mem-per-gpu="${SBATCH_MEM_PER_GPU}")
fi

echo "JOB_NAME=${JOB_NAME}"
echo "NUM_NODES=${NUM_NODES}"
echo "GLOBAL_TRAIN_BATCH_SIZE=${GLOBAL_TRAIN_BATCH_SIZE}"
echo "SBATCH_ACCOUNT=${SBATCH_ACCOUNT}"
echo "SBATCH_PARTITION=${SBATCH_PARTITION}"
echo "SBATCH_MEM=${SBATCH_MEM:-}"
echo "SBATCH_MEM_PER_GPU=${SBATCH_MEM_PER_GPU:-}"
echo "CONTAINER=${CONTAINER}"
echo "RESULTS_DIR=${RESULTS_DIR}"

sbatch \
  --nodes="${NUM_NODES}" \
  --account="${SBATCH_ACCOUNT}" \
  --job-name="${JOB_NAME}" \
  --partition="${SBATCH_PARTITION}" \
  --time="${SBATCH_TIME}" \
  --gres="gpu:${GPUS_PER_NODE}" \
  --output="${LOGS_DIR}/%x_%A_%a.log" \
  "${SBATCH_MEMORY_ARGS[@]}" \
  "${SBATCH_ARRAY_ARGS[@]}" \
  ray.sub
