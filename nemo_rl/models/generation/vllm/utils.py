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
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import GenerationDatumSpec


class AudioLoadError(RuntimeError):
    """Raised when audio cannot be loaded from a file path."""

    def __init__(self, path: str, reason: str = "unknown"):
        self.path = path
        super().__init__(f"Failed to load audio from {path}: {reason}")


def _load_audio_pyav(
    audio_path: str,
    target_sr: int = 16000,
    max_duration: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Load audio from a file using PyAV, matching SFT's AVDecoder.get_audio()."""
    import av
    import librosa

    _debug = os.environ.get("NRL_DEBUG", "0") == "1"

    try:
        container = av.open(audio_path)
    except Exception as e:
        if _debug:
            print(f"[AUDIO_PYAV] Cannot open {audio_path}: {e}", flush=True)
        return None

    if not container.streams.audio:
        container.close()
        if _debug:
            print(f"[AUDIO_PYAV] No audio stream in {audio_path}", flush=True)
        return None

    stream = container.streams.audio[0]
    stream.codec_context.thread_type = "NONE"
    native_sr = stream.rate
    n_channels = stream.channels if hasattr(stream, "channels") else None

    chunks: list[np.ndarray] = []
    try:
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray().astype(np.float32)
            if arr.ndim > 1 and arr.shape[0] > 1:
                arr = arr.mean(axis=0)
            elif arr.ndim > 1:
                arr = arr.reshape(-1)
            else:
                arr = arr.ravel()
            if arr.size > 0:
                chunks.append(arr)
    except Exception as e:
        if _debug:
            print(f"[AUDIO_PYAV] Decode error for {audio_path}: {e}", flush=True)
    finally:
        container.close()

    if not chunks:
        if _debug:
            print(
                f"[AUDIO_PYAV] No audio frames decoded from {audio_path}", flush=True
            )
        return None

    waveform = np.concatenate(chunks).astype(np.float32)

    if waveform.size == 0:
        if _debug:
            print(f"[AUDIO_PYAV] Zero-length audio from {audio_path}", flush=True)
        return None

    raw_len = len(waveform)

    if native_sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=native_sr, target_sr=target_sr)

    if max_duration is not None:
        max_samples = int(max_duration * target_sr)
        waveform = waveform[:max_samples]

    if _debug:
        print(
            f"[AUDIO_PYAV] _load_audio_pyav: path={audio_path.split('/')[-1]} "
            f"native_sr={native_sr} target_sr={target_sr} "
            f"channels={n_channels} raw_samples={raw_len} "
            f"resampled={'yes' if native_sr != target_sr else 'no'} "
            f"final_len={len(waveform)} "
            f"waveform_stats(mean={waveform.mean():.6f},std={waveform.std():.6f},"
            f"min={waveform.min():.6f},max={waveform.max():.6f}) "
            f"first5={waveform[:5].tolist()}",
            flush=True,
        )

    return waveform


def load_audio_waveform(
    audio_path: str,
    target_sr: int = 16000,
    max_duration: Optional[float] = None,
    raise_on_failure: bool = False,
    force_pyav: bool = False,
) -> Optional[np.ndarray]:
    """Load audio from a file path and return a 1-D float32 waveform."""
    import librosa

    _debug = os.environ.get("NRL_DEBUG", "0") == "1"
    native_sr = None
    waveform = None

    video_extensions = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".ts"}
    ext = os.path.splitext(audio_path)[1].lower()

    if not force_pyav and ext not in video_extensions:
        try:
            import soundfile as sf

            waveform, native_sr = sf.read(audio_path, dtype="float32", always_2d=True)
            waveform = waveform.mean(axis=1)
        except Exception:
            waveform = None

    if waveform is None:
        result = _load_audio_pyav(
            audio_path, target_sr=target_sr, max_duration=max_duration
        )
        if result is not None:
            return result
        if _debug:
            print(f"[DEBUG] Audio load failed for {audio_path}. Skipping audio.", flush=True)
        if raise_on_failure:
            raise AudioLoadError(
                audio_path, reason="both soundfile and PyAV failed"
            )
        return None

    if waveform.size == 0:
        if _debug:
            print(
                f"[DEBUG] Zero-length audio from {audio_path}. Skipping audio.",
                flush=True,
            )
        if raise_on_failure:
            raise AudioLoadError(audio_path, reason="zero-length waveform")
        return None

    waveform = waveform.astype(np.float32)

    if native_sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=native_sr, target_sr=target_sr)

    if max_duration is not None:
        max_samples = int(max_duration * target_sr)
        waveform = waveform[:max_samples]
    return waveform


def format_prompt_for_vllm_generation(
    data: BatchedDataDict[GenerationDatumSpec], sample_idx: Optional[int] = None
) -> list[dict[str, Any]]:
    """Format a list of prompts for vllm generation (which requires a specific format for its own `generate` method).

    See https://docs.vllm.ai/en/v0.9.1/features/multimodal_inputs.html for prompt format for multimodal inputs.
    """
    # Prepare prompts for vLLM (removing padding)
    prompts = []

    input_ids = data["input_ids"]
    batch_size = input_ids.shape[0]
    input_lengths = data["input_lengths"]

    # if sample_idx is None, return list of all prompts for the entire batch
    # else, return the prompt for the single sample specified by sample_idx
    return_all = sample_idx is None
    if sample_idx is None:
        start_idx = 0
        end_idx = batch_size
    else:
        start_idx = sample_idx
        end_idx = sample_idx + 1

    def _get_regular_prompt(index: int):
        valid_length = input_lengths[index].item()
        valid_ids = (
            input_ids[index, :valid_length]
            if valid_length > 0
            else input_ids[index, :0]
        )
        token_ids = valid_ids.tolist()
        return {"prompt_token_ids": token_ids}

    # Check if this is VLM generation by looking for message_log with images
    # Support for videos/audio/etc. can be added here
    # if 'message_log' in data and any('images' in msg for msg in data['message_log']):
    if "vllm_content" in data:
        # VLM generation using content and multi_modal_data
        for i in range(start_idx, end_idx):
            msg = data["vllm_content"][i]
            # if msg is None, this conversation had no multimodal content, fallback to regular prompt
            if msg is None:
                prompts.append(_get_regular_prompt(i))
                continue
            # init prompt dict
            prompt_dict = {"prompt": msg}
            # add additional data if present
            images = data.get("vllm_images", None)
            if images is None or len(images[i]) == 0:
                prompts.append(_get_regular_prompt(i))
                continue
            else:
                prompt_dict["multi_modal_data"] = {
                    "image": images[i][0] if len(images[i]) == 1 else images[i]
                }
            prompts.append(prompt_dict)
    else:
        # Regular LLM generation using token_ids
        for i in range(start_idx, end_idx):
            # Use input_lengths to get only valid tokens (not padding)
            prompts.append(_get_regular_prompt(i))

    return prompts if return_all else prompts[0]


def aggregate_spec_decode_counters(
    worker_metrics: list[dict[str, float | list[float]]],
) -> dict[str | tuple[str, int], float]:
    """Aggregate speculative decoding counters from multiple workers.

    Combines spec decode metrics collected from DP leader workers into
    a single aggregated counter dictionary.

    Args:
        worker_metrics: List of metric dictionaries from each worker.
            Each dict maps metric names to float values or lists of floats
            (for per-position metrics).

    Returns:
        Dictionary mapping metric names to their aggregated float values.
        Per-position metrics use (name, position) tuples as keys.

    Example:
        >>> metrics_from_workers = policy_generation.get_metrics()
        >>> counters = aggregate_spec_decode_counters(metrics_from_workers)
        >>> print(counters.get("vllm:spec_decode_num_drafts", 0))
        1234.0
    """
    counters: dict[str | tuple[str, int], float] = defaultdict(float)

    for report in worker_metrics:
        for metric_name, value in report.items():
            if "spec_decode" in metric_name:
                if isinstance(value, list):
                    # Per-position metrics (e.g., acceptance counts at each draft position)
                    for position, pos_value in enumerate(value, 1):
                        counters[metric_name, position] += pos_value
                else:
                    counters[metric_name] += value

    return dict(counters)


def compute_spec_decode_metrics(
    start_counters: dict[str | tuple[str, int], float],
    end_counters: dict[str | tuple[str, int], float],
) -> dict[str, float]:
    """Compute delta and derived metrics for speculative decoding.

    Calculates the difference between two counter snapshots and derives
    acceptance rate and acceptance length metrics for logging.

    Args:
        start_counters: Counter snapshot taken before generation.
        end_counters: Counter snapshot taken after generation.

    Returns:
        Dictionary of metrics suitable for logging to wandb/tensorboard.
        Keys are prefixed with "vllm/" for namespace consistency.
        Includes:
            - vllm/spec_num_drafts: Total number of draft batches
            - vllm/spec_num_draft_tokens: Total draft tokens generated
            - vllm/spec_num_accepted_tokens: Total tokens accepted
            - vllm/spec_acceptance_length: Average accepted tokens per draft + 1
            - vllm/spec_acceptance_rate: Ratio of accepted to draft tokens
            - vllm/{metric}-{position}: Per-position acceptance counts
            - vllm/spec_acceptance_rate-pos-{position}: Per-position acceptance rates
    """
    keys = set(start_counters) | set(end_counters)
    delta = {k: end_counters.get(k, 0.0) - start_counters.get(k, 0.0) for k in keys}

    num_drafts = delta.get("vllm:spec_decode_num_drafts", 0.0)
    num_draft_tokens = delta.get("vllm:spec_decode_num_draft_tokens", 0.0)
    num_accepted_tokens = delta.get("vllm:spec_decode_num_accepted_tokens", 0.0)

    # acceptance_length = 1 + (accepted / drafts) represents average tokens
    # generated per draft batch (1 target model token + accepted draft tokens)
    acceptance_length = (
        1.0 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1.0
    )
    acceptance_rate = (
        num_accepted_tokens / num_draft_tokens if num_draft_tokens > 0 else 0.0
    )

    spec_metrics: dict[str, float] = {
        "vllm/spec_num_drafts": num_drafts,
        "vllm/spec_num_draft_tokens": num_draft_tokens,
        "vllm/spec_num_accepted_tokens": num_accepted_tokens,
        "vllm/spec_acceptance_length": acceptance_length,
        "vllm/spec_acceptance_rate": acceptance_rate,
    }

    # Add per-position metrics for detailed analysis
    for key, value in delta.items():
        if isinstance(key, tuple):
            metric_name, position = key
            spec_metrics[f"vllm/{metric_name}-{position}"] = value
            if num_drafts > 0:
                spec_metrics[f"vllm/spec_acceptance_rate-pos-{position}"] = (
                    value / num_drafts
                )

    return spec_metrics
