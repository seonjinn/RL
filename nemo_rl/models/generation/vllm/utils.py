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

import hashlib
import json
import os
import threading
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

from nemo_rl.data.multimodal_utils import PackedTensor, resolve_to_image
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import GenerationDatumSpec
from nemo_rl.utils.vlm_debug import debug_enabled, write_stage


_PRECOMPUTED_IMG_SIZES_ENV = "NEMO_RL_VLLM_PRECOMPUTED_IMG_SIZES"
_DUMP_VLLM_REQUESTS_ENV = "NEMO_RL_DUMP_VLLM_REQUESTS"
_DUMP_VLLM_REQUESTS_LIMIT_ENV = "NEMO_RL_DUMP_VLLM_REQUESTS_LIMIT"
_DUMP_VLLM_REQUESTS_DEFAULT_LIMIT = 16
_DUMP_LOCK = threading.Lock()
_DUMP_REQUEST_COUNT = 0


class AudioLoadError(RuntimeError):
    """Raised when audio cannot be loaded from a file path."""

    def __init__(self, path: str, reason: str = "unknown"):
        self.path = path
        super().__init__(f"Failed to load audio from {path}: {reason}")


def extract_sampled_token_logprob(
    logprob_dict: Any,
    sampled_token_id: int,
) -> Optional[float]:
    """Return the vLLM logprob for the sampled token, not the first top-logprob."""
    if not logprob_dict:
        return None

    token_logprob = None
    try:
        token_logprob = logprob_dict.get(sampled_token_id)
    except AttributeError:
        token_logprob = None

    if token_logprob is None:
        try:
            items = logprob_dict.items()
        except AttributeError:
            items = ()
        for token_id, candidate_logprob in items:
            try:
                token_id_matches = int(token_id) == int(sampled_token_id)
            except (TypeError, ValueError):
                token_id_matches = token_id == sampled_token_id
            if token_id_matches:
                token_logprob = candidate_logprob
                break

    if token_logprob is None:
        return None

    value = getattr(token_logprob, "logprob", token_logprob)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_VIDEO_SAMPLING_STYLE_ENV = "NRL_VIDEO_SAMPLING_STYLE"
_VIDEO_SAMPLING_STYLE_CURRENT = "current_fixed"
_VIDEO_SAMPLING_STYLE_SFT_V2_DURATION = "sft_v2_duration"
_VIDEO_SAMPLING_STYLE_DEFAULT = _VIDEO_SAMPLING_STYLE_SFT_V2_DURATION
_SUPPORTED_VIDEO_SAMPLING_STYLES = {
    _VIDEO_SAMPLING_STYLE_CURRENT,
    _VIDEO_SAMPLING_STYLE_SFT_V2_DURATION,
}


def _get_video_sampling_style() -> str:
    style = os.environ.get(_VIDEO_SAMPLING_STYLE_ENV, _VIDEO_SAMPLING_STYLE_DEFAULT)
    style = style.strip().lower()
    if style not in _SUPPORTED_VIDEO_SAMPLING_STYLES:
        supported = ", ".join(sorted(_SUPPORTED_VIDEO_SAMPLING_STYLES))
        raise ValueError(
            f"Unsupported {_VIDEO_SAMPLING_STYLE_ENV}={style!r}; supported: {supported}"
        )
    return style


def _get_positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _round_video_frame_count(
    num_frames: int,
    *,
    total_frames_in_file: int,
    max_frames: int,
    temporal_patch_size: int,
) -> int:
    num_frames = min(num_frames, total_frames_in_file)
    if temporal_patch_size > 1 and num_frames % temporal_patch_size != 0:
        rounded_down = (num_frames // temporal_patch_size) * temporal_patch_size
        rounded_up = rounded_down + temporal_patch_size
        if rounded_up <= total_frames_in_file and rounded_up <= max_frames:
            num_frames = rounded_up
        else:
            num_frames = max(temporal_patch_size, rounded_down)
    return num_frames


def _select_video_frame_count(
    *,
    total_duration: float,
    requested_num_frames: int,
    total_frames_in_file: int,
    temporal_patch_size: int,
) -> int:
    requested_num_frames = max(1, int(requested_num_frames))
    sampling_style = _get_video_sampling_style()
    if sampling_style == _VIDEO_SAMPLING_STYLE_SFT_V2_DURATION:
        min_frames = _get_positive_int_env("NRL_VIDEO_SFT_MIN_FRAMES", 8)
        sft_max_frames = _get_positive_int_env("NRL_VIDEO_SFT_MAX_FRAMES", 256)
        default_fps = _get_positive_int_env("NRL_VIDEO_SFT_DEFAULT_FPS", 2)
        if total_frames_in_file < min_frames:
            num_frames = total_frames_in_file
        else:
            default_frames = int(default_fps * total_duration)
            num_frames = min(max(default_frames, min_frames), sft_max_frames)
        num_frames = min(num_frames, requested_num_frames)
    else:
        num_frames = requested_num_frames

    return _round_video_frame_count(
        num_frames,
        total_frames_in_file=total_frames_in_file,
        max_frames=requested_num_frames,
        temporal_patch_size=temporal_patch_size,
    )


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


def _compute_video_timestamps(
    total_duration: float,
    num_frames: int,
    total_frames_in_file: int,
    original_num_frames: int,
    temporal_patch_size: int,
) -> tuple[int, list[float]]:
    num_frames = _select_video_frame_count(
        total_duration=total_duration,
        requested_num_frames=original_num_frames,
        total_frames_in_file=total_frames_in_file,
        temporal_patch_size=temporal_patch_size,
    )

    if num_frames <= 1:
        return 1, [total_duration / 2.0]

    effective_span = max(total_duration - 1, 0)
    segment_size = effective_span / num_frames
    return num_frames, [segment_size * (i + 0.5) for i in range(num_frames)]


def _build_video_metadata(
    *,
    fps: float,
    total_frames: int,
    sampled_indices: list[int],
    backend: str,
) -> dict[str, Any]:
    return {
        "fps": fps,
        "duration": total_frames / fps,
        "total_num_frames": total_frames,
        "frames_indices": sampled_indices,
        "video_backend": backend,
        "video_sampling_style": _get_video_sampling_style(),
        "do_sample_frames": False,
    }


def _load_video_frames_pyav_with_metadata(
    video_path: str,
    num_frames: int = 8,
    temporal_patch_size: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    import av

    try:
        container = av.open(video_path)
    except Exception as exc:
        raise ValueError(f"Cannot open video: {video_path}") from exc

    if not container.streams.video:
        container.close()
        raise ValueError(f"No video stream in {video_path}")

    stream = container.streams.video[0]
    stream.codec_context.thread_type = "NONE"
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    if fps <= 0:
        container.close()
        raise ValueError(f"Video has invalid fps ({fps}): {video_path}")

    total_frames = stream.frames
    if total_frames <= 0:
        if stream.duration and stream.time_base:
            duration_est = float(stream.duration * stream.time_base)
        elif container.duration:
            duration_est = container.duration / av.time_base
        else:
            duration_est = 0.0
        total_frames = max(1, int(duration_est * fps))
    total_duration = total_frames / fps

    original_num_frames = num_frames
    num_frames, timestamps_s = _compute_video_timestamps(
        total_duration,
        num_frames,
        total_frames,
        original_num_frames,
        temporal_patch_size,
    )
    time_base = float(stream.time_base) if stream.time_base else 1.0 / fps
    target_pts_list = [int(ts / time_base) for ts in timestamps_s]
    sampled_indices = [
        max(0, min(int(ts * fps), total_frames - 1)) for ts in timestamps_s
    ]

    frames: list[np.ndarray] = []
    try:
        if target_pts_list:
            container.seek(max(0, target_pts_list[0]), stream=stream, any_frame=False)
        target_idx = 0
        best_frame = None
        frame_counter = 0
        for frame in container.decode(video=0):
            if target_idx >= len(target_pts_list):
                break
            best_frame = frame
            frame_counter += 1
            if frame.pts is None:
                while (
                    target_idx < len(target_pts_list)
                    and frame_counter >= target_idx + 1
                ):
                    frames.append(frame.reformat(format="rgb24").to_ndarray())
                    target_idx += 1
                continue
            frame_end = frame.pts + (frame.duration if frame.duration else 1)
            while target_idx < len(target_pts_list):
                if target_pts_list[target_idx] < frame_end:
                    frames.append(frame.reformat(format="rgb24").to_ndarray())
                    target_idx += 1
                else:
                    break
        if best_frame is not None:
            last_arr = best_frame.reformat(format="rgb24").to_ndarray()
            while len(frames) < len(target_pts_list):
                frames.append(last_arr.copy())
    finally:
        container.close()

    if not frames:
        raise ValueError(f"Failed to extract any frames from video: {video_path}")
    metadata = _build_video_metadata(
        fps=fps,
        total_frames=total_frames,
        sampled_indices=sampled_indices,
        backend="pyav",
    )
    return np.stack(frames), metadata


def _load_video_frames_decord_with_metadata(
    video_path: str,
    num_frames: int = 8,
    temporal_patch_size: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    from decord import VideoReader
    from decord import cpu as decord_cpu

    vr = VideoReader(video_path, ctx=decord_cpu(), num_threads=1)
    total_frames = len(vr)
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    fps = vr.get_avg_fps()
    if fps <= 0:
        raise ValueError(f"Video has invalid fps ({fps}): {video_path}")

    num_frames, timestamps_s = _compute_video_timestamps(
        total_frames / fps,
        num_frames,
        total_frames,
        num_frames,
        temporal_patch_size,
    )
    indices = [max(0, min(int(ts * fps), total_frames - 1)) for ts in timestamps_s]
    metadata = _build_video_metadata(
        fps=fps,
        total_frames=total_frames,
        sampled_indices=indices,
        backend="decord",
    )
    return vr.get_batch(indices).asnumpy(), metadata


def load_video_frames_with_metadata(
    video_path: str,
    num_frames: int = 8,
    temporal_patch_size: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load sampled RGB frames plus vLLM native-video metadata."""
    if os.environ.get("NRL_VIDEO_BACKEND", "pyav") == "decord":
        return _load_video_frames_decord_with_metadata(
            video_path, num_frames, temporal_patch_size
        )
    return _load_video_frames_pyav_with_metadata(
        video_path, num_frames, temporal_patch_size
    )


def _get_regular_prompt(
    input_ids: torch.Tensor, input_lengths: torch.Tensor, index: int
) -> dict[str, Any]:
    valid_length = input_lengths[index].item()
    valid_ids = input_ids[index, :valid_length] if valid_length > 0 else input_ids[index, :0]
    return {"prompt_token_ids": valid_ids.tolist()}


def _coerce_hw_pair_list(value: Any) -> list[list[int]] | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.tolist()
    elif hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) == 0:
        return []
    first = value[0]
    if isinstance(first, (int, float)):
        return [[int(v) for v in value]]
    pairs: list[list[int]] = []
    for item in value:
        if torch.is_tensor(item):
            item = item.tolist()
        if hasattr(item, "tolist") and not isinstance(item, (list, tuple)):
            item = item.tolist()
        if not isinstance(item, (list, tuple)):
            continue
        pairs.append([int(v) for v in item])
    return pairs


def _extract_hw_pairs(values: Any) -> list[tuple[int, int]]:
    """Normalize nested image-size payloads into ``(h, w)`` pairs."""
    pairs = _coerce_hw_pair_list(values)
    if pairs is None:
        return []
    normalized: list[tuple[int, int]] = []
    for pair in pairs:
        if len(pair) < 2:
            continue
        h, w = int(pair[0]), int(pair[1])
        if h > 0 and w > 0:
            normalized.append((h, w))
    return normalized


def _resolve_sample_imgs_sizes(
    data: BatchedDataDict[GenerationDatumSpec], index: int
) -> list[list[int]] | None:
    imgs_sizes = data.get("imgs_sizes", None)
    if imgs_sizes is None:
        return None

    sample_sizes: Any = None
    if isinstance(imgs_sizes, PackedTensor):
        dedup_indices = getattr(imgs_sizes, "_dedup_indices", None)
        resolved_index = dedup_indices[index] if dedup_indices is not None else index
        sample_sizes = imgs_sizes.tensors[resolved_index]
    elif torch.is_tensor(imgs_sizes):
        sample_sizes = imgs_sizes[index] if imgs_sizes.ndim >= 3 else imgs_sizes
    elif isinstance(imgs_sizes, (list, tuple)):
        if len(imgs_sizes) == 0:
            sample_sizes = []
        elif index < len(imgs_sizes) and isinstance(
            imgs_sizes[index], (list, tuple, torch.Tensor)
        ):
            candidate = imgs_sizes[index]
            if len(candidate) == 0:
                sample_sizes = candidate
            else:
                first = candidate[0]
                if isinstance(first, (list, tuple, torch.Tensor)):
                    sample_sizes = candidate
                else:
                    sample_sizes = imgs_sizes
        else:
            sample_sizes = imgs_sizes

    return _coerce_hw_pair_list(sample_sizes)


def _get_sample_images(
    data: BatchedDataDict[GenerationDatumSpec], index: int
) -> list[Any] | None:
    images = data.get("vllm_images", None)
    if images is None:
        return None
    sample_images = images[index]
    if sample_images is None:
        return None
    if isinstance(sample_images, list):
        return sample_images
    if isinstance(sample_images, tuple):
        return list(sample_images)
    return [sample_images]


def _get_sample_list(
    data: BatchedDataDict[GenerationDatumSpec],
    key: str,
    index: int,
) -> list[Any]:
    values = data.get(key, None)
    if values is None or index >= len(values):
        return []
    sample_values = values[index]
    if sample_values is None:
        return []
    if isinstance(sample_values, list):
        return sample_values
    if isinstance(sample_values, tuple):
        return list(sample_values)
    return [sample_values]


def _coerce_vllm_image(
    image: Any, image_cache: dict[str, Image.Image] | None = None
) -> Any:
    """Load local image paths eagerly so vLLM receives image objects, not strings."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, os.PathLike):
        image = os.fspath(image)
    if isinstance(image, str):
        local_path = image.removeprefix("file://")
        should_resolve = image.startswith(("http://", "https://", "data:", "file://")) or Path(
            local_path
        ).exists()
        if should_resolve:
            cache_key = image
            if image_cache is not None and cache_key in image_cache:
                return image_cache[cache_key]
            resolved_image = resolve_to_image(local_path if image.startswith("file://") else image)
            if image_cache is not None:
                image_cache[cache_key] = resolved_image
            return resolved_image
    return image


def _get_debug_image_sizes(images: list[Any]) -> list[list[int] | None]:
    debug_sizes: list[list[int] | None] = []
    for image in images:
        image_size = getattr(image, "size", None)
        if isinstance(image_size, tuple) and len(image_size) == 2:
            debug_sizes.append([int(image_size[0]), int(image_size[1])])
        else:
            debug_sizes.append(None)
    return debug_sizes


def _prompt_debug_enabled() -> bool:
    return debug_enabled() or os.environ.get("NRL_DEBUG", "0") == "1"


def _emit_audio_payload_debug(
    *,
    index: int,
    audio_index: int,
    audio_path: str,
    waveform: np.ndarray,
    max_audio_duration: Any,
    cached: bool,
) -> None:
    if not _prompt_debug_enabled():
        return
    sample_rate = 16000
    payload = {
        "sample_index": index,
        "audio_index": audio_index,
        "path": audio_path,
        "basename": os.path.basename(audio_path),
        "waveform_len": int(len(waveform)),
        "duration_s": float(len(waveform) / sample_rate),
        "sample_rate": sample_rate,
        "max_audio_duration": max_audio_duration,
        "cached": cached,
    }
    if debug_enabled():
        write_stage("vllm_audio_payload", payload)
    if os.environ.get("NRL_DEBUG", "0") == "1":
        print(
            "[VLLM_AUDIO_PAYLOAD_DEBUG] "
            f"sample={index} audio_index={audio_index} "
            f"path={os.path.basename(audio_path)} "
            f"waveform_len={payload['waveform_len']} "
            f"duration_s={payload['duration_s']:.6f} "
            f"sample_rate={sample_rate} "
            f"max_audio_duration={max_audio_duration} "
            f"cached={cached}",
            flush=True,
        )


def _emit_prompt_debug(
    index: int,
    prompt_type: str,
    prompt_text: str | None = None,
    images: list[Any] | None = None,
    videos: list[Any] | None = None,
    audio_count: int = 0,
    mm_processor_kwargs: dict[str, Any] | None = None,
    fallback_reason: str | None = None,
    precomputed_count: int | None = None,
    expected_precomputed_count: int | None = None,
    native_video: bool = False,
) -> None:
    if not _prompt_debug_enabled():
        return
    mm_kwargs = mm_processor_kwargs or {}
    image_count = 0 if images is None else len(images)
    video_count = 0 if videos is None else len(videos)
    image_tag_count = prompt_text.count("<image>") if isinstance(prompt_text, str) else 0
    video_tag_count = prompt_text.count("<video>") if isinstance(prompt_text, str) else 0
    debug_payload = {
        "sample_index": index,
        "prompt_type": prompt_type,
        "image_count": image_count,
        "video_count": video_count,
        "audio_count": audio_count,
        "image_sizes": [] if images is None else _get_debug_image_sizes(images),
        "mm_processor_kwargs": mm_kwargs or None,
        "has_precomputed_imgs_sizes": "precomputed_imgs_sizes" in mm_kwargs,
        "fallback_reason": fallback_reason,
        "precomputed_count": precomputed_count,
        "expected_precomputed_count": expected_precomputed_count,
        "native_video": native_video,
        "text_image_tags": image_tag_count,
        "text_video_tags": video_tag_count,
    }
    if debug_enabled():
        write_stage("vllm_prompt_format", debug_payload)
    if os.environ.get("NRL_DEBUG", "0") == "1":
        print(
            "[VLLM_PARITY_DEBUG] "
            f"sample={index} type={prompt_type} "
            f"images={image_count} videos={video_count} audio={audio_count} "
            f"native_video={native_video} "
            f"text_image_tags={image_tag_count} text_video_tags={video_tag_count} "
            f"precomputed={precomputed_count}/{expected_precomputed_count} "
            f"mm_kwargs={mm_kwargs or {}} "
            f"fallback={fallback_reason}",
            flush=True,
        )


def _coerce_mm_scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _env_enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _precomputed_img_sizes_enabled() -> bool:
    return _env_enabled(_PRECOMPUTED_IMG_SIZES_ENV, "0")


def _hash_local_file(path: Any) -> str | None:
    if isinstance(path, os.PathLike):
        path = os.fspath(path)
    if not isinstance(path, str):
        return None
    local_path = path.removeprefix("file://")
    if path.startswith(("http://", "https://", "data:")):
        return None
    if not os.path.exists(local_path) or not os.path.isfile(local_path):
        return None
    digest = hashlib.sha256()
    try:
        with open(local_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        if np.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if torch.is_tensor(value):
        return value.tolist()
    if isinstance(value, Image.Image):
        return {
            "type": "PIL.Image",
            "mode": value.mode,
            "size": list(value.size),
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except Exception:
            pass
    return str(value)


def _row_value(
    data: BatchedDataDict[GenerationDatumSpec], key: str, index: int
) -> Any:
    values = data.get(key, None)
    if values is None:
        return None
    if torch.is_tensor(values):
        return (
            values[index]
            if values.ndim > 0 and index < values.shape[0]
            else values
        )
    try:
        return values[index]
    except Exception:
        return None


def _image_debug_records(
    image_refs: list[Any],
    image_payload: Any,
) -> list[dict[str, Any]]:
    images = image_payload if isinstance(image_payload, list) else [image_payload]
    records: list[dict[str, Any]] = []
    for image_index, image in enumerate(images):
        ref = image_refs[image_index] if image_index < len(image_refs) else None
        size = getattr(image, "size", None)
        records.append(
            {
                "index": image_index,
                "ref": str(ref) if ref is not None else None,
                "ref_sha256": _hash_local_file(ref),
                "mode": getattr(image, "mode", None),
                "size_wh": list(size) if isinstance(size, tuple) else None,
            }
        )
    return records


def _sampling_params_summary(sampling_params: Any) -> dict[str, Any] | None:
    if sampling_params is None:
        return None
    summary: dict[str, Any] = {}
    for attr in (
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "min_tokens",
        "stop",
        "stop_token_ids",
        "bad_words",
        "include_stop_str_in_output",
        "skip_special_tokens",
        "spaces_between_special_tokens",
    ):
        if hasattr(sampling_params, attr):
            summary[attr] = _jsonable(getattr(sampling_params, attr))
    return summary


def dump_vllm_request_boundary(
    data: BatchedDataDict[GenerationDatumSpec],
    prompt: dict[str, Any],
    sampling_params: Any,
    *,
    source_index: int,
    prompt_position: int,
) -> None:
    """Optionally append a compact vLLM request-boundary record.

    Set ``NEMO_RL_DUMP_VLLM_REQUESTS=/path/to/dump.jsonl`` to enable. The dump
    is intentionally bounded and records hashes/previews rather than full token
    arrays or image bytes.
    """
    global _DUMP_REQUEST_COUNT

    output_path = os.environ.get(_DUMP_VLLM_REQUESTS_ENV)
    if not output_path:
        return

    try:
        limit = int(
            os.environ.get(
                _DUMP_VLLM_REQUESTS_LIMIT_ENV,
                str(_DUMP_VLLM_REQUESTS_DEFAULT_LIMIT),
            )
        )
    except ValueError:
        limit = _DUMP_VLLM_REQUESTS_DEFAULT_LIMIT
    if limit <= 0:
        return

    with _DUMP_LOCK:
        if _DUMP_REQUEST_COUNT >= limit:
            return
        _DUMP_REQUEST_COUNT += 1

    input_ids = data.get("input_ids")
    input_lengths = data.get("input_lengths")
    token_ids: list[int] = []
    input_length = None
    if torch.is_tensor(input_ids) and torch.is_tensor(input_lengths):
        input_length = int(input_lengths[source_index].item())
        token_ids = input_ids[source_index, :input_length].tolist()

    mm_data = prompt.get("multi_modal_data", {})
    image_payload = mm_data.get("image") if isinstance(mm_data, dict) else None
    image_refs = _get_sample_list(data, "vllm_images", source_index)
    prompt_text = prompt.get("prompt")
    prompt_token_ids = prompt.get("prompt_token_ids")

    record = {
        "timestamp_ms": int(time.time() * 1000),
        "pid": os.getpid(),
        "source_index": source_index,
        "prompt_position": prompt_position,
        "dataset_idx": _jsonable(_row_value(data, "idx", source_index)),
        "task_name": _jsonable(_row_value(data, "task_name", source_index)),
        "prompt_keys": sorted(prompt.keys()),
        "multi_modal_keys": (
            sorted(mm_data.keys()) if isinstance(mm_data, dict) else []
        ),
        "prompt_text_len": len(prompt_text) if isinstance(prompt_text, str) else None,
        "prompt_text_sha256": (
            hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
            if isinstance(prompt_text, str)
            else None
        ),
        "prompt_text_preview": (
            prompt_text[:512] if isinstance(prompt_text, str) else None
        ),
        "prompt_token_ids_len": (
            len(prompt_token_ids) if isinstance(prompt_token_ids, list) else None
        ),
        "input_length": input_length,
        "input_token_ids_first64": token_ids[:64],
        "input_token_ids_last64": token_ids[-64:] if token_ids else [],
        "input_token_ids_sha256": (
            hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if token_ids
            else None
        ),
        "image_records": (
            _image_debug_records(image_refs, image_payload)
            if image_payload is not None
            else []
        ),
        "source_imgs_sizes": _jsonable(
            _resolve_sample_imgs_sizes(data, source_index)
        ),
        "precomputed_img_sizes_enabled": _precomputed_img_sizes_enabled(),
        "mm_processor_kwargs": _jsonable(prompt.get("mm_processor_kwargs")),
        "sampling_params": _sampling_params_summary(sampling_params),
    }

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        line = json.dumps(_jsonable(record), sort_keys=True, separators=(",", ":"))
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
    except Exception as exc:
        if os.environ.get("NRL_DEBUG", "0") == "1":
            print(
                f"[VLLM_REQUEST_DUMP] failed to write {output_path}: {exc}",
                flush=True,
            )


_VIDEO_PROMPT_STYLE_ENV = "NRL_VIDEO_PROMPT_STYLE"
_VIDEO_PROMPT_STYLE_SFT_V2_GROUPED = "sft_v2_grouped"
_VIDEO_PROMPT_STYLE_DEFAULT = _VIDEO_PROMPT_STYLE_SFT_V2_GROUPED


def _is_sft_v2_grouped_video_prompt(style: Any) -> bool:
    if style is None:
        style = os.environ.get(_VIDEO_PROMPT_STYLE_ENV, _VIDEO_PROMPT_STYLE_DEFAULT)
    return str(style).strip().lower() == _VIDEO_PROMPT_STYLE_SFT_V2_GROUPED


def _build_mm_processor_kwargs(
    data: BatchedDataDict[GenerationDatumSpec],
    index: int,
    *,
    max_num_tiles: Any = None,
    max_num_patches: Any = None,
    include_precomputed_sizes: bool = False,
    expected_precomputed_count: int | None = None,
) -> tuple[dict[str, Any], int | None, bool | None]:
    mm_processor_kwargs: dict[str, Any] = {}

    if max_num_tiles is None:
        max_num_tiles_values = data.get("vllm_max_num_tiles", None)
        max_num_tiles = _get_row_scalar(max_num_tiles_values, index)
    if max_num_tiles is not None:
        mm_processor_kwargs["max_num_tiles"] = _coerce_mm_scalar(max_num_tiles)

    if max_num_patches is None:
        max_num_patches_values = data.get("vllm_max_num_patches", None)
        max_num_patches = _get_row_scalar(max_num_patches_values, index)
    if max_num_patches is not None:
        mm_processor_kwargs["max_num_patches"] = _coerce_mm_scalar(max_num_patches)

    precomputed_count: int | None = None
    precomputed_matches: bool | None = None
    if include_precomputed_sizes and _precomputed_img_sizes_enabled():
        sample_sizes = _resolve_sample_imgs_sizes(data, index)
        if sample_sizes is not None:
            precomputed_count = len(sample_sizes)
            precomputed_matches = (
                expected_precomputed_count is None
                or precomputed_count == expected_precomputed_count
            )
            if precomputed_matches:
                mm_processor_kwargs["precomputed_imgs_sizes"] = sample_sizes
            elif os.environ.get("NRL_DEBUG", "0") == "1":
                print(
                    "[VLLM_PRECOMPUTED_SIZES] "
                    f"sample={index} skipped because count mismatch: "
                    f"precomputed={precomputed_count} "
                    f"expected={expected_precomputed_count}",
                    flush=True,
                )

    return mm_processor_kwargs, precomputed_count, precomputed_matches


def _maybe_attach_mm_processor_kwargs(
    prompt_dict: dict[str, Any],
    data: BatchedDataDict[GenerationDatumSpec],
    index: int,
    *,
    max_num_tiles: Any = None,
    max_num_patches: Any = None,
    include_precomputed_sizes: bool = False,
    expected_precomputed_count: int | None = None,
) -> tuple[dict[str, Any], int | None, bool | None]:
    mm_processor_kwargs, precomputed_count, precomputed_matches = (
        _build_mm_processor_kwargs(
            data,
            index,
            max_num_tiles=max_num_tiles,
            max_num_patches=max_num_patches,
            include_precomputed_sizes=include_precomputed_sizes,
            expected_precomputed_count=expected_precomputed_count,
        )
    )
    if mm_processor_kwargs:
        prompt_dict["mm_processor_kwargs"] = mm_processor_kwargs
    return mm_processor_kwargs, precomputed_count, precomputed_matches


def _get_row_scalar(values: Any, index: int) -> Any:
    if values is None or index >= len(values):
        return None
    return values[index]


def _build_multimodal_prompt(
    data: BatchedDataDict[GenerationDatumSpec],
    index: int,
    prompt_text: str,
    sample_images: list[Any],
    *,
    max_num_tiles: Any = None,
    max_num_patches: Any = None,
    image_cache: dict[str, Image.Image] | None = None,
) -> dict[str, Any]:
    resolved_images = [
        _coerce_vllm_image(image, image_cache=image_cache) for image in sample_images
    ]
    prompt_dict: dict[str, Any] = {"prompt": prompt_text}
    prompt_dict["multi_modal_data"] = {
        "image": resolved_images[0] if len(resolved_images) == 1 else resolved_images
    }

    mm_processor_kwargs, precomputed_count, _ = _maybe_attach_mm_processor_kwargs(
        prompt_dict,
        data,
        index,
        max_num_tiles=max_num_tiles,
        max_num_patches=max_num_patches,
        include_precomputed_sizes=True,
        expected_precomputed_count=len(resolved_images),
    )

    _emit_prompt_debug(
        index,
        prompt_type="multimodal_prompt",
        prompt_text=prompt_text,
        images=resolved_images,
        mm_processor_kwargs=mm_processor_kwargs,
        precomputed_count=precomputed_count,
        expected_precomputed_count=len(resolved_images),
    )
    return prompt_dict


def _build_omni_multimodal_prompt(
    data: BatchedDataDict[GenerationDatumSpec],
    index: int,
    prompt_text: str,
    *,
    image_cache: dict[str, Image.Image] | None = None,
) -> dict[str, Any]:
    images = _get_sample_list(data, "vllm_images", index)
    videos = _get_sample_list(data, "vllm_videos", index)
    audio_paths = _get_sample_list(data, "vllm_audio_paths", index)
    cached_waveforms = _get_sample_list(data, "vllm_audio_waveforms", index)

    resolved_images = [
        _coerce_vllm_image(image, image_cache=image_cache) for image in images
    ]
    video_items: list[tuple[np.ndarray, dict[str, Any]]] = []
    num_frames = _get_row_scalar(data.get("vllm_num_frames", None), index) or 8
    temporal_patch_size = (
        _get_row_scalar(data.get("vllm_temporal_patch_size", None), index) or 1
    )
    video_prompt_style = _get_row_scalar(
        data.get("vllm_video_prompt_style", None), index
    )
    use_sft_v2_grouped_video_prompt = _is_sft_v2_grouped_video_prompt(
        video_prompt_style
    )
    if not use_sft_v2_grouped_video_prompt and videos:
        raise ValueError(
            "Native vLLM video generation only supports "
            f"{_VIDEO_PROMPT_STYLE_ENV}={_VIDEO_PROMPT_STYLE_SFT_V2_GROUPED!r}; "
            f"got {video_prompt_style!r}."
        )
    video_frame_indices = _get_sample_list(data, "vllm_video_frame_indices", index)
    video_fps = _get_sample_list(data, "vllm_video_fps", index)
    for video_index, video_path in enumerate(videos):
        frames, metadata = load_video_frames_with_metadata(
            video_path,
            num_frames=int(num_frames),
            temporal_patch_size=int(temporal_patch_size),
        )
        if video_index < len(video_frame_indices):
            frame_indices = video_frame_indices[video_index]
            if torch.is_tensor(frame_indices):
                frame_indices = frame_indices.tolist()
            if isinstance(frame_indices, (list, tuple)) and frame_indices:
                metadata["frames_indices"] = [
                    int(frame_index) for frame_index in frame_indices
                ]
        if video_index < len(video_fps):
            fps = video_fps[video_index]
            if torch.is_tensor(fps):
                fps = fps.item()
            if fps:
                metadata["fps"] = float(fps)
        video_items.append((frames, metadata))

    max_audio_duration = _get_row_scalar(
        data.get("vllm_max_audio_duration", None), index
    )
    audio_waveforms: list[np.ndarray] = []
    for audio_index, audio_path in enumerate(audio_paths):
        cached = cached_waveforms[audio_index] if audio_index < len(cached_waveforms) else None
        if isinstance(cached, np.ndarray):
            waveform = cached
            used_cached = True
        else:
            waveform = load_audio_waveform(
                audio_path,
                max_duration=max_audio_duration,
                raise_on_failure=True,
            )
            used_cached = False
        if waveform is None:
            raise AudioLoadError(audio_path, reason="empty waveform")
        _emit_audio_payload_debug(
            index=index,
            audio_index=audio_index,
            audio_path=audio_path,
            waveform=waveform,
            max_audio_duration=max_audio_duration,
            cached=used_cached,
        )
        audio_waveforms.append(waveform)

    image_payload = resolved_images
    if not image_payload and not video_items and not audio_waveforms:
        return _get_regular_prompt(data["input_ids"], data["input_lengths"], index)

    prompt_dict: dict[str, Any] = {"prompt": prompt_text, "multi_modal_data": {}}
    if image_payload:
        prompt_dict["multi_modal_data"]["image"] = (
            image_payload[0] if len(image_payload) == 1 else image_payload
        )
    if video_items:
        prompt_dict["multi_modal_data"]["video"] = (
            video_items[0] if len(video_items) == 1 else video_items
        )
    if audio_waveforms:
        prompt_dict["multi_modal_data"]["audio"] = (
            audio_waveforms[0] if len(audio_waveforms) == 1 else audio_waveforms
        )

    mm_processor_kwargs, precomputed_count, _ = _maybe_attach_mm_processor_kwargs(
        prompt_dict,
        data,
        index,
        include_precomputed_sizes=bool(image_payload),
        expected_precomputed_count=len(image_payload) if image_payload else None,
    )

    _emit_prompt_debug(
        index,
        prompt_type="omni_multimodal_prompt",
        prompt_text=prompt_text,
        images=image_payload,
        videos=video_items,
        audio_count=len(audio_waveforms),
        mm_processor_kwargs=mm_processor_kwargs,
        precomputed_count=precomputed_count,
        expected_precomputed_count=len(image_payload) if image_payload else None,
        native_video=bool(video_items),
    )
    return prompt_dict


def _format_prompts_from_compact_payload(
    data: BatchedDataDict[GenerationDatumSpec],
    compact: dict[str, Any],
    start_idx: int,
    end_idx: int,
    image_cache: dict[str, Image.Image] | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct per-row vLLM prompt dicts from a compact image payload."""
    input_ids = data["input_ids"]
    input_lengths = data["input_lengths"]

    schema_version = compact.get("schema_version")
    if schema_version is not None and schema_version != 2:
        raise ValueError(
            f"Unsupported compact payload schema version: {schema_version}"
        )

    row_use_token_prompt: list[bool] = compact["row_use_token_prompt"]
    row_content_idx: list[int] = compact["row_content_idx"]
    row_image_ref_indices: list[list[int]] = compact["row_image_ref_indices"]
    unique_contents: list[str] = compact["unique_contents"]
    unique_images: list[Any] = compact["unique_images"]
    row_max_num_tiles: list[Any] = compact.get("row_max_num_tiles", [])
    row_max_num_patches: list[Any] = compact.get("row_max_num_patches", [])

    prompts: list[dict[str, Any]] = []
    for index in range(start_idx, end_idx):
        if row_use_token_prompt[index]:
            prompt = _get_regular_prompt(input_ids, input_lengths, index)
            _emit_prompt_debug(
                index,
                prompt_type="token_ids",
                fallback_reason="compact_token_prompt",
            )
            prompts.append(prompt)
            continue

        content_index = row_content_idx[index]
        if content_index < 0 or content_index >= len(unique_contents):
            raise ValueError(
                f"Compact payload content index {content_index} is out of range"
            )

        sample_images = [unique_images[ref] for ref in row_image_ref_indices[index]]
        if len(sample_images) == 0:
            prompt = _get_regular_prompt(input_ids, input_lengths, index)
            _emit_prompt_debug(
                index,
                prompt_type="token_ids",
                fallback_reason="missing_vllm_images_compact",
            )
            prompts.append(prompt)
            continue

        prompts.append(
            _build_multimodal_prompt(
                data,
                index,
                unique_contents[content_index],
                sample_images,
                max_num_tiles=_get_row_scalar(row_max_num_tiles, index),
                max_num_patches=_get_row_scalar(row_max_num_patches, index),
                image_cache=image_cache,
            )
        )

    return prompts


def format_prompt_for_vllm_generation(
    data: BatchedDataDict[GenerationDatumSpec], sample_idx: Optional[int] = None
) -> list[dict[str, Any]] | dict[str, Any]:
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
    image_cache: dict[str, Image.Image] = {}

    if "vllm_mm_compact_payload" in data:
        prompts = _format_prompts_from_compact_payload(
            data, data["vllm_mm_compact_payload"], start_idx, end_idx, image_cache
        )
        return prompts if return_all else prompts[0]

    if "vllm_content" in data:
        for i in range(start_idx, end_idx):
            msg = data["vllm_content"][i]
            if msg is None:
                prompt = _get_regular_prompt(input_ids, input_lengths, i)
                _emit_prompt_debug(
                    i,
                    prompt_type="token_ids",
                    fallback_reason="missing_vllm_content",
                )
                prompts.append(prompt)
                continue

            sample_videos = _get_sample_list(data, "vllm_videos", i)
            sample_audio_paths = _get_sample_list(data, "vllm_audio_paths", i)
            if sample_videos or sample_audio_paths:
                prompts.append(
                    _build_omni_multimodal_prompt(
                        data,
                        i,
                        msg,
                        image_cache=image_cache,
                    )
                )
                continue

            sample_images = _get_sample_images(data, i)
            if sample_images is None or len(sample_images) == 0:
                prompt = _get_regular_prompt(input_ids, input_lengths, i)
                _emit_prompt_debug(
                    i,
                    prompt_type="token_ids",
                    fallback_reason="missing_vllm_images",
                )
                prompts.append(prompt)
                continue

            prompts.append(
                _build_multimodal_prompt(
                    data,
                    i,
                    msg,
                    sample_images,
                    image_cache=image_cache,
                )
            )
    else:
        for i in range(start_idx, end_idx):
            prompt = _get_regular_prompt(input_ids, input_lengths, i)
            _emit_prompt_debug(i, prompt_type="token_ids")
            prompts.append(prompt)

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


# =============================================================================
# Video frame loaders ported from Nemo-RL-Omni for MPO-VLM (load_video_frames
# is imported by examples/run_vlm_mpo.py).
#
# All heavy backends (av, decord) are lazy-imported inside the function bodies
# so this stays driver-safe.
# =============================================================================

def _compute_video_timestamps(
    total_duration: float, num_frames: int, total_frames_in_file: int,
    original_num_frames: int, temporal_patch_size: int, video_path: str,
) -> tuple[int, list[float]]:
    """Compute segment-midpoint timestamps matching SFT get_seq_frames_v3.

    Returns ``(final_num_frames, timestamps_in_seconds)``.
    """
    num_frames = min(num_frames, total_frames_in_file)

    if temporal_patch_size > 1:
        pre_round = num_frames
        rounded_down = (num_frames // temporal_patch_size) * temporal_patch_size
        rounded_up = rounded_down + temporal_patch_size
        if rounded_up <= total_frames_in_file and rounded_up <= original_num_frames:
            num_frames = rounded_up
        else:
            num_frames = max(temporal_patch_size, rounded_down)
        if os.environ.get("NRL_DEBUG", "0") == "1":
            print(
                f"[FRAME_SAMPLE_FIX] load_video_frames: T={temporal_patch_size} "
                f"pre_round={pre_round} -> post_round={num_frames} "
                f"(rounded_down={rounded_down} rounded_up={rounded_up} "
                f"original_requested={original_num_frames}) "
                f"total_frames={total_frames_in_file} video={video_path.split('/')[-1]}",
                flush=True,
            )

    if num_frames == 1:
        timestamps_s = [total_duration / 2.0]
    else:
        effective_span = max(total_duration - 1, 0)
        seg_size = effective_span / num_frames
        timestamps_s = [seg_size * (i + 0.5) for i in range(num_frames)]

    return num_frames, timestamps_s




def _load_video_frames_pyav(video_path: str, num_frames: int = 8, temporal_patch_size: int = 1) -> np.ndarray:
    """Load video frames using PyAV, matching SFT's AVDecoder.get_clips().

    PyAV (the Python binding for FFmpeg) is the same backend that SFT's
    Megatron-Energon ``AVDecoder`` uses internally.  This function replicates
    the AVDecoder's seek-then-decode-forward strategy:

    1. Seek once to the keyframe before the *first* target timestamp.
    2. Decode forward through the stream, collecting frames as each target
       PTS is reached (single-pass, no redundant re-seeking).
    3. Convert to rgb24 numpy array.

    This avoids the frame-index truncation issue in the decord path where
    ``int(timestamp * fps)`` can select an adjacent frame.
    """
    import av

    try:
        container = av.open(video_path)
    except av.error.InvalidDataError as e:
        raise ValueError(f"Cannot open video (corrupt/unreadable): {video_path}") from e

    stream = container.streams.video[0]
    stream.codec_context.thread_type = "NONE"

    fps = float(stream.average_rate) if stream.average_rate else 0.0
    if fps <= 0:
        container.close()
        raise ValueError(f"Video has invalid fps ({fps}): {video_path}")

    # Prefer stream.frames; fall back to duration-based estimate to avoid
    # decoding the entire file just to count frames.
    total_frames = stream.frames
    if total_frames <= 0:
        if stream.duration and stream.time_base:
            total_duration_est = float(stream.duration * stream.time_base)
        elif container.duration:
            total_duration_est = container.duration / av.time_base
        else:
            total_duration_est = 0.0
        total_frames = max(1, int(total_duration_est * fps))
    total_duration = total_frames / fps

    original_num_frames = num_frames
    num_frames, timestamps_s = _compute_video_timestamps(
        total_duration, num_frames, total_frames, original_num_frames,
        temporal_patch_size, video_path,
    )

    time_base = float(stream.time_base) if stream.time_base else 1.0 / fps
    target_pts_list = [int(ts / time_base) for ts in timestamps_s]

    _debug = os.environ.get("NRL_DEBUG", "0") == "1"

    # Single-pass decode matching SFT's FastseekReaderByPts.seek_read():
    # A frame is selected when its display interval *covers* the target PTS,
    # i.e. target_pts < frame.pts + frame.duration (not frame.pts >= target).
    frames: list[np.ndarray] = []
    _collected_pts: list = []
    _pts_none_count = 0
    target_idx_at_loop_end = 0
    try:
        first_pts = max(0, target_pts_list[0]) if target_pts_list else 0
        container.seek(first_pts, stream=stream, any_frame=False)

        target_idx = 0
        best_frame = None
        frame_counter = 0

        for frame in container.decode(video=0):
            if target_idx >= len(target_pts_list):
                break

            best_frame = frame

            if frame.pts is None:
                _pts_none_count += 1
                frame_counter += 1
                while target_idx < len(target_pts_list) and frame_counter >= target_idx + 1:
                    frames.append(frame.reformat(format="rgb24").to_ndarray())
                    _collected_pts.append(None)
                    target_idx += 1
                continue

            frame_counter += 1
            frame_end = frame.pts + (frame.duration if frame.duration else 1)

            while target_idx < len(target_pts_list):
                target_pts = target_pts_list[target_idx]
                if target_pts < frame_end:
                    frames.append(frame.reformat(format="rgb24").to_ndarray())
                    _collected_pts.append(frame.pts)
                    target_idx += 1
                else:
                    break

        target_idx_at_loop_end = target_idx
        if best_frame is not None:
            last_arr = best_frame.reformat(format="rgb24").to_ndarray()
            while len(frames) < len(target_pts_list):
                frames.append(last_arr.copy())
                _collected_pts.append("filled")
    except (av.error.InvalidDataError, av.error.EOFError) as e:
        if _debug:
            print(
                f"[PYAV_WARN] Decode error at frame {len(frames)}/{len(target_pts_list)} "
                f"for {video_path}: {e}",
                flush=True,
            )
        if frames:
            while len(frames) < len(target_pts_list):
                frames.append(frames[-1].copy())
                _collected_pts.append("error_filled")

    container.close()

    if not frames:
        raise ValueError(f"Failed to extract any frames from video: {video_path}")

    if _pts_none_count > 0 and _debug:
        print(
            f"[PYAV_PTS_WARN] {_pts_none_count} frames had pts=None in {video_path.split('/')[-1]}, "
            f"used sequential fallback",
            flush=True,
        )

    frames_nd = np.stack(frames)

    if _debug:
        print(
            f"[PYAV_FRAME_SELECTION] _load_video_frames_pyav: "
            f"targets={len(target_pts_list)} collected={len(frames)} "
            f"filled={len(frames) - target_idx_at_loop_end} "
            f"pts_none_frames={_pts_none_count} "
            f"frame_selection='display_interval' "
            f"first3_target_pts={target_pts_list[:3]} "
            f"first3_frame_pts={_collected_pts[:3]} "
            f"timestamps_s(first 5)={[f'{t:.4f}' for t in timestamps_s[:5]]} "
            f"total_frames={total_frames} num_frames={num_frames} "
            f"fps={fps:.2f} total_duration={total_duration:.4f}s "
            f"video={video_path.split('/')[-1]}",
            flush=True,
        )

    return frames_nd




def _load_video_frames_decord(video_path: str, num_frames: int = 8, temporal_patch_size: int = 1) -> np.ndarray:
    """Load video frames using decord (legacy path, kept as fallback)."""
    from decord import VideoReader, cpu as _decord_cpu

    vr = VideoReader(video_path, ctx=_decord_cpu(), num_threads=1)
    total_frames = len(vr)
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")

    fps = vr.get_avg_fps()
    if fps <= 0:
        raise ValueError(f"Video has invalid fps ({fps}): {video_path}")
    total_duration = total_frames / fps

    original_num_frames = num_frames
    num_frames, timestamps_s = _compute_video_timestamps(
        total_duration, num_frames, total_frames, original_num_frames,
        temporal_patch_size, video_path,
    )

    frame_indices = [max(0, min(int(t * fps), total_frames - 1)) for t in timestamps_s]

    if os.environ.get("NRL_DEBUG", "0") == "1":
        if total_duration < 1.0:
            print(
                f"[SHORT_VIDEO_WARN] decord load_video_frames: total_duration={total_duration:.4f}s < 1.0s, "
                f"video={video_path.split('/')[-1]}",
                flush=True,
            )
        print(
            f"[FRAME_SAMPLE_FIX] decord timestamp-based indices (first 5): "
            f"{frame_indices[:5]} timestamps_s(first 5)={[f'{t:.4f}' for t in timestamps_s[:5]]} "
            f"total_frames={total_frames} num_frames={num_frames} "
            f"fps={fps:.2f} total_duration={total_duration:.4f}s",
            flush=True,
        )

    frames_nd = vr.get_batch(frame_indices).asnumpy()

    if frames_nd.shape[0] == 0:
        raise ValueError(f"Failed to extract any frames from video: {video_path}")

    return frames_nd


# NRL_VIDEO_BACKEND controls which decoder to use.
#   "pyav"   - PyAV (default, matches SFT's AVDecoder)
#   "decord" - decord VideoReader (legacy)
_VIDEO_BACKEND = os.environ.get("NRL_VIDEO_BACKEND", "pyav")




def load_video_frames(video_path: str, num_frames: int = 8, temporal_patch_size: int = 1) -> np.ndarray:
    """Load a video and return uniformly sampled frames as a numpy array.

    Uses PyAV by default to match SFT's ``AVDecoder.get_clips(video_unit="seconds")``
    which seeks to the nearest keyframe then decodes forward to the target
    timestamp.  Set ``NRL_VIDEO_BACKEND=decord`` to use the legacy decord path.

    Args:
        video_path: Path to video file.
        num_frames: Number of frames to sample.
        temporal_patch_size: Conv3D tubelet size T. Frame count is rounded to
            a multiple of T (matching SFT ``video_to_frames``).

    Returns:
        numpy array of shape (num_frames, height, width, 3) with RGB values.
    """
    if _VIDEO_BACKEND == "decord":
        return _load_video_frames_decord(video_path, num_frames, temporal_patch_size)
    return _load_video_frames_pyav(video_path, num_frames, temporal_patch_size)
