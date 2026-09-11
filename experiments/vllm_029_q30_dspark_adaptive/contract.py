"""Immutable experiment identity for the vLLM 0.29 standalone study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ArmKey = Literal["baseline", "dspark_k5", "dspark_k7", "dspark_adaptive_k7"]
TargetAttentionBackend = Literal["FLEX_ATTENTION", "TRITON_ATTN"]


@dataclass(frozen=True, slots=True)
class ExperimentContract:
    """Pinned Q30 synchronous-rollout workload and runtime configuration."""

    account: str = "coreai_dlalgo_nemorl"
    vllm_version: str = "0.29.0"
    vllm_commit: str = "98dff2a81d747d1dba01a47f939f48c3526d4206"
    target_path: str = (
        "/lustre/fsw/portfolios/coreai/users/sna/hf-local/Qwen/Qwen3-30B-A3B"
    )
    drafter_path: str = (
        "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/sna/"
        "specdec_ptv23/ptv3_swa/"
        "sd2p3swa-q30-base-ptv3swe-dspark-b8-16n/exported-checkpoint-44000"
    )
    prompt_count: int = 64
    generations_per_prompt: int = 32
    worker_count: int = 16
    max_output_tokens: int = 4_096
    max_model_len: int = 8_192
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 32_768
    target_attention_backend: TargetAttentionBackend = "FLEX_ATTENTION"
    cuda_graph_mode: str = "FULL_AND_PIECEWISE"
    max_cudagraph_capture_size: int = 1_024
    temperature: float = 1.0
    top_p: float = 1.0
    base_seed: int = 20_260_911

    @property
    def global_sample_count(self) -> int:
        return self.prompt_count * self.generations_per_prompt

    @property
    def samples_per_worker(self) -> int:
        return self.global_sample_count // self.worker_count

    @property
    def prompts_per_worker(self) -> int:
        return self.prompt_count // self.worker_count


@dataclass(frozen=True, slots=True)
class Arm:
    """One matched benchmark arm."""

    key: ArmKey
    verifier_k: int | None
    adaptive_verification: bool

    def speculative_config(self, drafter_path: str) -> dict[str, object] | None:
        if self.verifier_k is None:
            return None
        return {
            "method": "dspark",
            "model": drafter_path,
            "num_speculative_tokens": self.verifier_k,
            "draft_tensor_parallel_size": 1,
            "attention_backend": "FLASH_ATTN",
            "draft_sample_method": "probabilistic",
            "enable_adaptive_verification": self.adaptive_verification,
        }


@dataclass(frozen=True, slots=True)
class WorkerAssignment:
    worker_index: int
    prompt_offset: int
    seed: int


def build_arms(_: ExperimentContract | None = None) -> tuple[Arm, ...]:
    """Return the exact publication comparison matrix."""
    return (
        Arm("baseline", None, False),
        Arm("dspark_k5", 5, False),
        Arm("dspark_k7", 7, False),
        Arm("dspark_adaptive_k7", 7, True),
    )


def worker_assignment(
    contract: ExperimentContract, worker_index: int
) -> WorkerAssignment:
    """Partition the 64 prompts and per-request seeds across 16 engines."""
    if type(worker_index) is not int or not 0 <= worker_index < contract.worker_count:
        raise ValueError("worker_index is outside the fixed 16-engine topology")
    return WorkerAssignment(
        worker_index=worker_index,
        prompt_offset=worker_index * contract.prompts_per_worker,
        seed=contract.base_seed + worker_index * contract.samples_per_worker,
    )
