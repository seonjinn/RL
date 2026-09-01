"""Immutable workload contract for the Q30 vLLM 0.28 benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping, TypeAlias


_ASSET_ROOT = (
    "/lustre/fsw/coreai_dlalgo_llm/users/sna/modelopt-specdec/assets/"
    "q30-base-opb-drafters-s4166-eval-v1"
)


def _drafter_paths() -> Mapping[str, str]:
    return MappingProxyType(
        {
            "dflash": f"{_ASSET_ROOT}/dflash-s4166",
            "dspark": f"{_ASSET_ROOT}/dspark-s4166",
        }
    )


def _drafter_block_sizes() -> Mapping[str, int]:
    return MappingProxyType({"dflash": 8, "dspark": 8})


Drafter: TypeAlias = Literal["dflash", "dspark"]
Stage: TypeAlias = Literal["calibration", "barrier"]
Method: TypeAlias = Literal["baseline", "fixed", "dynamic", "adaptive"]
Controller: TypeAlias = Literal[
    "none",
    "fixed_k",
    "k0_diagnostic",
    "dynamicsd",
    "dspark_adaptive_verification",
]


@dataclass(frozen=True, slots=True)
class ExperimentContract:
    """Literal, validated identity of the Lyris Q30 rollout workload."""

    target_path: str = f"{_ASSET_ROOT}/q30-base"
    drafter_paths: Mapping[str, str] = field(default_factory=_drafter_paths)
    drafter_block_sizes: Mapping[str, int] = field(
        default_factory=_drafter_block_sizes
    )
    container_path: str = (
        "/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/"
        "vllm-openai-v0.28.0-mrv2-dynamick-core-aarch64-ubuntu2404.sqsh"
    )
    vllm_version: str = "0.28.0"
    vllm_commit: str = "2cf0a6915ce544dc493a0990f2ea38d81601128a"
    max_tokens: int = 1_024
    prompt_count: int = 64
    generations_per_prompt: int = 32
    engine_count: int = 16
    requests_per_engine: int = 128
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    engine_coordination: str = "external"
    cuda_graph_mode: str = "FULL_AND_PIECEWISE"
    temperature: float = 1.0
    top_p: float = 1.0
    calibration_batch_sizes: tuple[int, ...] = (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        96,
        128,
    )
    calibration_k_values: tuple[int, ...] = (0, 1, 2, 3, 5, 7)

    def __post_init__(self) -> None:
        expected_values = {
            "target_path": f"{_ASSET_ROOT}/q30-base",
            "drafter_paths": _drafter_paths(),
            "drafter_block_sizes": _drafter_block_sizes(),
            "container_path": (
                "/lustre/fsw/coreai_dlalgo_llm/users/sna/containers/"
                "vllm-openai-v0.28.0-mrv2-dynamick-core-aarch64-ubuntu2404.sqsh"
            ),
            "vllm_version": "0.28.0",
            "vllm_commit": "2cf0a6915ce544dc493a0990f2ea38d81601128a",
            "max_tokens": 1_024,
            "prompt_count": 64,
            "generations_per_prompt": 32,
            "engine_count": 16,
            "requests_per_engine": 128,
            "tensor_parallel_size": 1,
            "data_parallel_size": 1,
            "engine_coordination": "external",
            "cuda_graph_mode": "FULL_AND_PIECEWISE",
            "temperature": 1.0,
            "top_p": 1.0,
            "calibration_batch_sizes": (1, 2, 4, 8, 16, 32, 64, 96, 128),
            "calibration_k_values": (0, 1, 2, 3, 5, 7),
        }
        for field_name, expected in expected_values.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} must be {expected!r}")
        if self.global_request_count != self.engine_count * self.requests_per_engine:
            raise ValueError(
                "global_request_count must equal engine_count * requests_per_engine"
            )

    @property
    def global_request_count(self) -> int:
        """Return the fixed total work in one synchronous rollout step."""
        return self.prompt_count * self.generations_per_prompt


@dataclass(frozen=True, slots=True)
class MethodPlan:
    """One immutable calibration cell or barrier comparison arm."""

    key: str
    stage: Stage
    drafter: Drafter | None
    method: Method
    controller: Controller
    batch_size: int | None
    verifier_k: int | None
    physical_block_size: int | None

    def __post_init__(self) -> None:
        if self.stage == "calibration":
            self._validate_calibration()
        elif self.stage == "barrier":
            self._validate_barrier()
        else:
            raise ValueError(f"unsupported stage: {self.stage!r}")

    def _validate_calibration(self) -> None:
        contract = ExperimentContract()
        if self.drafter not in ("dflash", "dspark"):
            raise ValueError("calibration drafter must be dflash or dspark")
        if self.batch_size not in contract.calibration_batch_sizes:
            raise ValueError("calibration batch_size is outside the fixed grid")
        if self.verifier_k not in contract.calibration_k_values:
            raise ValueError("calibration verifier_k is outside the fixed grid")
        if self.method != "fixed":
            raise ValueError("calibration method must be fixed")
        expected_controller = (
            "k0_diagnostic" if self.verifier_k == 0 else "fixed_k"
        )
        if self.controller != expected_controller:
            raise ValueError(
                f"calibration controller must be {expected_controller!r}"
            )
        expected_block_size = contract.drafter_block_sizes[self.drafter]
        if self.physical_block_size != expected_block_size:
            raise ValueError(
                f"physical_block_size must be {expected_block_size} for {self.drafter}"
            )
        expected_key = (
            f"calibration_{self.drafter}_bs{self.batch_size}_k{self.verifier_k}"
        )
        if self.key != expected_key:
            raise ValueError(f"calibration key must be {expected_key!r}")

    def _validate_barrier(self) -> None:
        if self.batch_size is not None:
            raise ValueError("barrier rows must not set batch_size")
        if self.verifier_k is not None:
            raise ValueError("barrier verifier_k is selected after calibration")

        expected_identity: dict[
            str,
            tuple[Drafter | None, Method, Controller, int | None],
        ] = {
            "target_only": (None, "baseline", "none", None),
            "dflash_fixed_best": ("dflash", "fixed", "fixed_k", 8),
            "dflash_dynamicsd": ("dflash", "dynamic", "dynamicsd", 8),
            "dspark_fixed_best": ("dspark", "fixed", "fixed_k", 8),
            "dspark_dynamicsd": ("dspark", "dynamic", "dynamicsd", 8),
            "dspark_adaptive_verification": (
                "dspark",
                "adaptive",
                "dspark_adaptive_verification",
                8,
            ),
        }
        expected = expected_identity.get(self.key)
        actual = (
            self.drafter,
            self.method,
            self.controller,
            self.physical_block_size,
        )
        if expected is None or actual != expected:
            raise ValueError(f"invalid barrier method identity for {self.key!r}")


def build_calibration_rows(
    contract: ExperimentContract | None = None,
) -> tuple[MethodPlan, ...]:
    """Build the exhaustive fixed-K calibration grid for both drafters."""
    experiment = contract or ExperimentContract()
    rows: list[MethodPlan] = []
    for drafter in ("dflash", "dspark"):
        block_size = experiment.drafter_block_sizes[drafter]
        for batch_size in experiment.calibration_batch_sizes:
            for verifier_k in experiment.calibration_k_values:
                rows.append(
                    MethodPlan(
                        key=(
                            f"calibration_{drafter}_bs{batch_size}_k{verifier_k}"
                        ),
                        stage="calibration",
                        drafter=drafter,
                        method="fixed",
                        controller=(
                            "k0_diagnostic" if verifier_k == 0 else "fixed_k"
                        ),
                        batch_size=batch_size,
                        verifier_k=verifier_k,
                        physical_block_size=block_size,
                    )
                )
    return tuple(rows)


def build_barrier_rows(
    contract: ExperimentContract | None = None,
) -> tuple[MethodPlan, ...]:
    """Build distinct barrier arms whose K values are filled after calibration."""
    experiment = contract or ExperimentContract()
    block_sizes = experiment.drafter_block_sizes
    return (
        MethodPlan("target_only", "barrier", None, "baseline", "none", None, None, None),
        MethodPlan(
            "dflash_fixed_best",
            "barrier",
            "dflash",
            "fixed",
            "fixed_k",
            None,
            None,
            block_sizes["dflash"],
        ),
        MethodPlan(
            "dflash_dynamicsd",
            "barrier",
            "dflash",
            "dynamic",
            "dynamicsd",
            None,
            None,
            block_sizes["dflash"],
        ),
        MethodPlan(
            "dspark_fixed_best",
            "barrier",
            "dspark",
            "fixed",
            "fixed_k",
            None,
            None,
            block_sizes["dspark"],
        ),
        MethodPlan(
            "dspark_dynamicsd",
            "barrier",
            "dspark",
            "dynamic",
            "dynamicsd",
            None,
            None,
            block_sizes["dspark"],
        ),
        MethodPlan(
            "dspark_adaptive_verification",
            "barrier",
            "dspark",
            "adaptive",
            "dspark_adaptive_verification",
            None,
            None,
            block_sizes["dspark"],
        ),
    )
