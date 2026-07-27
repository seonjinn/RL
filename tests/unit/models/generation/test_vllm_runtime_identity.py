import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

_MODULE_PATH = (
    Path(__file__).parents[4]
    / "nemo_rl"
    / "models"
    / "generation"
    / "vllm"
    / "runtime_identity.py"
)
_SPEC = importlib.util.spec_from_file_location("vllm_runtime_identity", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_vllm_runtime_identity = _MODULE.build_vllm_runtime_identity


class TargetModel:
    pass


class DFlashProposer:
    def __init__(self, model):
        self.model = model


class EagleProposer:
    def __init__(self, model):
        self.model = model


class Qwen3DFlashModel:
    pass


@dataclass(frozen=True)
class Descriptor:
    num_tokens: int


class FULL:
    value = 3
    name = "FULL"


def _runner(
    *,
    method="dflash",
    proposer=True,
    expected=(4, 8, 12, 16, 20, 24, 28, 32),
    captured=(4, 8, 12, 16, 20, 24, 28, 32),
    keys_initialized=True,
):
    entries = {
        Descriptor(size): SimpleNamespace(cudagraph=object()) for size in captured
    }
    target_wrapper = SimpleNamespace(
        runtime_mode=FULL(),
        concrete_cudagraph_entries=entries,
        runnable=TargetModel(),
    )
    spec_config = (
        SimpleNamespace(
            method=method,
            model=f"/checkpoints/{method}",
            num_speculative_tokens=3,
        )
        if method != "none"
        else None
    )
    proposer_value = None
    if proposer:
        cls = DFlashProposer if method == "dflash" else EagleProposer
        proposer_value = cls(Qwen3DFlashModel())
    return SimpleNamespace(
        model=target_wrapper,
        drafter=proposer_value,
        cudagraph_dispatcher=SimpleNamespace(
            keys_initialized=keys_initialized,
            cudagraph_keys={FULL(): {Descriptor(size) for size in expected}},
        ),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model="Qwen/Qwen3-30B-A3B"),
            speculative_config=spec_config,
            compilation_config=SimpleNamespace(
                cudagraph_mode="FULL",
                cudagraph_capture_sizes=list(expected),
            ),
        ),
    )


def test_runtime_identity_reports_initialized_dflash_and_completed_exact_graphs():
    identity = build_vllm_runtime_identity(_runner())

    assert identity["method"] == "dflash"
    assert identity["configured_method"] == "dflash"
    assert identity["proposer_class"] == "DFlashProposer"
    assert identity["draft_model_class"] == "Qwen3DFlashModel"
    assert identity["target_model_class"] == "TargetModel"
    assert identity["target_model_id"] == "Qwen/Qwen3-30B-A3B"
    assert identity["draft_model_id"] == "/checkpoints/dflash"
    assert identity["k"] == 3
    assert identity["cudagraph_mode"] == "FULL"
    assert identity["full_cudagraph_expected_sizes"] == [
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        32,
    ]
    assert identity["full_cudagraph_captured_sizes"] == [
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        32,
    ]
    assert identity["full_graph_ready"] is True


def test_runtime_identity_does_not_trust_config_without_an_initialized_proposer():
    identity = build_vllm_runtime_identity(_runner(proposer=False))

    assert identity["configured_method"] == "dflash"
    assert identity["method"] == "none"
    assert identity["k"] == 0
    assert identity["proposer_class"] is None


def test_runtime_identity_fails_closed_when_graph_capture_is_incomplete():
    identity = build_vllm_runtime_identity(
        _runner(captured=(4, 8, 12), keys_initialized=True)
    )

    assert identity["full_graph_ready"] is False
    assert identity["full_cudagraph_expected_sizes"] == [
        4,
        8,
        12,
        16,
        20,
        24,
        28,
        32,
    ]
    assert identity["full_cudagraph_captured_sizes"] == [4, 8, 12]
    assert identity["graph_ready_source"] is None


def test_runtime_identity_fails_closed_when_dispatch_keys_are_not_initialized():
    identity = build_vllm_runtime_identity(_runner(keys_initialized=False))

    assert identity["full_graph_ready"] is False


def test_runtime_identity_resolves_eagle3_from_initialized_eagle_proposer():
    identity = build_vllm_runtime_identity(_runner(method="eagle3"))

    assert identity["method"] == "eagle3"


def test_runtime_identity_baseline_is_none():
    runner = _runner(method="none", proposer=False)

    identity = build_vllm_runtime_identity(runner)

    assert identity["method"] == "none"
    assert identity["configured_method"] == "none"
    assert identity["draft_model_class"] is None
    assert identity["k"] == 0
