import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from nemo_rl.utils.vllm_runtime_patches import (
    atomic_replace_text,
    ensure_draft_model_cudagraph_support,
    requires_draft_model_cudagraph_support,
)


VLLM_020_DRAFT_INIT = """\
        # Initialize drafter's cudagraph dispatcher if using spec decode.
        if self.speculative_config and (
            self.speculative_config.use_eagle()
            or self.speculative_config.uses_extract_hidden_states()
        ):
            assert isinstance(
                self.drafter,
                EagleProposer | DFlashProposer | ExtractHiddenStatesProposer,
            )
            self.drafter.initialize_cudagraph_keys(cudagraph_mode)
"""


def test_atomic_replace_preserves_mode_and_never_leaves_partial_text(
    tmp_path: Path,
) -> None:
    target = tmp_path / "gpu_model_runner.py"
    target.write_text("original")
    target.chmod(0o640)
    candidates = [f"content-{index}-" + (str(index) * 100_000) for index in range(8)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda content: atomic_replace_text(target, content), candidates))

    assert target.read_text() in candidates
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert list(tmp_path.glob(".gpu_model_runner.py.*.tmp")) == []


@pytest.mark.parametrize("method", ["draft_model", "pard2"])
def test_pard_methods_require_draft_cuda_graph_support(method: str) -> None:
    assert requires_draft_model_cudagraph_support(method, enforce_eager=False)


@pytest.mark.parametrize("method", [None, "eagle3", "suffix", "ngram"])
def test_non_pard_methods_do_not_require_generic_draft_patch(
    method: str | None,
) -> None:
    assert not requires_draft_model_cudagraph_support(
        method, enforce_eager=False, has_draft_model=False
    )


def test_implicit_draft_model_requires_cuda_graph_support() -> None:
    assert requires_draft_model_cudagraph_support(
        None, enforce_eager=False, has_draft_model=True
    )


def test_eager_pard_ablation_does_not_require_cuda_graph_patch() -> None:
    assert not requires_draft_model_cudagraph_support("draft_model", enforce_eager=True)


def test_adds_draft_model_to_vllm_020_cudagraph_initialization() -> None:
    patched, changed = ensure_draft_model_cudagraph_support(VLLM_020_DRAFT_INIT)

    assert changed is True
    assert "NRL_DRAFT_MODEL_CUDAGRAPH_INIT_PATCH" in patched
    assert "self.speculative_config.uses_draft_model()" in patched
    assert "DraftModelProposer" in patched


def test_whole_stock_source_does_not_false_positive_on_unrelated_tokens() -> None:
    stock_source = (
        "from vllm.v1.spec_decode.draft_model import DraftModelProposer\n"
        "if self.speculative_config.uses_draft_model():\n"
        "    other_drafter.initialize_cudagraph_keys(cudagraph_mode)\n"
        + VLLM_020_DRAFT_INIT
    )

    patched, changed = ensure_draft_model_cudagraph_support(stock_source)

    assert changed is True
    target_block = patched.split(
        "# Initialize drafter's cudagraph dispatcher if using spec decode.", 1
    )[1]
    assert "self.speculative_config.uses_draft_model()" in target_block


def test_draft_model_cudagraph_patch_is_idempotent() -> None:
    patched, _ = ensure_draft_model_cudagraph_support(VLLM_020_DRAFT_INIT)

    second_pass, changed = ensure_draft_model_cudagraph_support(patched)

    assert changed is False
    assert second_pass == patched


def test_rejects_marker_without_verified_draft_model_support() -> None:
    with pytest.raises(RuntimeError, match="CUDA Graph"):
        ensure_draft_model_cudagraph_support(
            "# NRL_DRAFT_MODEL_CUDAGRAPH_INIT_PATCH\n"
            "# truncated before DraftModelProposer initialization\n"
        )


def test_commented_tokens_do_not_count_as_cuda_graph_support() -> None:
    commented_only = VLLM_020_DRAFT_INIT.replace(
        "        if self.speculative_config and (\n",
        "        # NRL_DRAFT_MODEL_CUDAGRAPH_INIT_PATCH\n"
        "        # self.speculative_config.uses_draft_model()\n"
        "        # DraftModelProposer\n"
        "        if self.speculative_config and (\n",
    )

    with pytest.raises(RuntimeError, match="CUDA Graph"):
        ensure_draft_model_cudagraph_support(commented_only)


def test_rejects_unknown_vllm_source_instead_of_silent_eager_fallback() -> None:
    with pytest.raises(RuntimeError, match="CUDA Graph"):
        ensure_draft_model_cudagraph_support("unknown vLLM source layout\n")


def test_accepts_native_vllm_draft_model_cudagraph_support() -> None:
    native = VLLM_020_DRAFT_INIT.replace(
        "            or self.speculative_config.uses_extract_hidden_states()\n",
        "            or self.speculative_config.uses_draft_model()\n"
        "            or self.speculative_config.uses_extract_hidden_states()\n",
    ).replace(
        "                EagleProposer | DFlashProposer | ExtractHiddenStatesProposer,\n",
        "                EagleProposer\n"
        "                | DFlashProposer\n"
        "                | DraftModelProposer\n"
        "                | ExtractHiddenStatesProposer,\n",
    )

    verified, changed = ensure_draft_model_cudagraph_support(native)

    assert changed is False
    assert verified == native
