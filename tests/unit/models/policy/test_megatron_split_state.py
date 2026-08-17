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
"""CPU state-machine tests for MegatronPolicyWorkerImpl's split-API.

These tests cover the lifecycle and call-order invariants — they do NOT
exercise real distributed comms, the mcore scheduler, or the optimizer.
Numerical equivalence vs sync ``train()`` lives in the GPU parity tests.

The bugs these catch:
  - silent gradient over-counting if ``model.no_sync()`` is not wrapped
    around ``megatron_forward_backward`` (the mcore DDP hooks would
    dispatch a per-call reduce, ADDING to an already-reduced bucket).
  - PP>1 pipeline-schedule bypass if ``model.config.grad_sync_func`` is
    not nulled for the step's duration.
  - ``trainer_version`` advancing on abort.
  - ``zero_grad_buffer`` not called at begin (mcore's contiguous grad
    buffer leaks stale grads otherwise).
  - off-by-one in ``total_num_microbatches`` (used to scale MoE aux-loss).
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

# megatron.bridge is only available with the mcore extras. Without it the
# eager import of megatron_policy_worker (transitively imports megatron.bridge)
# fails at COLLECTION time on non-mcore shards, which then breaks every other
# test in that shard. importorskip stops collection cleanly here.
pytest.importorskip("megatron.bridge")

# Eagerly import the worker module so ``unittest.mock.patch`` can resolve
# attributes on it via ``getattr``. Without this the patch path
# ``nemo_rl.models.policy.workers.megatron_policy_worker.<symbol>`` fails
# at ``getattr(workers, "megatron_policy_worker")``.
import nemo_rl.models.policy.workers.megatron_policy_worker  # noqa: E402,F401

pytestmark = pytest.mark.mcore

# Module path of the worker under test
WORKER_MOD = "nemo_rl.models.policy.workers.megatron_policy_worker"


# ── Mock fabric ──────────────────────────────────────────────────────────


def _make_mock_model():
    """A mcore-DDP-shaped mock: exposes the methods + attributes the
    split-API touches, plus an ``inference_params`` attribute and a
    ``modules()`` that yields nothing (so the inference-cache reset loop
    is a no-op)."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.grad_sync_func = "ORIGINAL_GRAD_SYNC_FUNC"  # sentinel
    model.config.num_moe_experts = None  # disable MoE branch
    # no_sync() is a context manager — return a MagicMock that supports
    # __enter__/__exit__ so the `with self.model.no_sync():` block works.
    model.no_sync = MagicMock(
        return_value=MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        )
    )
    model.modules = MagicMock(return_value=iter([]))
    model.inference_params = None
    model.parameters = MagicMock(
        return_value=iter([])
    )  # no params for the rescale loop
    return model


def _make_worker(loss_type):
    """Construct a MegatronPolicyWorkerImpl instance with all heavy
    attributes mocked. Bypasses __init__ via ``object.__new__``."""
    # Lazy import so the module-level mcore imports happen inside the
    # mcore-marked test process.
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    w = object.__new__(MegatronPolicyWorkerImpl)
    w.model = _make_mock_model()
    w.optimizer = MagicMock()
    # MegatronOptimizer.step returns (success, grad_norm, num_zeros)
    w.optimizer.step.return_value = (True, 0.5, 0)
    w.optimizer.param_groups = [{"lr": 1e-4, "weight_decay": 0.01}]
    w.scheduler = MagicMock()
    w.scheduler.get_lr.return_value = 1e-4
    w.scheduler.get_wd.return_value = 0.01
    w.mcore_state = MagicMock()
    w.mcore_state.straggler_timer = None
    w.cfg = {
        "train_global_batch_size": 32,
        "train_micro_batch_size": 4,
        "megatron_cfg": {
            "empty_unused_memory_level": 0,
            "moe_per_layer_logging": False,
            "use_fused_linear_logprobs": False,
            # overlap_grad_reduce=False matches the production default and
            # the sync-GRPO path. finish_train_step relies on this to gate
            # the explicit start_grad_sync call.
            "distributed_data_parallel_config": {
                "overlap_grad_reduce": False,
            },
        },
    }
    w.dp_size = 2
    w.cp_size = 1
    w.sampling_params = None
    w.draft_model = None
    w.defer_fp32_logits = False
    w.dtype = torch.float32
    w._is_reward_model = False
    w._router_replay_enabled = False

    # Stash a loss_fn with the requested loss_type for tests that need one.
    w._test_loss_fn = MagicMock(loss_type=loss_type)
    return w


@pytest.fixture
def mock_module_symbols():
    """Patch every module-level symbol that the split-API methods call
    into. Yields a dict of name → mock for assertions."""
    # Make `aggregate_training_statistics` return ({}, scalar) — what the
    # finish path expects.
    agg_ret = ({"loss": [0.0]}, torch.tensor(0.5))

    patches = {
        "megatron_forward_backward": [
            {"loss": 0.5, "global_valid_seqs": 8.0, "global_valid_toks": 256.0}
        ],
        "get_microbatch_iterator": (iter([]), 2, 4, 16, 16),  # 2 pipeline mbs per call
        "LossPostProcessor": MagicMock(),
        "broadcast_loss_metrics_from_last_stage": lambda m: m,
        "get_pg_collection": MagicMock(mp=MagicMock()),
        "logical_and_across_model_parallel_group": lambda v, mp_group: v,
        "reduce_max_stat_across_model_parallel_group": lambda v, mp_group: v,
        "aggregate_training_statistics": agg_ret,
        "get_moe_metrics": MagicMock(return_value={}),
    }

    with (
        patch(
            f"{WORKER_MOD}.megatron_forward_backward",
            return_value=patches["megatron_forward_backward"],
        ) as mfb,
        patch(
            f"{WORKER_MOD}.get_microbatch_iterator",
            return_value=patches["get_microbatch_iterator"],
        ) as gmi,
        patch(
            f"{WORKER_MOD}.LossPostProcessor", return_value=patches["LossPostProcessor"]
        ) as lpp,
        patch(
            f"{WORKER_MOD}.broadcast_loss_metrics_from_last_stage",
            side_effect=patches["broadcast_loss_metrics_from_last_stage"],
        ) as bcast,
        patch(
            f"{WORKER_MOD}.get_pg_collection", return_value=patches["get_pg_collection"]
        ) as gpgc,
        patch(
            f"{WORKER_MOD}.logical_and_across_model_parallel_group",
            side_effect=patches["logical_and_across_model_parallel_group"],
        ) as land,
        patch(
            f"{WORKER_MOD}.reduce_max_stat_across_model_parallel_group",
            side_effect=patches["reduce_max_stat_across_model_parallel_group"],
        ) as rmax,
        patch(
            f"{WORKER_MOD}.aggregate_training_statistics",
            return_value=patches["aggregate_training_statistics"],
        ) as agg,
        patch(f"{WORKER_MOD}.get_moe_metrics", return_value={}) as moe,
        patch(f"{WORKER_MOD}.get_rerun_state_machine") as grsm,
        patch(f"{WORKER_MOD}.parallel_state") as pstate,
        patch("torch.distributed.all_reduce") as ar,
        patch("torch.cuda.empty_cache") as cec,
        patch("torch.cuda.get_device_name", return_value="H100"),
        patch("torch.distributed.get_rank", return_value=0),
    ):
        # rerun state machine: fire forward+backward once per train_microbatch
        rsm = MagicMock()
        rsm.should_run_forward_backward.side_effect = [True, False] * 100
        grsm.return_value = rsm

        # parallel_state mocks
        pstate.is_pipeline_last_stage.return_value = True
        pstate.get_data_parallel_group.return_value = MagicMock()

        yield {
            "mfb": mfb,
            "gmi": gmi,
            "lpp": lpp,
            "bcast": bcast,
            "gpgc": gpgc,
            "land": land,
            "rmax": rmax,
            "agg": agg,
            "moe": moe,
            "grsm": grsm,
            "pstate": pstate,
            "all_reduce": ar,
            "empty_cache": cec,
        }


def _fake_batch():
    """A minimal BatchedDataDict-ish object the mask-sum block can read.
    train_microbatch reads ``data["sample_mask"]``, ``data["token_mask"]``,
    and (only as a fallback for the no-token-mask path) ``data["input_ids"]``."""
    # 8 samples, all valid (mask=1); 256 valid tokens each
    sample_mask = torch.ones(8, dtype=torch.float32)
    token_mask = torch.ones(8, 257, dtype=torch.float32)  # token_mask[:, 1:] → 256 toks
    input_ids = torch.zeros(8, 257, dtype=torch.long)
    return {
        "sample_mask": sample_mask,
        "token_mask": token_mask,
        "input_ids": input_ids,
    }


# ── BEGIN ────────────────────────────────────────────────────────────────


class TestBegin:
    def test_opens_state(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn, gbs=16, mbs=4)
        assert w._train_step_state is not None
        assert w._train_step_state["loss_type"] == LossType.TOKEN_LEVEL
        assert w._train_step_state["gbs"] == 16
        assert w._train_step_state["mbs"] == 4
        assert w._train_step_state["total_num_microbatches"] == 0

    def test_calls_zero_grad_and_zero_grad_buffer(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.model.zero_grad_buffer.assert_called_once()
        w.optimizer.zero_grad.assert_called_once()
        w.model.train.assert_called_once()

    def test_saves_and_nulls_grad_sync_func(self, mock_module_symbols):
        """The PP scheduler's direct reduce dispatch must be suppressed
        for the duration of the step. Otherwise PP>1 silently corrupts
        grads even when ``no_sync`` is set on the bucket groups."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        assert w.model.config.grad_sync_func == "ORIGINAL_GRAD_SYNC_FUNC"
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w.model.config.grad_sync_func is None
        assert w._train_step_state["saved_grad_sync_func"] == "ORIGINAL_GRAD_SYNC_FUNC"

    def test_double_begin_raises(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        with pytest.raises(RuntimeError, match="already open"):
            w.begin_train_step(loss_fn=w._test_loss_fn)

    def test_uses_cfg_defaults_when_gbs_mbs_omitted(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w._train_step_state["gbs"] == w.cfg["train_global_batch_size"]
        assert w._train_step_state["mbs"] == w.cfg["train_micro_batch_size"]


# ── _assert_step_open ────────────────────────────────────────────────────


class TestAssertStepOpen:
    def test_raises_when_no_step_open(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        with pytest.raises(RuntimeError, match="no train step open"):
            w._assert_step_open()

    def test_train_microbatch_without_begin_raises(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        with pytest.raises(RuntimeError, match="no train step open"):
            w.train_microbatch(_fake_batch())

    def test_finish_without_begin_raises(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        with pytest.raises(RuntimeError, match="no train step open"):
            w.finish_train_step()


# ── train_microbatch ─────────────────────────────────────────────────────


class TestTrainMicrobatch:
    def test_wraps_forward_backward_in_no_sync(self, mock_module_symbols):
        """The single most important assertion in this file. Without the
        no_sync wrap, mcore DDP dispatches a per-call cross-DP reduce on
        the partially-accumulated buffer — silently corrupting grads."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        # no_sync() must have been ENTERED (called as a context manager).
        # MagicMock with __enter__/__exit__ records the __enter__ call.
        ctx = w.model.no_sync.return_value
        ctx.__enter__.assert_called()
        ctx.__exit__.assert_called()

    def test_invokes_megatron_forward_backward_once(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        assert mock_module_symbols["mfb"].call_count == 1

    def test_passes_placeholder_n_one_to_loss(self, mock_module_symbols):
        """The N=1 trick: loss must be called with global_valid_*=1 so it
        returns un-normalized sums; finish does the 1/N rescale."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        kwargs = mock_module_symbols["mfb"].call_args.kwargs
        # placeholder_n is a tensor(1.0)
        assert "global_valid_seqs" in kwargs
        assert "global_valid_toks" in kwargs
        assert float(kwargs["global_valid_seqs"].item()) == pytest.approx(1.0)
        assert float(kwargs["global_valid_toks"].item()) == pytest.approx(1.0)

    def test_accumulates_mask_sums_across_calls(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        # _fake_batch has sample_mask sum = 8, token_mask*sample_mask sum = 8*256 = 2048
        w.train_microbatch(_fake_batch())
        assert float(w._train_step_state["local_valid_seqs"].item()) == pytest.approx(
            8.0
        )
        assert float(w._train_step_state["local_valid_toks"].item()) == pytest.approx(
            2048.0
        )
        w.train_microbatch(_fake_batch())
        assert float(w._train_step_state["local_valid_seqs"].item()) == pytest.approx(
            16.0
        )
        assert float(w._train_step_state["local_valid_toks"].item()) == pytest.approx(
            4096.0
        )

    def test_total_num_microbatches_accumulates(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        # get_microbatch_iterator mock returns num_microbatches=2 per call
        w.train_microbatch(_fake_batch())
        w.train_microbatch(_fake_batch())
        w.train_microbatch(_fake_batch())
        assert w._train_step_state["total_num_microbatches"] == 6

    def test_does_not_call_optimizer_step(self, mock_module_symbols):
        """trainer_version semantics: optimizer.step() must NOT fire
        per train_microbatch — only at finish."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        w.train_microbatch(_fake_batch())
        w.optimizer.step.assert_not_called()


# ── finish_train_step ────────────────────────────────────────────────────


class TestFinish:
    def _setup_open_step(self, mock_module_symbols, loss_type):
        w = _make_worker(loss_type)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        return w

    def test_rescales_grads_with_inv_n(self, mock_module_symbols):
        """The 1/N rescale must happen ON the local main_grad BEFORE the
        cross-DP reduce — otherwise the reduce sees un-rescaled sums."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w.finish_train_step()
        # scale_gradients should have been called with some 1/N scalar < 1
        w.model.scale_gradients.assert_called_once()
        arg = w.model.scale_gradients.call_args.args[0]
        assert 0 < arg <= 1.0

    @pytest.mark.parametrize("overlap_grad_reduce", [False, True])
    def test_grad_sync_call_order_after_rescale(
        self, mock_module_symbols, overlap_grad_reduce
    ):
        """Call order matters: scale_gradients -> [start_grad_sync when
        overlap=True] -> finish_grad_sync -> optimizer.step.

        With overlap=False, Megatron's finish_grad_sync internally calls
        start_grad_sync(force_all_reduce=True), so calling start_grad_sync
        ourselves would double-reduce.
        """
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w.cfg["megatron_cfg"]["distributed_data_parallel_config"][
            "overlap_grad_reduce"
        ] = overlap_grad_reduce
        # Record call order via a shared list
        order: list[str] = []
        w.model.scale_gradients.side_effect = lambda s: order.append("scale")
        w.model.start_grad_sync.side_effect = lambda: order.append("start_sync")
        w.model.finish_grad_sync.side_effect = lambda: order.append("finish_sync")
        w.optimizer.step.side_effect = lambda: (
            order.append("opt_step") or (True, 0.5, 0)
        )
        w.finish_train_step()
        if overlap_grad_reduce:
            assert order == ["scale", "start_sync", "finish_sync", "opt_step"]
        else:
            assert order == ["scale", "finish_sync", "opt_step"]

    def test_picks_global_valid_toks_for_token_level_loss(self, mock_module_symbols):
        """N selection: TOKEN_LEVEL → N = global_valid_toks (not seqs)."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w.finish_train_step()
        # local_valid_toks accumulated = 2048; with mocked all_reduce as no-op,
        # global_valid_toks == 2048 → inv_n = 1/2048
        arg = w.model.scale_gradients.call_args.args[0]
        assert arg == pytest.approx(1.0 / 2048.0, rel=1e-4)

    def test_picks_global_valid_seqs_for_sequence_level_loss(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.SEQUENCE_LEVEL)
        w.finish_train_step()
        # local_valid_seqs = 8 → inv_n = 1/8
        arg = w.model.scale_gradients.call_args.args[0]
        assert arg == pytest.approx(1.0 / 8.0, rel=1e-4)

    def test_restores_grad_sync_func(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w.finish_train_step()
        assert w.model.config.grad_sync_func == "ORIGINAL_GRAD_SYNC_FUNC"

    def test_clears_train_step_state(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w.finish_train_step()
        assert w._train_step_state is None

    def test_calls_scheduler_step_with_increment_gbs(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w._train_step_state["gbs"] = 64
        w.finish_train_step()
        w.scheduler.step.assert_called_once_with(increment=64)

    def test_returns_metrics_dict(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        metrics = w.finish_train_step()
        for key in (
            "global_loss",
            "rank",
            "gpu_name",
            "model_dtype",
            "all_mb_metrics",
            "grad_norm",
        ):
            assert key in metrics, f"missing {key!r}"

    def test_moe_branch_skipped_when_num_experts_is_none(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w.model.config.num_moe_experts = None
        metrics = w.finish_train_step()
        assert "moe_metrics" not in metrics

    def test_moe_branch_uses_total_num_microbatches_for_scale(
        self, mock_module_symbols
    ):
        """MoE aux-loss scale must use the accumulated total, not the
        per-call num_microbatches."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.model.config.num_moe_experts = 4
        # Have get_moe_metrics return non-empty so the branch fires
        mock_module_symbols["moe"].return_value = {"aux_loss": 0.1}
        w.begin_train_step(loss_fn=w._test_loss_fn)
        # 3 train_microbatch calls × 2 pipeline mbs each = 6
        for _ in range(3):
            w.train_microbatch(_fake_batch())
        w.finish_train_step()
        # get_moe_metrics receives loss_scale=1/6
        kwargs = mock_module_symbols["moe"].call_args.kwargs
        assert kwargs["loss_scale"] == pytest.approx(1.0 / 6.0, rel=1e-6)

    def test_loss_advertised_normalizers_applied(self, mock_module_symbols):
        """finish scales each metric by the denominator the loss advertised:
        TOKENS → 1/global_valid_toks, SEQUENCES → 1/global_valid_seqs,
        NONE → unscaled, unadvertised → gradient normalization (inv_n)."""
        from nemo_rl.algorithms.loss.interfaces import LossType, MetricNormalizer

        w = _make_worker(LossType.TOKEN_LEVEL)
        w._test_loss_fn.metric_normalizations = {
            "tok_metric": MetricNormalizer.TOKENS,
            "seq_metric": MetricNormalizer.SEQUENCES,
            "raw_metric": MetricNormalizer.NONE,
        }
        mock_module_symbols["mfb"].return_value = [
            {
                "tok_metric": 2048.0,
                "seq_metric": 8.0,
                "raw_metric": 8.0,
                "other_metric": 2048.0,
            }
        ]
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())  # 8 seqs / 2048 valid toks
        w.finish_train_step()
        m = mock_module_symbols["agg"].call_args.kwargs["all_mb_metrics"][0]
        assert m["tok_metric"] == pytest.approx(1.0)  # 2048 / 2048
        assert m["seq_metric"] == pytest.approx(1.0)  # 8 / 8
        assert m["raw_metric"] == pytest.approx(8.0)  # unscaled
        # unadvertised → inv_n of the loss_type (TOKEN_LEVEL → 1/2048)
        assert m["other_metric"] == pytest.approx(1.0)

    def test_raw_count_metrics_not_rescaled_by_inv_n(self, mock_module_symbols):
        """Raw-count metrics (num_valid_samples, num_unmasked_tokens) are
        absolute counts the loss advertises as NONE; finish must leave them
        unscaled so the downstream sum recovers the true global count
        (PR #2683 review, F-COUNT)."""
        from nemo_rl.algorithms.loss.interfaces import LossType, MetricNormalizer

        w = _make_worker(LossType.TOKEN_LEVEL)
        w._test_loss_fn.metric_normalizations = {
            "num_valid_samples": MetricNormalizer.NONE,
            "num_unmasked_tokens": MetricNormalizer.NONE,
        }
        mock_module_symbols["mfb"].return_value = [
            {"loss": 0.5, "num_valid_samples": 8.0, "num_unmasked_tokens": 2048.0}
        ]
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())  # inv_n = 1/2048
        w.finish_train_step()
        m = mock_module_symbols["agg"].call_args.kwargs["all_mb_metrics"][0]
        assert m["num_valid_samples"] == pytest.approx(8.0)
        assert m["num_unmasked_tokens"] == pytest.approx(2048.0)

    def test_flag_keyed_normalizers_from_real_loss(self, mock_module_symbols):
        """seq-mask-tis + token-level loss: is_oob_ratio was reduced by
        global_valid_seqs even though the gradient normalizer is tokens.
        The advertised mapping from a real ClippedPGLossFn must key it on
        the TIS type, not loss_type (PR #2683 review, F-SEQ)."""
        from nemo_rl.algorithms.loss.interfaces import LossType
        from nemo_rl.algorithms.loss.loss_functions import (
            ClippedPGLossConfig,
            ClippedPGLossFn,
        )

        real_loss = ClippedPGLossFn(
            ClippedPGLossConfig(
                token_level_loss=True,
                use_importance_sampling_correction=True,
                truncated_importance_sampling_type="seq-mask-tis",
                truncated_importance_sampling_ratio=2.0,
                truncated_importance_sampling_ratio_min=0.5,
            )
        )
        w = _make_worker(LossType.TOKEN_LEVEL)
        w._test_loss_fn.metric_normalizations = real_loss.metric_normalizations
        mock_module_symbols["mfb"].return_value = [
            {
                "loss": 2048.0,
                "is_oob_ratio": 8.0,
                "sampling_importance_ratio": 2048.0,
            }
        ]
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())  # 8 seqs / 2048 valid toks
        w.finish_train_step()
        m = mock_module_symbols["agg"].call_args.kwargs["all_mb_metrics"][0]
        assert m["loss"] == pytest.approx(1.0)  # ÷ toks (loss_type)
        assert m["is_oob_ratio"] == pytest.approx(1.0)  # ÷ seqs, NOT toks
        assert m["sampling_importance_ratio"] == pytest.approx(1.0)  # ÷ toks


# ── abort_train_step ─────────────────────────────────────────────────────


class TestAbort:
    def test_aborted_train_step_clears_graph_route_generation(
        self, mock_module_symbols
    ):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn, gbs=16, mbs=1)
        w._active_router_route_generation = 3

        w.abort_train_step()

        assert w._active_router_route_generation is None

    def test_restores_grad_sync_func(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.abort_train_step()
        assert w.model.config.grad_sync_func == "ORIGINAL_GRAD_SYNC_FUNC"

    def test_zero_grad_buffer_and_zero_grad_called(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.model.zero_grad_buffer.reset_mock()
        w.optimizer.zero_grad.reset_mock()
        w.abort_train_step()
        w.model.zero_grad_buffer.assert_called_once()
        w.optimizer.zero_grad.assert_called_once()

    def test_does_not_call_optimizer_step(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        w.abort_train_step()
        w.optimizer.step.assert_not_called()

    def test_clears_train_step_state(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.abort_train_step()
        assert w._train_step_state is None

    def test_idempotent_with_no_open_step(self, mock_module_symbols):
        """abort is a no-op when nothing is open."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        # Should not raise
        w.abort_train_step()
        assert getattr(w, "_train_step_state", None) is None

    def test_can_begin_new_step_after_abort(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        w.abort_train_step()
        # New step opens cleanly
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w._train_step_state is not None
        assert float(w._train_step_state["local_valid_seqs"].item()) == 0.0


# ── frozen eager/graph R3 parity diagnostic ───────────────────────────


class _ParityStateModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0, -3.0]))
        self.weight.main_grad = torch.tensor([7.0, -11.0])
        self.weight.grad = torch.tensor([13.0, -17.0])
        self.register_buffer("running", torch.tensor([19.0]))
        self.config = SimpleNamespace(
            grad_sync_func="saved-grad-sync",
            no_sync_func="saved-no-sync",
        )
        self.inference_params = {"sentinel": 23}

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * self.weight


def _make_r3_router_graph_parity_worker():
    from nemo_rl.algorithms.loss.interfaces import LossType

    worker = _make_worker(LossType.TOKEN_LEVEL)
    worker.model = _ParityStateModel()
    worker._te_cuda_graph_lifecycle = None
    worker._te_cuda_graph_bank_manager = None
    worker._te_cuda_graph_installed_key = None
    worker._active_router_route_generation = None
    worker._next_router_route_generation = 5
    worker._train_step_state = None
    return worker


def test_r3_router_graph_parity_zeroes_existing_grad_storage_in_place() -> None:
    worker = _make_r3_router_graph_parity_worker()
    main_grad = worker.model.weight.main_grad
    grad = worker.model.weight.grad
    main_grad_ptr = main_grad.data_ptr()
    grad_ptr = grad.data_ptr()

    worker._zero_r3_router_graph_parity_grad_storage()

    assert main_grad.data_ptr() == main_grad_ptr
    assert grad.data_ptr() == grad_ptr
    assert torch.equal(main_grad, torch.zeros_like(main_grad))
    assert torch.equal(grad, torch.zeros_like(grad))


def test_r3_router_graph_parity_full_compare_detects_unsampled_element() -> None:
    worker = _make_r3_router_graph_parity_worker()
    eager = torch.zeros(1025)
    graph = eager.clone()
    graph[1001] = 1.0

    comparison = worker._r3_router_graph_parity_compare_full_tensor(
        eager,
        graph,
        rtol=5e-2,
        atol=5e-2,
        chunk_numel=127,
    )

    assert comparison["numel"] == 1025
    assert comparison["max_abs_diff"] == 1.0
    assert comparison["mismatch_count"] == 1


def test_r3_router_graph_parity_runtime_compare_uses_installed_routes() -> None:
    worker = _make_r3_router_graph_parity_worker()
    eager = [
        {
            "sequence_index": 0,
            "layer_number": 1,
            "payload_index": 0,
            "route_sha256": "a" * 64,
            "shape": [2, 2],
            "dtype": "torch.int64",
            "expert_counts": [1, 1, 1, 1],
            "invalid_expert_count": 0,
            "generation": 5,
        }
    ]
    graph = copy.deepcopy(eager)
    graph[0]["generation"] = 6
    graph[0]["route_sha256"] = "b" * 64

    comparison = worker._r3_router_graph_parity_compare_runtime_routes(eager, graph)

    assert comparison == {"compared_routes": 1, "mismatch_count": 1}


def test_r3_router_graph_parity_captures_installed_route_and_graph_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nemo_rl.models.megatron.train as megatron_train

    worker = _make_r3_router_graph_parity_worker()
    replay = SimpleNamespace(
        _nrl_layer_number=7,
        _nrl_payload_idx=2,
        _nrl_graph_input_signature=SimpleNamespace(num_experts=4),
        target_topk_idx=None,
        graph_input_launch_record=None,
    )
    worker.model.router_replay = replay

    def set_forward(model, routed_experts, **kwargs) -> None:
        del model, kwargs
        replay.target_topk_idx = routed_experts

    def record_consumers(model, **kwargs) -> None:
        del model, kwargs
        replay.graph_input_launch_record = SimpleNamespace(
            copy_generation=77,
            graph_index=3,
        )

    monkeypatch.setattr(megatron_train, "set_router_replay_forward", set_forward)
    monkeypatch.setattr(
        megatron_train,
        "record_router_replay_graph_consumers",
        record_consumers,
    )
    worker._r3_router_graph_parity_route_phase = "measured"
    route = torch.tensor([[0, 1], [1, 3]], dtype=torch.int64)

    with worker._capture_r3_router_graph_parity_runtime_routes() as records:
        megatron_train.set_router_replay_forward(
            worker.model,
            route,
            microbatch_generation=6,
        )
        megatron_train.record_router_replay_graph_consumers(
            worker.model,
            microbatch_generation=6,
            schedule_key=1,
            graph_launch_expected=True,
        )

    assert megatron_train.set_router_replay_forward is set_forward
    assert megatron_train.record_router_replay_graph_consumers is record_consumers
    assert len(records["measured"]) == 1
    evidence = records["measured"][0]
    assert evidence["layer_number"] == 7
    assert evidence["payload_index"] == 2
    assert evidence["expert_counts"] == [1, 2, 0, 1]
    assert evidence["invalid_expert_count"] == 0
    assert len(evidence["route_sha256"]) == 64
    assert evidence["graph_launch"]["successful"] is True
    assert evidence["graph_launch"]["copy_generation"] == 77
    assert evidence["graph_launch"]["graph_index"] == 3


def test_r3_router_graph_parity_arm_restores_rng_buffers_and_grad_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _make_r3_router_graph_parity_worker()
    worker.model.router_replay = SimpleNamespace(
        target_topk_idx=None,
        recorded_topk_idx=None,
        router_replay_action=None,
        replay_backward_list=[],
        graph_input_generation=None,
        _nrl_last_graph_input_generation=3,
        _nrl_last_graph_input_copy_generation=17,
        _nrl_graph_route_counters={"stale_generation_count": 0},
    )
    worker.model.eval()
    worker.model._inference_key_value_memory = {"cache": torch.tensor([31.0])}
    worker.model.config.param_sync_func = "saved-param-sync"
    parameter_before = worker.model.weight.detach().clone()
    buffer_before = worker.model.running.detach().clone()
    main_grad_before = worker.model.weight.main_grad.clone()
    grad_before = worker.model.weight.grad.clone()
    main_grad_ptr = worker.model.weight.main_grad.data_ptr()
    grad_ptr = worker.model.weight.grad.data_ptr()
    cpu_rng_before = torch.get_rng_state().clone()
    cuda_rng_state = [torch.tensor([29], dtype=torch.uint8)]
    restored_cuda_rng: list[torch.Tensor] = []
    optimizer_state = {"state": {0: {"moment": torch.tensor([37.0])}}}
    scheduler_state = {"steps": 41}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: [state.clone() for state in cuda_rng_state],
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda states: restored_cuda_rng.extend(state.clone() for state in states),
    )

    def restore_optimizer_state(saved) -> None:
        optimizer_state.clear()
        optimizer_state.update(saved)

    def restore_scheduler_state(saved) -> None:
        scheduler_state.clear()
        scheduler_state.update(saved)

    def begin_train_step(*, loss_fn, gbs=None, mbs=None) -> None:
        del loss_fn, gbs, mbs
        worker._train_step_state = {
            "mb_losses": [],
            "all_mb_metrics": [],
            "te_cuda_graph_call_state": None,
        }

    def train_microbatch(data) -> None:
        del data
        torch.rand(4)
        worker.model.train()
        worker.model.running.add_(101)
        worker.model.inference_params["sentinel"] = 99
        worker.model._inference_key_value_memory["cache"].zero_()
        worker.model.config.grad_sync_func = "mutated-grad-sync"
        worker.model.config.no_sync_func = "mutated-no-sync"
        worker.model.config.param_sync_func = "mutated-param-sync"
        worker.model.router_replay.replay_backward_list.append(torch.tensor([1]))
        worker.model.router_replay._nrl_last_graph_input_generation = 97
        worker.model.router_replay._nrl_last_graph_input_copy_generation = 101
        worker.model.router_replay._nrl_graph_route_counters[
            "stale_generation_count"
        ] = 1
        optimizer_state["state"][0]["moment"].zero_()
        scheduler_state["steps"] = 99
        worker.model.weight.main_grad.copy_(torch.tensor([0.25, -0.5]))
        worker.model.weight.grad.copy_(torch.tensor([0.75, -1.0]))
        worker._train_step_state["mb_losses"].append(torch.tensor(1.25))
        worker._train_step_state["all_mb_metrics"].append(
            {
                "token_mult_prob_error": torch.tensor(0.01),
                "gen_kl_error": torch.tensor(0.02),
                "policy_kl_error": torch.tensor(0.03),
                "num_valid_samples": torch.tensor(1.0),
            }
        )
        worker._r3_router_graph_parity_capture.update(
            {
                "output": torch.tensor([1.0, 2.0, 3.0]),
                "output_grad": torch.tensor([0.125, 0.25, 0.5]),
                "input": torch.tensor([4.0, 5.0]),
                "input_grad": torch.tensor([0.5, -0.25]),
            }
        )

    abort_calls: list[None] = []

    def abort_train_step() -> None:
        abort_calls.append(None)
        worker._train_step_state = None

    worker.begin_train_step = begin_train_step
    worker.train_microbatch = train_microbatch
    worker.abort_train_step = abort_train_step
    worker.optimizer = MagicMock()
    worker.scheduler = MagicMock()
    worker.optimizer.state_dict.side_effect = lambda: optimizer_state
    worker.optimizer.load_state_dict.side_effect = restore_optimizer_state
    worker.scheduler.state_dict.side_effect = lambda: scheduler_state
    worker.scheduler.load_state_dict.side_effect = restore_scheduler_state
    batch = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "input_lengths": torch.tensor([3]),
        "routed_experts": torch.tensor([[[0, 1], [1, 2], [2, 3]]]),
        "token_mask": torch.tensor([[0.0, 1.0, 1.0]]),
        "sample_mask": torch.tensor([1.0]),
        "rewards": torch.tensor([1.0]),
    }

    result = worker.run_r3_router_graph_parity_arm(
        data=batch,
        loss_fn=worker._test_loss_fn,
        arm="eager",
        simulated_learning_rate=0.1,
    )

    assert result["arm"] == "eager"
    assert result["loss"] == pytest.approx(1.25)
    assert set(result["parameter_gradients"]) == {"weight"}
    assert result["parameter_gradients"]["weight"]["values"] == [0.25, -0.5]
    assert result["simulated_parameter_deltas"]["weight"]["values"] == [
        -0.025,
        0.05,
    ]
    assert len(result["token_digest"]) == 64
    assert len(result["route_digest"]) == 64
    assert result["selected_output"]["values"] == [1.0, 2.0, 3.0]
    assert result["selected_input_gradient"]["values"] == [0.5, -0.25]
    assert abort_calls == [None]
    assert torch.equal(torch.get_rng_state(), cpu_rng_before)
    assert len(restored_cuda_rng) == 1
    assert torch.equal(restored_cuda_rng[0], cuda_rng_state[0])
    assert torch.equal(worker.model.weight, parameter_before)
    assert torch.equal(worker.model.running, buffer_before)
    assert worker.model.training is False
    assert worker.model.inference_params == {"sentinel": 23}
    assert torch.equal(
        worker.model._inference_key_value_memory["cache"], torch.tensor([31.0])
    )
    assert worker.model.config.grad_sync_func == "saved-grad-sync"
    assert worker.model.config.no_sync_func == "saved-no-sync"
    assert worker.model.config.param_sync_func == "saved-param-sync"
    assert worker.model.router_replay.replay_backward_list == []
    assert worker.model.router_replay._nrl_last_graph_input_generation == 3
    assert worker.model.router_replay._nrl_last_graph_input_copy_generation == 17
    assert worker.model.router_replay._nrl_graph_route_counters == {
        "stale_generation_count": 0
    }
    assert torch.equal(optimizer_state["state"][0]["moment"], torch.tensor([37.0]))
    assert scheduler_state == {"steps": 41}
    assert worker.model.weight.main_grad.data_ptr() == main_grad_ptr
    assert worker.model.weight.grad.data_ptr() == grad_ptr
    assert torch.equal(worker.model.weight.main_grad, main_grad_before)
    assert torch.equal(worker.model.weight.grad, grad_before)
    assert worker._train_step_state is None
    worker.optimizer.step.assert_not_called()
    worker.scheduler.step.assert_not_called()


def test_r3_router_graph_parity_arm_aborts_and_restores_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _make_r3_router_graph_parity_worker()
    buffer_before = worker.model.running.clone()
    main_grad_before = worker.model.weight.main_grad.clone()
    cpu_rng_before = torch.get_rng_state().clone()

    def begin_train_step(*, loss_fn, gbs=None, mbs=None) -> None:
        del loss_fn, gbs, mbs
        worker._train_step_state = {"open": True}

    def train_microbatch(data) -> None:
        del data
        torch.rand(2)
        worker.model.running.zero_()
        worker.model.weight.main_grad.zero_()
        raise RuntimeError("diagnostic forward failed")

    abort_calls: list[None] = []

    def abort_train_step() -> None:
        abort_calls.append(None)
        worker._train_step_state = None

    worker.begin_train_step = begin_train_step
    worker.train_microbatch = train_microbatch
    worker.abort_train_step = abort_train_step
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="diagnostic forward failed"):
        worker.run_r3_router_graph_parity_arm(
            data={
                "input_ids": torch.tensor([[1]]),
                "routed_experts": torch.tensor([[[0, 1]]]),
            },
            loss_fn=worker._test_loss_fn,
            arm="eager",
            simulated_learning_rate=0.1,
        )

    assert abort_calls == [None]
    assert torch.equal(torch.get_rng_state(), cpu_rng_before)
    assert torch.equal(worker.model.running, buffer_before)
    assert torch.equal(worker.model.weight.main_grad, main_grad_before)
    assert worker._train_step_state is None


def test_r3_router_graph_parity_restores_parameter_value_after_mutation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _make_r3_router_graph_parity_worker()
    parameter = worker.model.weight
    parameter_before = parameter.detach().clone()
    parameter_identity = id(parameter)

    def begin_train_step(*, loss_fn, gbs=None, mbs=None) -> None:
        del loss_fn, gbs, mbs
        worker._train_step_state = {"open": True}

    def train_microbatch(data) -> None:
        del data
        with torch.no_grad():
            parameter.add_(1000)
        raise RuntimeError("failure after parameter mutation")

    def abort_train_step() -> None:
        worker._train_step_state = None

    worker.begin_train_step = begin_train_step
    worker.train_microbatch = train_microbatch
    worker.abort_train_step = abort_train_step
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="failure after parameter mutation"):
        worker.run_r3_router_graph_parity_arm(
            data={},
            loss_fn=worker._test_loss_fn,
            arm="eager",
            simulated_learning_rate=0.1,
        )

    assert id(worker.model.weight) == parameter_identity
    assert torch.equal(worker.model.weight, parameter_before)
    assert worker._train_step_state is None
    assert worker._r3_router_graph_parity_active is False


def test_r3_router_graph_parity_rejects_and_restores_parameter_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _make_r3_router_graph_parity_worker()
    parameter = worker.model.weight
    parameter_before = parameter.detach().clone()
    parameter_identity = id(parameter)

    def begin_train_step(*, loss_fn, gbs=None, mbs=None) -> None:
        del loss_fn, gbs, mbs
        worker._train_step_state = {"te_cuda_graph_call_state": None}

    def train_microbatch(data) -> None:
        del data
        with torch.no_grad():
            parameter.mul_(17)

    def abort_train_step() -> None:
        worker._train_step_state = None

    worker.begin_train_step = begin_train_step
    worker.train_microbatch = train_microbatch
    worker.abort_train_step = abort_train_step
    worker._collect_r3_router_graph_parity_result = MagicMock(
        return_value={"arm": "eager"}
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="mutated parameters.*weight"):
        worker.run_r3_router_graph_parity_arm(
            data={},
            loss_fn=worker._test_loss_fn,
            arm="eager",
            simulated_learning_rate=0.1,
        )

    assert id(worker.model.weight) == parameter_identity
    assert torch.equal(worker.model.weight, parameter_before)
    assert worker._train_step_state is None
    assert worker._r3_router_graph_parity_active is False


def test_r3_router_graph_parity_capture_and_hit_use_fresh_route_generations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _make_r3_router_graph_parity_worker()
    original_lifecycle = SimpleNamespace(_capacity=1)
    manager = SimpleNamespace(
        active_bank=None,
        _execution_counter=SimpleNamespace(eligible_calls=43, graph_calls=47),
    )

    class OriginalBank:
        def activate(self) -> None:
            manager.active_bank = self

    original_bank = OriginalBank()
    manager.active_bank = original_bank
    worker._te_cuda_graph_lifecycle = original_lifecycle
    worker._te_cuda_graph_bank_manager = manager
    worker._te_cuda_graph_installed_key = "original-key"
    worker.model._nrl_graph_route_counters = {"stale_generation_count": 0}
    observed_generations: list[int] = []

    def begin_train_step(*, loss_fn, gbs=None, mbs=None) -> None:
        del loss_fn, gbs, mbs
        worker._train_step_state = {
            "te_cuda_graph_call_state": SimpleNamespace(capture_count=1)
        }

    def train_microbatch(data) -> None:
        del data
        observed_generations.append(worker._next_router_route_generation)
        worker._next_router_route_generation += 1
        manager._execution_counter.eligible_calls += 1
        manager._execution_counter.graph_calls += 1

    def abort_train_step() -> None:
        worker._train_step_state = None

    worker.begin_train_step = begin_train_step
    worker.train_microbatch = train_microbatch
    worker.abort_train_step = abort_train_step
    worker._collect_r3_router_graph_parity_result = MagicMock(
        return_value={"arm": "graph"}
    )
    worker._finalize_te_cuda_graph_call = MagicMock(
        return_value=(
            {
                "capture_count": 1,
                "replay_count": 0,
                "cache_hit_count": 0,
                "cache_miss_count": 1,
                "eviction_count": 0,
                "fallback_count": 0,
                "graph_calls": 1,
                "eligible_calls": 1,
            },
            {},
        )
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    result = worker.run_r3_router_graph_parity_arm(
        data={},
        loss_fn=worker._test_loss_fn,
        arm="graph",
        simulated_learning_rate=0.1,
    )

    assert result == {"arm": "graph"}
    assert observed_generations == [5, 6]
    assert worker._next_router_route_generation == 5
    assert worker.model._nrl_graph_route_counters["stale_generation_count"] == 0
    assert worker._te_cuda_graph_lifecycle is original_lifecycle
    assert worker._te_cuda_graph_installed_key == "original-key"
    assert manager.active_bank is original_bank
    assert manager._execution_counter.eligible_calls == 43
    assert manager._execution_counter.graph_calls == 47


def test_r3_router_graph_parity_distinguishes_input_and_output_gradients() -> None:
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        _R3ParityLossPostProcessor,
    )

    worker = _make_r3_router_graph_parity_worker()
    capture: dict[str, torch.Tensor] = {}
    handles = worker._install_r3_router_graph_parity_input_hooks(capture)

    class PostProcessor:
        def __call__(self, *args, **kwargs):
            del args, kwargs
            return lambda output: (output.square().sum(), {})

    try:
        hidden = torch.tensor([3.0, 5.0], requires_grad=True)
        output = worker.model(hidden)
        loss, _ = _R3ParityLossPostProcessor(PostProcessor(), capture)()(output)
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    assert capture["input"].data_ptr() == hidden.data_ptr()
    assert torch.equal(capture["input_grad"], torch.tensor([24.0, 90.0]))
    assert torch.equal(capture["output_grad"], torch.tensor([12.0, -30.0]))
    assert not torch.equal(capture["input_grad"], capture["output_grad"])


# ── grad_sync_func full lifecycle (integration of begin → finish/abort) ─


class TestGradSyncFuncLifecycle:
    def test_begin_finish_round_trip(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        sentinel = "MY_CUSTOM_GRAD_SYNC"
        w.model.config.grad_sync_func = sentinel
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w.model.config.grad_sync_func is None
        w.train_microbatch(_fake_batch())
        w.finish_train_step()
        assert w.model.config.grad_sync_func == sentinel

    def test_begin_abort_round_trip(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        sentinel = "MY_CUSTOM_GRAD_SYNC"
        w.model.config.grad_sync_func = sentinel
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w.model.config.grad_sync_func is None
        w.abort_train_step()
        assert w.model.config.grad_sync_func == sentinel

    def test_handles_originally_none_grad_sync_func(self, mock_module_symbols):
        """When PP=1 (or align_grad_reduce=False), grad_sync_func is None
        to begin with. begin → finish must leave it as None."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.model.config.grad_sync_func = None
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w.model.config.grad_sync_func is None
        w.train_microbatch(_fake_batch())
        w.finish_train_step()
        assert w.model.config.grad_sync_func is None
