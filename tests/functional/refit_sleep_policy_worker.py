# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Megatron test extension: deterministic updates without changing the recipe."""

import ray
import torch

from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
    get_runtime_env_for_policy_worker,
)
from tests.functional.refit_sleep_utils import scale_parameters


@ray.remote(runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"))
class RefitSleepPolicyWorker(MegatronPolicyWorkerImpl):
    @torch.no_grad()
    def apply_refit_test_update(self, factor: float) -> int:
        self.sync_params_before_refit()
        changed = scale_parameters(self.model.parameters(), factor=factor)
        # Keep optimizer master shards consistent with the deliberately changed
        # model so a later materialization cannot restore the previous state.
        self.optimizer.reload_model_params()
        torch.cuda.synchronize()
        return changed
