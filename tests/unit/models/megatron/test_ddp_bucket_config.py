import unittest
from unittest.mock import MagicMock, patch

import torch


class TestCreateMegatronConfig(unittest.TestCase):
    def test_propagates_ddp_bucket_tuning(self):
        from nemo_rl.models.megatron.setup import _create_megatron_config

        config = {
            "train_global_batch_size": 64,
            "megatron_cfg": {
                "train_iters": 20,
                "optimizer": {"use_distributed_optimizer": True},
                "distributed_data_parallel_config": {
                    "grad_reduce_in_fp32": True,
                    "overlap_grad_reduce": True,
                    "overlap_param_gather": True,
                    "data_parallel_sharding_strategy": "optim_grads_params",
                    "bucket_size": 499400281,
                    "pad_buckets_for_high_nccl_busbw": True,
                },
            },
        }

        with (
            patch("nemo_rl.models.megatron.setup.ConfigContainer"),
            patch("nemo_rl.models.megatron.setup.DistributedDataParallelConfig") as mock_ddp,
            patch("nemo_rl.models.megatron.setup.LoggerConfig"),
            patch("nemo_rl.models.megatron.setup.OptimizerConfig"),
            patch("nemo_rl.models.megatron.setup.SchedulerConfig"),
            patch("nemo_rl.models.megatron.setup.TokenizerConfig"),
            patch("nemo_rl.models.megatron.setup.TrainingConfig"),
        ):
            _create_megatron_config(
                model_cfg=MagicMock(),
                checkpoint_config=MagicMock(),
                config=config,
                hf_model_name="test-model",
                dtype=torch.bfloat16,
            )

        ddp_kwargs = mock_ddp.call_args.kwargs
        self.assertEqual(ddp_kwargs["bucket_size"], 499400281)
        self.assertIs(ddp_kwargs["pad_buckets_for_high_nccl_busbw"], True)


if __name__ == "__main__":
    unittest.main()
