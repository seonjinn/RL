"""Contracts for the approved New Draft study, separate from the old cohort."""

import importlib.util
import unittest


class StudyTests(unittest.TestCase):
    def module(self):
        name = "research.qwen3_8b_rp25_swa.study"
        self.assertIsNotNone(
            importlib.util.find_spec(name), "New Draft study renderer missing"
        )
        return __import__(name, fromlist=["build_new_arms"])

    def test_eleven_200step_arms(self):
        arms = self.module().build_new_arms()
        self.assertEqual(len(arms), 11)
        self.assertEqual(len({a.name for a in arms}), 11)
        self.assertTrue(all(a.max_steps == 200 for a in arms))
        self.assertTrue(
            all(
                a.global_batch_size == 8 and a.output_sequence_length == 1024
                for a in arms
            )
        )
        self.assertTrue(
            all(
                a.context_parallel_size == 1 and not a.sequence_packing_enabled
                for a in arms
            )
        )

    def test_update_steps(self):
        for arm in self.module().build_new_arms():
            if arm.cadence in ("static", "baseline"):
                self.assertEqual(arm.deterministic_update_steps(), ())
            elif arm.cadence == "always":
                self.assertEqual(len(arm.deterministic_update_steps()), 200)
            else:
                interval = int(arm.cadence.split("-")[1])
                self.assertEqual(
                    arm.deterministic_update_steps(),
                    tuple(range(interval, 201, interval)),
                )

    def test_resume_gate_continues_to_four_with_checkpoint_two_preserved(self):
        study = self.module()
        for arm in study.build_new_arms():
            if arm.cadence != "always":
                with self.assertRaises(ValueError):
                    study.overrides(arm, "/lustre/test", resume_check=True)
                continue
            values = dict(
                x.lstrip("+").split("=", 1)
                for x in study.overrides(arm, "/lustre/test", resume_check=True)
            )
            self.assertEqual(values["grpo.max_num_steps"], "4")
            self.assertEqual(values["checkpointing.save_period"], "2")
            self.assertEqual(
                values["cadence_runtime.required_checkpoint_steps"], "[2,4]"
            )
            self.assertEqual(
                values["checkpointing.checkpoint_dir"], "/lustre/test/checkpoints"
            )
            self.assertEqual(values["policy.draft.update_probe_enabled"], "true")
            self.assertTrue(values["logger.wandb.name"].endswith("-resume-check"))
        with self.assertRaises(ValueError):
            study.overrides(
                study.build_new_arms()[1],
                "/lustre/test",
                canary=True,
                resume_check=True,
            )

    def test_new_checkpoint_contract(self):
        study = self.module()
        for arm in study.build_new_arms():
            values = dict(
                x.lstrip("+").split("=", 1)
                for x in study.overrides(arm, "/lustre/test/newdraft")
            )
            self.assertEqual(values["policy.generation.vllm_kwargs.max_num_seqs"], "8")
            if arm.drafter == "none":
                self.assertEqual(values["policy.draft.enabled"], "false")
                self.assertEqual(
                    values["policy.generation.vllm_kwargs.speculative_config"], "null"
                )
                continue
            self.assertIn(
                f"sd2p3rp-q8b-base-ptv3rp25-{arm.drafter}-b8-16n/exported-checkpoint-44000",
                values["policy.draft.model_name"],
            )
            self.assertEqual(values["policy.draft.model_revision"], "null")
            self.assertEqual(
                values["policy.generation.vllm_kwargs.speculative_config.revision"],
                "null",
            )
            self.assertEqual(values["policy.draft.sliding_window"], "2048")
            self.assertEqual(
                values[
                    "policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens"
                ],
                "5",
            )
            key = "gamma" if arm.drafter == "dflash" else "block_size"
            self.assertEqual(
                values[f"policy.draft.{key}"], "7" if key == "gamma" else "8"
            )


if __name__ == "__main__":
    unittest.main()
