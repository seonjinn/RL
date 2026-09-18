"""Run with the policy interpreter on a cluster GPU, not a laptop."""

import unittest

import torch

from reference_snapshot import snapshot_policy_state


class ReferenceSnapshotTests(unittest.TestCase):
    def test_refreshes_stale_backup_and_restores_noncontiguous_views(self) -> None:
        source = torch.arange(24, dtype=torch.bfloat16, device="cuda")
        backup = torch.full((24,), -10, dtype=torch.bfloat16, pin_memory=True)
        view = source[4:16].reshape(3, 4).transpose(0, 1)
        uncovered = torch.tensor([31.0], device="cuda")
        original = view.cpu().clone()
        state = {"weight": view, "buffer": uncovered, "extra": None}
        saved, reused = snapshot_policy_state(state, [(source, backup)])
        self.assertEqual(reused, 24)
        self.assertEqual(saved["weight"].untyped_storage().data_ptr(), backup.untyped_storage().data_ptr())
        torch.testing.assert_close(saved["weight"], original, rtol=0, atol=0)
        torch.testing.assert_close(backup, torch.arange(24, dtype=torch.bfloat16), rtol=0, atol=0)
        view.fill_(77)
        uncovered.fill_(99)
        view.copy_(saved["weight"])
        uncovered.copy_(saved["buffer"])
        torch.testing.assert_close(view.cpu(), original, rtol=0, atol=0)
        self.assertEqual(uncovered.item(), 31)
        self.assertIsNone(saved["extra"])
        source.add_(2)
        saved_again, _ = snapshot_policy_state(state, [(source, backup)])
        torch.testing.assert_close(saved_again["weight"], original + 2, rtol=0, atol=0)

    def test_rejects_incompatible_backup_before_mutating_it(self) -> None:
        source = torch.zeros(8, dtype=torch.bfloat16, device="cuda")
        backup = torch.full((7,), 9, dtype=torch.bfloat16, pin_memory=True)
        with self.assertRaises(ValueError):
            snapshot_policy_state({"weight": source}, [(source, backup)])
        self.assertTrue(bool(torch.all(backup == 9)))


if __name__ == "__main__":
    unittest.main()
