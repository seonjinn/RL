"""Unit tests for the init-only accounting experiment; no GPU imports."""

import unittest

from cumem_accounting import unmapped_idle_bytes


class AccountingTests(unittest.TestCase):
    def test_counts_only_missing_idle_segments_on_target_device(self) -> None:
        segments = [
            dict(device=0, address=1, allocated_size=0, active_size=0, total_size=54),
            dict(device=0, address=2, allocated_size=64, active_size=64, total_size=65),
            dict(device=1, address=3, allocated_size=1, active_size=1, total_size=200),
        ]
        self.assertEqual(unmapped_idle_bytes(segments, {2}, 0, 119), 54)

    def test_rejects_missing_active_segment(self) -> None:
        segment = dict(device=0, address=1, allocated_size=0, active_size=1, total_size=54)
        with self.assertRaises(ValueError):
            unmapped_idle_bytes([segment], set(), 0, 119)

    def test_rejects_correction_larger_than_reserved(self) -> None:
        segment = dict(device=0, address=1, allocated_size=0, active_size=0, total_size=54)
        with self.assertRaises(ValueError):
            unmapped_idle_bytes([segment], set(), 0, 53)


if __name__ == "__main__":
    unittest.main()
