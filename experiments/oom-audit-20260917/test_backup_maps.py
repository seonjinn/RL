import unittest

from backup_maps import backup_mappings


class BackupMappingTests(unittest.TestCase):
    def test_matches_overlapping_ranges_once_and_preserves_mapping_counters(
        self,
    ) -> None:
        smaps = (
            "1000-2000 rw-s 00000000 00:01 123 /memfd:backup (deleted)\n"
            "Rss: 4 kB\nPss: 2 kB\nAnonymous: 0 kB\n"
            "2000-3000 rw-p 00000000 00:00 0\n"
            "Rss: 4 kB\nPss: 4 kB\nAnonymous: 4 kB\n"
            "4000-5000 rw-p 00000000 00:00 0\nRss: 4 kB\n"
        )
        rows = backup_mappings(smaps, [(0x1800, 4096), (0x1900, 32)])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["Rss"], 4096)
        self.assertEqual(rows[0]["Pss"], 2048)
        self.assertEqual(rows[1]["Anonymous"], 4096)


if __name__ == "__main__":
    unittest.main()
