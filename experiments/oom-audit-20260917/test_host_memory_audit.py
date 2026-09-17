import unittest

from nemo_rl.utils.host_memory_audit import (
    StorageRecord,
    parse_kib_fields,
    summarize_storages,
)


class StorageAccountingTest(unittest.TestCase):
    def test_views_are_counted_once_across_categories(self) -> None:
        storage = StorageRecord("cpu", 100, 4096, True)
        result = summarize_storages(
            {"reference": [storage, storage], "backup": [storage]}
        )
        self.assertEqual(result["reference"]["cpu_unique_bytes"], 4096)
        self.assertEqual(result["reference"]["pinned_unique_bytes"], 4096)
        self.assertEqual(result["backup"]["cpu_unique_bytes"], 0)
        self.assertEqual(result["backup"]["shared_bytes"], 4096)

    def test_cuda_address_does_not_alias_cpu(self) -> None:
        result = summarize_storages(
            {
                "optimizer": [
                    StorageRecord("cpu", 100, 1024, False),
                    StorageRecord("cuda:0", 100, 2048, False),
                    StorageRecord("cpu", 0, 0, False),
                ]
            }
        )
        self.assertEqual(result["optimizer"]["cpu_unique_bytes"], 1024)
        self.assertEqual(result["optimizer"]["cuda_unique_bytes"], 2048)
        self.assertEqual(result["optimizer"]["pinned_unique_bytes"], 0)

    def test_proc_kib_conversion_ignores_non_numeric_lines(self) -> None:
        self.assertEqual(
            parse_kib_fields("header\nRss: 10 kB\nPss: 7 kB\n"),
            {"Rss": 10240, "Pss": 7168},
        )


if __name__ == "__main__":
    unittest.main()
