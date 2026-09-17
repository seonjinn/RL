import unittest
from types import SimpleNamespace

from backup_mode import pageable_backups


class TestBackupMode(unittest.TestCase):
    def test_restores_flag_after_error(self) -> None:
        module = SimpleNamespace(PIN_MEMORY=True)
        with self.assertRaises(RuntimeError):
            with pageable_backups(module, True):
                self.assertFalse(module.PIN_MEMORY)
                raise RuntimeError("test")
        self.assertTrue(module.PIN_MEMORY)

    def test_control_preserves_original(self) -> None:
        module = SimpleNamespace(PIN_MEMORY=True)
        with pageable_backups(module, False):
            self.assertTrue(module.PIN_MEMORY)
