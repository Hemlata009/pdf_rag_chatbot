import tempfile
import unittest
from pathlib import Path
from src.config import Settings

class SettingsTests(unittest.TestCase):
    def test_valid_settings_create_runtime_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Settings(data_dir=root / "data", memory_db=root / "data" / "memory.db",
                           documents_dir=root / "data" / "docs", indexes_dir=root / "data" / "indexes")
            cfg.validate()
            self.assertTrue(cfg.documents_dir.is_dir())
            self.assertTrue(cfg.indexes_dir.is_dir())

    def test_retrieval_mode_must_be_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Settings(data_dir=root / "data", memory_db=root / "data" / "memory.db",
                           documents_dir=root / "data" / "docs", indexes_dir=root / "data" / "indexes",
                           retrieval_mode="bm25-only")
            with self.assertRaises(ValueError):
                cfg.validate()

    def test_overlap_must_be_smaller_than_chunk_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = Settings(data_dir=root / "data", memory_db=root / "data" / "memory.db",
                           documents_dir=root / "data" / "docs", indexes_dir=root / "data" / "indexes",
                           chunk_size=100, chunk_overlap=100)
            with self.assertRaises(ValueError):
                cfg.validate()

if __name__ == "__main__":
    unittest.main()
