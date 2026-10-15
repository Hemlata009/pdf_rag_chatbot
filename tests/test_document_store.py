import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import hashlib
from src.ingestion.document_store import DocumentStore

class DocumentStoreTests(unittest.TestCase):
    def test_rejects_non_pdf_and_empty_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DocumentStore(Path(tmp))
            with self.assertRaises(ValueError):
                store.ingest_bytes("notes.txt", b"text")
            with self.assertRaises(ValueError):
                store.ingest_bytes("empty.pdf", b"")

    def test_duplicate_content_hash_is_not_reindexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = DocumentStore(Path(tmp))
            record = {"filename": "report.pdf", "sha256": "same", "pages": 1,
                      "extractable_pages": 1, "chunk_count": 1, "chunks_path": "same/chunks.json"}
            store._write_registry({"same": record})
            with patch("src.ingestion.document_store.hashlib.sha256") as mock_hash:
                mock_hash.return_value.hexdigest.return_value = "same"
                key, created, loaded = store.ingest_bytes("another-name.pdf", b"anything")
            self.assertEqual(key, "same")
            self.assertFalse(created)
            self.assertEqual(loaded["filename"], "report.pdf")

if __name__ == "__main__":
    unittest.main()
