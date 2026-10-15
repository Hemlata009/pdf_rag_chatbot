"""Content-addressed PDF storage using LangChain loaders and text splitting."""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any


class DocumentStore:
    def __init__(self, root: Path, chunk_size: int = 700, overlap: int = 100):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.json"
        self.chunk_size, self.overlap = chunk_size, overlap
        if not self.registry_path.exists():
            self._write_registry({})

    def _read_registry(self) -> dict:
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_registry(self, data: dict) -> None:
        temporary = self.registry_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.registry_path)

    @staticmethod
    def safe_name(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._") or "document.pdf"

    def list_documents(self) -> list[dict]:
        return [{"document_id": key, **value} for key, value in self._read_registry().items()]

    def ingest_bytes(self, filename: str, content: bytes):
        """Extract, split, and persist a PDF. Duplicate file content is indexed once."""
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported")
        if not content:
            raise ValueError("The uploaded PDF is empty")
        if len(content) > 50 * 1024 * 1024:
            raise ValueError("PDF exceeds the 50 MB upload limit")

        digest = hashlib.sha256(content).hexdigest()
        registry = self._read_registry()
        if digest in registry:
            return digest, False, registry[digest]

        # Lazy imports keep storage metadata tests lightweight; ingestion uses LangChain components.
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        safe_filename = self.safe_name(filename)
        try:
            with tempfile.TemporaryDirectory(prefix="lc_pdf_ingest_") as temporary_dir:
                temp_path = Path(temporary_dir) / safe_filename
                temp_path.write_bytes(content)
                page_documents = PyPDFLoader(str(temp_path)).load()
        except Exception as exc:
            raise ValueError(f"Could not read this PDF: {exc}") from exc
        if not page_documents:
            raise ValueError("This PDF contains no pages")
        extractable_pages = [doc for doc in page_documents if doc.page_content.strip()]
        if not extractable_pages:
            raise ValueError("No extractable text was found. Scanned PDFs need OCR before ingestion.")

        for page_doc in extractable_pages:
            page_number = int(page_doc.metadata.get("page", 0)) + 1
            page_doc.metadata = {
                "source": safe_filename,
                "page": page_number,
                "document_id": digest,
            }

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
            add_start_index=True,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        split_docs = splitter.split_documents(extractable_pages)
        chunks: list[dict[str, Any]] = []
        per_page_counter: dict[int, int] = {}
        for doc in split_docs:
            page_number = int(doc.metadata.get("page", 1))
            per_page_counter[page_number] = per_page_counter.get(page_number, 0) + 1
            chunk_index = per_page_counter[page_number]
            chunks.append({
                "chunk_id": f"{safe_filename}::p{page_number}::c{chunk_index}",
                "text": doc.page_content,
                "metadata": {
                    "source": safe_filename,
                    "page": page_number,
                    "document_id": digest,
                    "start_index": int(doc.metadata.get("start_index", 0)),
                },
            })
        if not chunks:
            raise ValueError("No chunks were produced from this PDF")

        document_dir = self.root / digest
        document_dir.mkdir(parents=True, exist_ok=True)
        (document_dir / "original.pdf").write_bytes(content)
        chunks_path = document_dir / "chunks.json"
        chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        record = {
            "filename": safe_filename,
            "sha256": digest,
            "pages": len(page_documents),
            "extractable_pages": len(extractable_pages),
            "chunk_count": len(chunks),
            "chunks_path": f"{digest}/chunks.json",
        }
        registry[digest] = record
        self._write_registry(registry)
        return digest, True, record

    def load_chunks(self, document_ids=None) -> list[dict]:
        registry = self._read_registry()
        ids = list(document_ids) if document_ids is not None else list(registry)
        chunks = []
        for document_id in ids:
            if document_id not in registry:
                continue
            path = self.root / registry[document_id]["chunks_path"]
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Stored chunks for {registry[document_id]['filename']} are unavailable or corrupt") from exc
            for chunk in payload:
                chunk["metadata"]["document_id"] = document_id
                chunk["chunk_id"] = f"{document_id[:12]}::{chunk.get('chunk_id', 'chunk')}"
                chunks.append(chunk)
        return chunks

    def remove_document(self, document_id: str) -> bool:
        registry = self._read_registry()
        if document_id not in registry:
            return False
        registry.pop(document_id)
        self._write_registry(registry)
        import shutil
        shutil.rmtree(self.root / document_id, ignore_errors=True)
        return True
