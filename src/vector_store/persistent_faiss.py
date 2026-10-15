"""Persistent LangChain FAISS collections keyed by selected PDFs and chunk fingerprint."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

logger = logging.getLogger(__name__)


class PersistentFaissStore:
    def __init__(self, root: Path, embedding_model_name: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.embedding_model_name = embedding_model_name

    @staticmethod
    def collection_key(document_ids) -> str:
        canonical = "\n".join(sorted(set(document_ids)))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _fingerprint(chunks) -> str:
        digest = hashlib.sha256()
        for chunk in chunks:
            digest.update(str(chunk.get("chunk_id", "")).encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(chunk.get("text", "")).encode("utf-8"))
            digest.update(b"\0")
            digest.update(json.dumps(chunk.get("metadata", {}), sort_keys=True, ensure_ascii=False).encode("utf-8"))
        return digest.hexdigest()

    def _collection_dir(self, document_ids) -> Path:
        return self.root / self.collection_key(document_ids)

    def _manifest_path(self, document_ids) -> Path:
        return self._collection_dir(document_ids) / "manifest.json"

    @staticmethod
    def to_langchain_documents(chunks) -> list[Document]:
        documents = []
        for chunk in chunks:
            metadata = dict(chunk.get("metadata") or {})
            metadata["chunk_id"] = str(chunk.get("chunk_id", ""))
            documents.append(Document(page_content=str(chunk.get("text", "")), metadata=metadata))
        return documents

    def load(self, document_ids, chunks, embeddings):
        directory = self._collection_dir(document_ids)
        manifest_path = directory / "manifest.json"
        if not (directory / "index.faiss").exists() or not manifest_path.exists():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("embedding_model") != self.embedding_model_name:
                return None
            if manifest.get("document_ids") != sorted(set(document_ids)):
                return None
            if manifest.get("chunk_fingerprint") != self._fingerprint(chunks):
                return None
            # Only app-generated indexes under this local data directory should be loaded.
            return FAISS.load_local(
                str(directory), embeddings, allow_dangerous_deserialization=True,
                distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
            )
        except Exception as exc:
            logger.warning("Could not reuse a persisted FAISS collection; rebuilding it: %s", exc)
            return None

    def save(self, document_ids, chunks, vector_store) -> None:
        directory = self._collection_dir(document_ids)
        directory.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(directory))
        manifest = {
            "embedding_model": self.embedding_model_name,
            "document_ids": sorted(set(document_ids)),
            "chunk_fingerprint": self._fingerprint(chunks),
            "chunk_count": len(chunks),
            "vector_store": "langchain_community.vectorstores.FAISS",
        }
        temporary = directory / "manifest.tmp"
        temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        temporary.replace(directory / "manifest.json")

    def remove_document_indexes(self, document_id: str) -> int:
        removed = 0
        for directory in self.root.iterdir():
            manifest_path = directory / "manifest.json"
            if not directory.is_dir() or not manifest_path.exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if document_id in manifest.get("document_ids", []):
                shutil.rmtree(directory, ignore_errors=True)
                removed += 1
        return removed

    def get_or_build(self, document_ids, chunks, embeddings):
        if not chunks:
            raise ValueError("The selected PDF collection contains no text chunks")
        store = self.load(document_ids, chunks, embeddings)
        if store is not None:
            return store, True
        documents = self.to_langchain_documents(chunks)
        store = FAISS.from_documents(
            documents,
            embeddings,
            normalize_L2=True,
            distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
        )
        self.save(document_ids, chunks, store)
        return store, False
