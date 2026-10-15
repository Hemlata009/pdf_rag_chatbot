"""Application service: document collections, LCEL chains, and persistent chat turns."""
from __future__ import annotations

from typing import Any

from src.langchain_rag import build_conversational_rag_chain, build_summary_chain


class ChatService:
    def __init__(self, settings, memory, document_store, vector_store, embeddings, llm):
        self.settings = settings
        self.memory = memory
        self.document_store = document_store
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.llm = llm
        self.summary_chain = build_summary_chain(llm)
        self._collection_cache: dict[tuple[str, ...], tuple[list[dict], Any, bool]] = {}

    def get_collection(self, document_ids):
        """Load or build a persistent LangChain FAISS store for this PDF selection."""
        key = tuple(sorted(set(document_ids)))
        if not key:
            raise ValueError("Select at least one PDF")
        chunks = self.document_store.load_chunks(key)
        if not chunks:
            raise ValueError("The selected PDF collection has no indexed text")
        vector_index, reused = self.vector_store.get_or_build(key, chunks, self.embeddings)
        self._collection_cache[key] = (chunks, vector_index, reused)
        return chunks, vector_index, reused

    def answer(self, session_id: str, question: str, document_ids, chunks, vector_index, reranker=None):
        if not question or not question.strip():
            raise ValueError("Please enter a question")
        self.memory.ensure_session(session_id)
        chain = build_conversational_rag_chain(
            settings=self.settings,
            memory=self.memory,
            vector_store=vector_index,
            chunks=chunks,
            llm=self.llm,
            reranker=reranker if self.settings.enable_reranker else None,
        )
        result = chain.invoke(
            {"question": question.strip(), "session_id": session_id},
            config={"configurable": {"session_id": session_id}},
        )
        answer_text = str(result.get("answer", "")).strip()
        self.memory.maybe_update_summary(session_id, self._summarize)

        used_documents = result.get("used_documents") or result.get("ranked_documents") or []
        sources = []
        for document in used_documents:
            metadata = dict(getattr(document, "metadata", {}) or {})
            sources.append({
                "text": document.page_content,
                "metadata": metadata,
                "rerank_score": metadata.get("rerank_score"),
            })
        trace = {
            "retrieval_mode": self.settings.retrieval_mode,
            "retrieval_query": result.get("retrieval_query", question),
            "candidate_count": len(result.get("candidate_documents") or []),
            "reranked_count": len(result.get("ranked_documents") or []),
            "context_count": len(used_documents),
            "context_token_estimate": result.get("context_token_estimate", 0),
            "memory_summary_used": bool(result.get("memory_summary")),
            "reranker_enabled": bool(reranker and self.settings.enable_reranker),
            "selected_document_count": len(document_ids),
        }
        return answer_text, sources, trace

    def _summarize(self, old_summary: str, old_messages: str) -> str:
        return self.summary_chain.invoke({
            "old_summary": old_summary or "(none)",
            "old_messages": old_messages,
        })
