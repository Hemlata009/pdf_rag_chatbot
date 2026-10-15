"""Terminal entry point using the same LangChain pipeline as the Streamlit UI."""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

from src.config import settings
from src.conversation_memory import ConversationMemory
from src.ingestion.document_store import DocumentStore
from src.langchain_rag import create_chat_model, create_embeddings, create_reranker
from src.services.chat_service import ChatService
from src.vector_store.persistent_faiss import PersistentFaissStore

DEFAULT_PDF = "sample_docs/PBI_midterm_report.pdf"
CHAT_LOG_DIR = Path("chat_logs")


def print_banner() -> None:
    print("\n" + "=" * 76)
    print("              PDF RAG CHATBOT — LANGCHAIN")
    print("=" * 76)
    print("PyPDFLoader → RecursiveCharacterTextSplitter → HuggingFace embeddings")
    print("FAISS retriever → optional BM25/RRF → CrossEncoder → LCEL answer chain")
    print("RunnableWithMessageHistory → SQLite persistent conversation memory")
    print("=" * 76)


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain terminal PDF RAG chatbot")
    parser.add_argument("pdf", nargs="?", default=DEFAULT_PDF, help=f"PDF file (default: {DEFAULT_PDF})")
    parser.add_argument("--session-id", default=None, help="Conversation ID to resume an existing SQLite history")
    parser.add_argument("--no-reranker", action="store_true", help="Disable CrossEncoder reranking for this run")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        raise SystemExit(1)

    try:
        print_banner()
        print("Loading LangChain/OpenRouter model…")
        llm = create_chat_model(settings)
        print("Loading Sentence Transformers embeddings…")
        embeddings = create_embeddings(settings.embedding_model)
        reranker = None
        if settings.enable_reranker and not args.no_reranker:
            print("Loading CrossEncoder reranker…")
            reranker = create_reranker(settings.reranker_model)

        memory = ConversationMemory(
            str(settings.memory_db),
            recent_turns=settings.recent_turns,
            summary_trigger_turns=settings.summary_trigger_turns,
            summary_token_limit=settings.summary_token_limit,
        )
        document_store = DocumentStore(settings.documents_dir, settings.chunk_size, settings.chunk_overlap)
        vector_store = PersistentFaissStore(settings.indexes_dir, settings.embedding_model)

        print(f"Ingesting/reusing PDF: {pdf_path.name}")
        document_id, _, record = document_store.ingest_bytes(pdf_path.name, pdf_path.read_bytes())
        service = ChatService(settings, memory, document_store, vector_store, embeddings, llm)
        chunks, collection_index, reused = service.get_collection([document_id])

        session_id = (args.session_id or uuid.uuid4().hex).strip()
        memory.ensure_session(session_id)
        print(f"Document: {record['filename']} | pages: {record['pages']} | chunks: {len(chunks)}")
        print(f"FAISS collection: {'reused' if reused else 'built'}")
        print(f"Conversation ID: {session_id}")
        print("Enter a question; type 'clear' to clear this conversation or 'exit' to stop.")

        CHAT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = CHAT_LOG_DIR / f"chat_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        while True:
            try:
                question = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit", "q"}:
                print("Goodbye!")
                break
            if question.lower() == "clear":
                memory.clear_session(session_id)
                print("Conversation history and summary cleared.")
                continue

            try:
                answer, sources, trace = service.answer(
                    session_id=session_id,
                    question=question,
                    document_ids=[document_id],
                    chunks=chunks,
                    vector_index=collection_index,
                    reranker=reranker,
                )
                print("\nAssistant:\n" + answer)
                if sources:
                    print("\nSources:")
                    for rank, source in enumerate(sources, start=1):
                        metadata = source.get("metadata", {})
                        print(f"  {rank}. {metadata.get('source', 'unknown')} — page {metadata.get('page', '?')}")
                        if source.get("rerank_score") is not None:
                            print(f"     CrossEncoder score: {source['rerank_score']:.4f}")
                print(f"\nDiagnostics: {json.dumps(trace, ensure_ascii=False)}")
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "session_id": session_id,
                        "question": question,
                        "answer": answer,
                        "sources": [item.get("metadata", {}) for item in sources],
                        "trace": trace,
                    }, ensure_ascii=False, default=str) + "\n")
            except Exception as exc:
                print(f"\nERROR: {exc}", file=sys.stderr)

    except Exception as exc:
        print(f"\nERROR while starting the chatbot: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
