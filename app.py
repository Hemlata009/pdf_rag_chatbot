"""Streamlit UI for the LangChain conversational PDF RAG application."""
from __future__ import annotations

import uuid
import streamlit as st

from src.config import settings
from src.conversation_memory import ConversationMemory
from src.ingestion.document_store import DocumentStore
from src.langchain_rag import create_chat_model, create_embeddings, create_reranker
from src.services.chat_service import ChatService
from src.vector_store.persistent_faiss import PersistentFaissStore

st.set_page_config(page_title=settings.app_title, page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading LangChain embedding model…")
def get_embeddings(model_name: str):
    return create_embeddings(model_name)


@st.cache_resource(show_spinner="Loading OpenRouter chat model…")
def get_chat_model():
    return create_chat_model(settings)


@st.cache_resource(show_spinner="Loading CrossEncoder reranker…")
def get_reranker(model_name: str):
    return create_reranker(model_name)


@st.cache_resource
def get_memory():
    return ConversationMemory(
        str(settings.memory_db),
        recent_turns=settings.recent_turns,
        summary_trigger_turns=settings.summary_trigger_turns,
        summary_token_limit=settings.summary_token_limit,
    )


@st.cache_resource
def get_document_store():
    return DocumentStore(settings.documents_dir, settings.chunk_size, settings.chunk_overlap)


@st.cache_resource
def get_vector_store():
    return PersistentFaissStore(settings.indexes_dir, settings.embedding_model)


@st.cache_resource(show_spinner="Preparing LangChain services…")
def get_chat_service():
    # Fail fast on a missing API key before loading large local embedding models.
    llm = get_chat_model()
    embeddings = get_embeddings(settings.embedding_model)
    return ChatService(settings, get_memory(), get_document_store(), get_vector_store(), embeddings, llm)


st.title("📄 PDF RAG Chatbot")
st.caption(
    "LangChain LCEL · ChatOpenAI/OpenRouter · PyPDFLoader · RecursiveCharacterTextSplitter · "
    "HuggingFace embeddings · FAISS retriever · optional BM25/RRF · CrossEncoder reranking · SQLite chat history"
)

# Initialize local storage separately so PDFs can be ingested before the first model call.
memory = get_memory()
document_store = get_document_store()
vector_store = get_vector_store()

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
if "active_documents" not in st.session_state:
    st.session_state.active_documents = []

with st.sidebar:
    st.header("Conversation")
    session_id = st.text_input(
        "Conversation ID (use it to resume)",
        value=st.session_state.session_id,
        max_chars=128,
        help="A conversation ID selects one SQLite-backed conversation. Treat it like a private identifier.",
    ).strip()
    if not session_id:
        st.warning("Conversation ID cannot be empty.")
        st.stop()
    st.session_state.session_id = session_id
    memory.ensure_session(session_id)
    st.caption("Messages are stored in SQLite. Only bounded recent history and a rolling summary are sent to the model.")
    if st.button("Clear this conversation", use_container_width=True):
        memory.clear_session(session_id)
        st.rerun()

    st.divider()
    st.header("Add PDFs")
    uploads = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Maximum 50 MB per PDF. Text-based PDFs are supported; scanned PDFs require OCR before ingestion.",
    )
    if st.button("Ingest uploaded PDFs", type="primary", disabled=not uploads, use_container_width=True):
        succeeded, failed = 0, []
        progress = st.progress(0)
        for index, uploaded in enumerate(uploads, start=1):
            try:
                _, created, record = document_store.ingest_bytes(uploaded.name, uploaded.getvalue())
                succeeded += 1
                if created:
                    st.toast(f"Indexed {record['filename']} ({record['chunk_count']} chunks)")
                else:
                    st.toast(f"Already indexed: {record['filename']}")
            except Exception as exc:
                failed.append(f"{uploaded.name}: {exc}")
            progress.progress(index / len(uploads))
        if succeeded:
            st.success(f"Processed {succeeded} PDF(s).")
        for error in failed:
            st.error(error)
        st.rerun()

records = document_store.list_documents()
with st.sidebar:
    st.divider()
    st.header("Document collection")
    if records:
        options = {
            f"{record['filename']} · {record['chunk_count']} chunks · {record['pages']} pages · {record['document_id'][:8]}": record["document_id"]
            for record in records
        }
        default_labels = [label for label, doc_id in options.items() if doc_id in st.session_state.active_documents]
        selected_labels = st.multiselect(
            "Search within",
            list(options.keys()),
            default=default_labels or list(options.keys()),
        )
        selected_ids = [options[label] for label in selected_labels]
        st.session_state.active_documents = selected_ids
        if st.button("Remove selected PDFs", disabled=not selected_ids, use_container_width=True):
            for document_id in selected_ids:
                document_store.remove_document(document_id)
                vector_store.remove_document_indexes(document_id)
            st.session_state.active_documents = []
            st.rerun()
        st.caption("Collections are persisted and reused when the documents and embedding model match.")
    else:
        selected_ids = []

if not records:
    st.info("Upload at least one text-based PDF from the sidebar to begin.")
    st.stop()
if not selected_ids:
    st.warning("Select at least one PDF in the sidebar.")
    st.stop()

try:
    with st.spinner("Preparing the selected PDF collection…"):
        service = get_chat_service()
        chunks, collection_index, reused = service.get_collection(selected_ids)
    st.caption(
        f"Collection: {len(selected_ids)} PDF(s) · {len(chunks)} chunks · "
        f"LangChain FAISS index {'reused' if reused else 'built/updated'}"
    )
except Exception as exc:
    st.error(f"Could not prepare the LangChain document collection: {exc}")
    st.stop()

st.subheader("Chat with your documents")
for message in memory.get_full_messages(session_id):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the selected PDFs…")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            reranker = None
            if settings.enable_reranker:
                with st.spinner("Loading the reranker on first use…"):
                    reranker = get_reranker(settings.reranker_model)
            with st.spinner("Rewriting the query if needed, retrieving evidence, and generating an answer…"):
                answer, sources, trace = service.answer(
                    session_id=session_id,
                    question=question,
                    document_ids=selected_ids,
                    chunks=chunks,
                    vector_index=collection_index,
                    reranker=reranker,
                )
            st.markdown(answer)
            if sources:
                with st.expander(f"Sources used ({len(sources)})"):
                    for index, source in enumerate(sources, start=1):
                        metadata = source.get("metadata", {})
                        st.markdown(
                            f"**{index}. {metadata.get('source', 'unknown')} — page {metadata.get('page', 'unknown')}**"
                        )
                        if source.get("rerank_score") is not None:
                            st.caption(f"CrossEncoder score: {source['rerank_score']:.4f}")
                        st.write(source.get("text", ""))
            if trace:
                with st.expander("Request diagnostics"):
                    st.json(trace)
        except Exception as exc:
            st.error(f"Unable to answer this turn: {exc}")
