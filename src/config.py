"""Environment-backed configuration for the LangChain PDF RAG app."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc


@dataclass(frozen=True)
class Settings:
    app_title: str = os.getenv("APP_TITLE", "PDF RAG Chatbot — LangChain")
    data_dir: Path = Path(os.getenv("RAG_DATA_DIR", "data"))
    memory_db: Path = Path(os.getenv("MEMORY_DB_PATH", "data/chat_history.db"))
    documents_dir: Path = Path(os.getenv("DOCUMENTS_DIR", "data/documents"))
    indexes_dir: Path = Path(os.getenv("INDEXES_DIR", "data/indexes"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_site_url: str = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501")
    openrouter_app_name: str = os.getenv("OPENROUTER_APP_NAME", "PDF RAG Chatbot")
    chunk_size: int = _env_int("CHUNK_SIZE", 700)
    chunk_overlap: int = _env_int("CHUNK_OVERLAP", 100)
    retrieval_k: int = _env_int("RETRIEVAL_K", 6)
    llm_context_k: int = _env_int("LLM_CONTEXT_K", 3)
    retrieval_mode: str = os.getenv("RETRIEVAL_MODE", "dense").strip().lower()
    recent_turns: int = _env_int("HISTORY_MAX_TURNS", 5)
    history_token_budget: int = _env_int("HISTORY_TOKEN_BUDGET", 350)
    summary_trigger_turns: int = _env_int("SUMMARY_TRIGGER_TURNS", 5)
    summary_token_limit: int = _env_int("SUMMARY_TOKEN_LIMIT", 250)
    context_limit: int = _env_int("CONTEXT_LIMIT", 4096)
    max_output_tokens: int = _env_int("MAX_OUTPUT_TOKENS", 800)
    safety_margin: int = _env_int("SAFETY_MARGIN", 50)
    llm_temperature: float = _env_float("LLM_TEMPERATURE", 0.1)
    request_timeout_seconds: int = _env_int("REQUEST_TIMEOUT_SECONDS", 60)
    llm_max_retries: int = _env_int("LLM_MAX_RETRIES", 2)
    enable_reranker: bool = os.getenv("ENABLE_RERANKER", "true").lower() in {"true", "1", "yes"}

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
        if self.retrieval_k < 1 or self.llm_context_k < 1:
            raise ValueError("RETRIEVAL_K and LLM_CONTEXT_K must be positive")
        if self.retrieval_mode not in {"dense", "hybrid"}:
            raise ValueError("RETRIEVAL_MODE must be either dense or hybrid")
        if self.recent_turns < 1 or self.history_token_budget < 50 or self.summary_token_limit < 50:
            raise ValueError("Memory limits are too small")
        if self.context_limit <= self.max_output_tokens + self.safety_margin:
            raise ValueError("CONTEXT_LIMIT must exceed MAX_OUTPUT_TOKENS + SAFETY_MARGIN")
        if self.request_timeout_seconds <= 0 or self.llm_max_retries < 0:
            raise ValueError("Timeout must be positive and retry count cannot be negative")
        for path in (self.data_dir, self.documents_dir, self.indexes_dir, self.memory_db.parent):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.validate()
