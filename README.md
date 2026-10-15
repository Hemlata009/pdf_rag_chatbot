# PDF RAG Chatbot — LangChain Rewrite

A local multi-PDF conversational retrieval-augmented generation (RAG) app. The active Streamlit application uses LangChain components for PDF loading, text splitting, embeddings, a persistent FAISS vector store, LCEL orchestration, prompt templates, output parsing, and conversation-history wiring. OpenRouter provides the chat model through LangChain's `ChatOpenAI` integration.

## What changed in this rewrite?

The previous app already used LangChain for its generation call, but its ingestion, embedding/index wrappers, retrieval stages, and history orchestration were largely custom. The active app now follows this LangChain path:

```text
PDF upload
   ↓
PyPDFLoader → RecursiveCharacterTextSplitter
   ↓
HuggingFaceEmbeddings → LangChain FAISS (persisted per PDF collection)
   ↓
Question + SQLite-backed RunnableWithMessageHistory
   ↓
Bound recent messages + rolling summary
   ↓
Optional context-aware query rewrite (ChatPromptTemplate | ChatOpenAI | StrOutputParser)
   ↓
FAISS retriever (optional BM25 + Reciprocal Rank Fusion)
   ↓
Optional CrossEncoder reranking → token-budgeted evidence
   ↓
ChatPromptTemplate | ChatOpenAI | StrOutputParser
   ↓
Answer + source metadata + diagnostics
```

## LangChain concepts used

| Concept | Where it is used |
|---|---|
| `PyPDFLoader` | Reads each uploaded PDF into page-level `Document` objects |
| `RecursiveCharacterTextSplitter` | Splits pages into overlapping chunks while carrying page metadata |
| `HuggingFaceEmbeddings` | Embeds chunks and queries using Sentence Transformers |
| LangChain `FAISS` | Stores vectors, persists collections, and exposes a retriever |
| `ChatPromptTemplate` | Keeps rewrite, answer, and summary prompts structured |
| LCEL `|` composition | Connects prompt → chat model → `StrOutputParser` |
| `RunnablePassthrough.assign` | Carries the request state through retrieval and generation steps |
| `RunnableLambda` / `RunnableBranch` | Implements formatting, token-bounded context, and conditional query rewriting |
| `RunnableWithMessageHistory` | Loads and saves conversation messages through a SQLite-backed history adapter |
| `MessagesPlaceholder` | Supplies selected recent turns to the answer prompt |

The optional reranker remains a Sentence Transformers `CrossEncoder`, because reranking is a separate model stage. Hybrid mode combines LangChain's dense and BM25 retrievers with a small reciprocal-rank-fusion helper.

## Windows PowerShell setup

Use Python 3.11 or 3.12 for this environment for the smoothest compatibility with the current ML packages.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace `OPENROUTER_API_KEY` with a fresh key from OpenRouter. Never commit or share `.env`. If a key has appeared in an uploaded archive or chat, revoke it and create a new one.

Run the UI:

```powershell
python -m streamlit run app.py
```

If activation is blocked by the PowerShell execution policy, run the virtual environment's executable directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

The first run may download the embedding model and, if enabled, the CrossEncoder model. Scanned/image-only PDFs need OCR, which is not included here.

## Settings

- `RETRIEVAL_MODE=dense`: FAISS similarity retriever only; this is the default.
- `RETRIEVAL_MODE=hybrid`: FAISS plus BM25, merged with reciprocal-rank fusion. Requires `rank-bm25` from `requirements.txt`.
- `ENABLE_RERANKER=true`: rank retrieved candidate chunks with the CrossEncoder. Set `false` to avoid loading that model.
- `RETRIEVAL_K`: candidate count before reranking.
- `LLM_CONTEXT_K`: maximum number of reranked chunks considered for the answer.
- `HISTORY_MAX_TURNS` and `HISTORY_TOKEN_BUDGET`: bound recent chat history sent to the LLM.
- `SUMMARY_TRIGGER_TURNS` and `SUMMARY_TOKEN_LIMIT`: configure rolling summaries for older turns.
- `CONTEXT_LIMIT`, `MAX_OUTPUT_TOKENS`, and `SAFETY_MARGIN`: approximate token-budget settings. Token counting is approximate and model tokenizers can differ.

## Persistence and safety

- Uploaded PDFs/chunks and FAISS indexes are stored under `data/`.
- Conversation messages and summaries live in `data/chat_history.db`.
- The app loads its own generated FAISS index from the local data directory. Do not place untrusted FAISS pickle/index files in that directory.
- The conversation ID is used as the local history selector, not as authentication. Do not expose this local app to multiple users without authentication and authorization.
- PDF content and chat history are treated as untrusted data in the prompts. Prompt instructions reduce prompt-injection risk but do not replace access controls or broader security controls.
- `.env`, local `data/`, generated databases, and Python bytecode are excluded from the delivered archive.

## Validation

Run syntax checks and unit tests from the project root:

```powershell
python -m compileall -q app.py src evaluation tests
python -m unittest discover -s tests -v
```

These tests do not call OpenRouter. A full end-to-end check requires installing the requirements, configuring an API key, uploading a text-based PDF, asking several grounded questions, restarting the app, and verifying that the stored index and history are reused.

## Historical evaluation scripts

The `evaluation/` directory and historical result files are retained to preserve the original retrieval experiments. Some scripts import helper functions from the previous custom `src/rag_pipeline.py` implementation; they are a legacy evaluation baseline and are **not** the runtime pipeline used by `app.py`. Rebaseline evaluation on the new LangChain retrieval path before comparing results or claiming an accuracy improvement.
