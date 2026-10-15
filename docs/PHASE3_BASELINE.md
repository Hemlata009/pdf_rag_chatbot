# Phase 3 Baseline

## Baseline source
Historical evaluation JSON and Experiment 3/4 logs are preserved from the Phase 2.5 project under `evaluation/results/` and `logs/`.

## Current pipeline
PDF text extraction -> paragraph-aware chunking -> Sentence Transformers embeddings -> FAISS Top-6 candidate retrieval -> CrossEncoder reranking -> token-aware context selection -> LangChain LCEL generation via OpenRouter.

## Memory
SQLite stores messages by conversation ID. The application retrieves a bounded recent history and rolling summary, rewrites follow-up queries, and checks the final assembled prompt against the configured input budget.

## Before/after policy
Record a fresh run with identical settings before making further retrieval/prompt changes. Historical retrieval metrics are a baseline, not proof the reorganized code produces identical live answers. Run tests, then end-to-end checks on a fixed PDF and question set.


## Milestone 4 update (retrieval and answer quality)

- Retrieval default remains `dense`, preserving the Experiment 4 path for baseline comparisons.
- Optional `hybrid` mode merges dense FAISS results with BM25 lexical results using Reciprocal Rank Fusion, then uses the existing CrossEncoder reranker.
- `evaluation/compare_retrieval_modes.py` compares dense, dense+reranker, hybrid, and hybrid+reranker on the existing versioned dataset.
- The system, query-rewriter, and summary instructions are passed as actual LangChain `system` messages separate from evidence/history input; document contents and conversation history are treated as untrusted data, with explicit directions to ignore embedded override/tool/prompt disclosure instructions.
- Answer citations are requested using source filename and page number. The app's source panel continues to show the exact selected evidence chunks.
- No live hybrid-vs-dense result is recorded in this artifact because the execution environment did not have FAISS or Sentence Transformers installed. Run the comparison after dependency installation before selecting a new default.
