# Accuracy Phase 1 — Single-Turn RAG

## Frozen retrieval baseline
- 700-character paragraph-aware chunks
- 100-character overlap
- SentenceTransformer: all-MiniLM-L6-v2
- FAISS candidate retrieval: Top-6
- CrossEncoder: ms-marco-MiniLM-L-6-v2
- All six FAISS candidates are reranked

## Accuracy changes
- LLM context defaults to the top 3 reranked evidence chunks.
- Compact first-page document identity is included for document-level
  entity association.
- Stronger grounded generation prompt.
- Cross-chunk synthesis is allowed when explicitly supported.
- Temperature defaults to 0.1.
- Final prompt token count is checked before the API call.
- Accuracy Dataset V2 separates answer-quality evaluation from the
  historical retrieval benchmark.

## Run
python accuracy_test_v2.py PBI_midterm_report.pdf --limit 5

Then:
python accuracy_test_v2.py PBI_midterm_report.pdf

The original run_evaluation.py and evaluation_dataset.py are preserved.
Do not overwrite them.
