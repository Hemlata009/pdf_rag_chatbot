"""Compare dense and hybrid retrieval on the versioned ground-truth dataset.

Run from the project root after installing requirements:
    python -m evaluation.compare_retrieval_modes

No OpenRouter API key is needed because this evaluates retrieval/reranking only.
Model files may need to be downloaded the first time the script runs.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

from evaluation.evaluation_dataset import evaluation_dataset
from evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank
from src.rag_pipeline import (
    build_faiss_index,
    create_chunks,
    extract_pdf_documents,
    load_embedding_model,
    load_reranker,
    retrieve_documents,
    rerank_documents,
)
from src.retrieval.hybrid_retrieval import hybrid_retrieve


def metric_summary(rankings: List[List[str]], relevant_sets: List[Set[str]], k: int) -> Dict:
    """Summarize metrics over positive-answer queries only."""
    positive_pairs = [
        (ranking, relevant)
        for ranking, relevant in zip(rankings, relevant_sets)
        if relevant
    ]
    if not positive_pairs:
        return {"positive_queries": 0, "hit_rate_at_k": 0.0,
                "precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}

    hits, precisions, recalls, reciprocal_ranks = [], [], [], []
    for ranking, relevant in positive_pairs:
        hits.append(1.0 if any(cid in relevant for cid in ranking[:k]) else 0.0)
        precisions.append(precision_at_k(ranking, relevant, k))
        recalls.append(recall_at_k(ranking, relevant, k))
        reciprocal_ranks.append(reciprocal_rank(ranking, relevant))

    count = len(positive_pairs)
    return {
        "positive_queries": count,
        "hit_rate_at_k": round(sum(hits) / count, 4),
        "precision_at_k": round(sum(precisions) / count, 4),
        "recall_at_k": round(sum(recalls) / count, 4),
        "mrr": round(sum(reciprocal_ranks) / count, 4),
    }


def run(pdf_path: Path, k: int, candidate_k: int, save_results: bool = True) -> Dict:
    source_name = pdf_path.name
    print(f"Loading embedding model: {__import__('os').getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')}")
    embedding_model = load_embedding_model()
    reranker = load_reranker()

    documents = extract_pdf_documents(str(pdf_path), source_name=source_name)
    chunks = create_chunks(documents, chunk_size=700, overlap=100)
    actual_ids = {chunk["chunk_id"] for chunk in chunks}
    ground_truth_ids = {
        chunk_id for item in evaluation_dataset for chunk_id in item["relevant_chunk_ids"]
    }
    missing = ground_truth_ids - actual_ids
    if missing:
        raise ValueError(
            "The fixed evaluation set does not match chunks made with CHUNK_SIZE=700 / "
            "CHUNK_OVERLAP=100. Missing IDs: " + ", ".join(sorted(missing))
        )

    index = build_faiss_index(chunks, embedding_model)
    variant_rankings: Dict[str, List[List[str]]] = {
        "dense": [], "dense_reranked": [], "hybrid_rrf": [], "hybrid_rrf_reranked": []
    }
    per_question = []

    for item in evaluation_dataset:
        question = item["question"]
        relevant = set(item["relevant_chunk_ids"])
        dense = retrieve_documents(question, embedding_model, index, chunks, top_k=candidate_k)
        hybrid = hybrid_retrieve(question, dense, chunks, top_k=candidate_k)
        dense_reranked = rerank_documents(question, dense, reranker, top_k=None)
        hybrid_reranked = rerank_documents(question, hybrid, reranker, top_k=None)
        results_by_variant = {
            "dense": dense,
            "dense_reranked": dense_reranked,
            "hybrid_rrf": hybrid,
            "hybrid_rrf_reranked": hybrid_reranked,
        }
        question_result = {"question": question, "relevant_chunk_ids": sorted(relevant), "rankings": {}}
        for variant, result in results_by_variant.items():
            ids = [doc["chunk_id"] for doc in result]
            variant_rankings[variant].append(ids)
            question_result["rankings"][variant] = ids[:k]
        per_question.append(question_result)

    relevant_sets = [set(item["relevant_chunk_ids"]) for item in evaluation_dataset]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pdf": str(pdf_path),
        "chunk_size": 700,
        "chunk_overlap": 100,
        "candidate_k": candidate_k,
        "evaluation_k": k,
        "chunks": len(chunks),
        "questions": len(evaluation_dataset),
        "positive_queries": sum(bool(s) for s in relevant_sets),
        "negative_queries": sum(not s for s in relevant_sets),
        "metrics": {
            name: metric_summary(rankings, relevant_sets, k)
            for name, rankings in variant_rankings.items()
        },
        "per_question": per_question,
        "notes": [
            "Negative/no-answer queries are excluded from retrieval ranking metrics because the dataset has no relevant chunk IDs for them.",
            "No LLM generation is run by this script; grounded answer correctness and citation support need a separate answer-level evaluation.",
            "The dense mode remains the default until measured results justify changing it.",
        ],
    }

    print("\nRetrieval comparison (positive-answer queries only)")
    print("=" * 76)
    print(f"{'Variant':28} {'Hit@K':>8} {'Precision':>11} {'Recall':>9} {'MRR':>8}")
    for name, metrics in report["metrics"].items():
        print(f"{name:28} {metrics['hit_rate_at_k']:>8.4f} {metrics['precision_at_k']:>11.4f} "
              f"{metrics['recall_at_k']:>9.4f} {metrics['mrr']:>8.4f}")

    if save_results:
        output_dir = Path("evaluation/results")
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"retrieval_comparison_{stamp}.json"
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved report: {output_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", default="sample_docs/PBI_midterm_report.pdf", help="Evaluation PDF path")
    parser.add_argument("--k", type=int, default=4, help="K for metrics")
    parser.add_argument("--candidate-k", type=int, default=6, help="Candidate pool size for dense/BM25")
    args = parser.parse_args()
    if args.k < 1 or args.candidate_k < args.k:
        parser.error("Require k >= 1 and candidate-k >= k")
    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        parser.error(f"PDF not found: {pdf_path}")
    run(pdf_path, k=args.k, candidate_k=args.candidate_k)


if __name__ == "__main__":
    main()
