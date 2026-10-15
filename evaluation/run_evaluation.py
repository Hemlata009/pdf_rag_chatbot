import json
import os
from contextlib import redirect_stdout
from datetime import datetime

from src.rag_pipeline import (
    build_faiss_index,
    create_chunks,
    extract_pdf_documents,
    load_embedding_model,
    load_reranker,
    retrieve_documents,
    rerank_documents,
    select_documents_by_token_budget,
)

from evaluation.evaluation_dataset import evaluation_dataset


PDF_PATH = "sample_docs/PBI_midterm_report.pdf"
SOURCE_NAME = "PBI_midterm_report.pdf"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
RETRIEVAL_K = 6
EVAL_K = 4


# ============================================================
# METRICS
# ============================================================

def precision_at_k(retrieved_ids, relevant_ids, k):
    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    return sum(
        chunk_id in relevant_ids
        for chunk_id in top_k
    ) / len(top_k)


def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    return sum(
        chunk_id in relevant_ids
        for chunk_id in top_k
    ) / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):
    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in relevant_ids:
            return 1 / rank

    return 0.0


# ============================================================
# EXPERIMENT 4
# ============================================================

def run_evaluation():

    print("\n" + "=" * 80)
    print("        RAG EVALUATION - EXPERIMENT 4")
    print("        FAISS + CROSS-ENCODER + TOKEN BUDGET")
    print("=" * 80)

    print("\nConfiguration")
    print("-" * 80)
    print(f"PDF Path       : {PDF_PATH}")
    print(f"Chunk Size     : {CHUNK_SIZE}")
    print(f"Chunk Overlap  : {CHUNK_OVERLAP}")
    print(f"Retrieval K    : {RETRIEVAL_K}")
    print(f"Evaluation K   : {EVAL_K}")

    print("\nPipeline:")
    print(
        "Query -> FAISS Top-6 -> CrossEncoder "
        "-> Reranked Top-6 -> Token Budget -> LLM"
    )

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 1: LOAD EMBEDDING MODEL")
    print("=" * 80)

    embedding_model = load_embedding_model()

    print("Embedding model loaded successfully.")

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 2: LOAD CROSS-ENCODER RERANKER")
    print("=" * 80)

    reranker = load_reranker()

    print("CrossEncoder reranker loaded successfully.")

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 3: EXTRACT PDF")
    print("=" * 80)

    documents = extract_pdf_documents(
        PDF_PATH,
        source_name=SOURCE_NAME,
    )

    print(
        f"Pages extracted: {len(documents)}"
    )

    for document in documents:
        print(
            f"Page {document['metadata']['page']} | "
            f"Characters: {len(document['text'])}"
        )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 4: CREATE CHUNKS")
    print("=" * 80)

    chunks = create_chunks(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    print(
        f"Total chunks created: {len(chunks)}"
    )

    # --------------------------------------------------------
    # VALIDATE EVALUATION GROUND-TRUTH IDS
    # --------------------------------------------------------

    actual_chunk_ids = {chunk["chunk_id"] for chunk in chunks}
    evaluation_chunk_ids = {
        chunk_id
        for item in evaluation_dataset
        for chunk_id in item["relevant_chunk_ids"]
    }

    missing_ground_truth_ids = evaluation_chunk_ids - actual_chunk_ids

    if missing_ground_truth_ids:
        raise ValueError(
            "Evaluation dataset contains chunk IDs that do not exist "
            "in the generated chunks: "
            + ", ".join(sorted(missing_ground_truth_ids))
        )

    print("Evaluation ground-truth chunk IDs validated successfully.")

    for chunk in chunks:
        print("\n" + "-" * 80)
        print(f"Chunk ID    : {chunk['chunk_id']}")
        print(f"Page        : {chunk['metadata']['page']}")
        print(f"Characters  : {len(chunk['text'])}")
        print("Text preview:")
        print(
            chunk["text"][:300]
            .replace("\n", " ")
        )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 5: BUILD FAISS INDEX")
    print("=" * 80)

    index = build_faiss_index(
        chunks,
        embedding_model,
    )

    print("FAISS index built successfully.")
    print(f"Number of vectors: {index.ntotal}")

    # --------------------------------------------------------
    # METRIC STORAGE
    # --------------------------------------------------------

    all_precision = []
    all_recall = []
    all_rr = []
    question_results = []

    # --------------------------------------------------------
    # STEP 6
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STEP 6: RUN EXPERIMENT 4")
    print("=" * 80)

    for question_number, item in enumerate(
        evaluation_dataset,
        start=1,
    ):

        question = item["question"]

        relevant_ids = set(
            item["relevant_chunk_ids"]
        )

        print("\n\n" + "=" * 80)
        print(f"QUESTION {question_number}")
        print("=" * 80)

        print(f"Question:\n{question}")

        print("\nExpected relevant chunks:")

        if relevant_ids:
            for chunk_id in sorted(relevant_ids):
                print(f"  ✓ {chunk_id}")
        else:
            print(
                "  None "
                "(negative / no-answer query)"
            )

        # ====================================================
        # STAGE 1: FAISS
        # ====================================================

        print("\n" + "-" * 80)
        print("STAGE 1: FAISS CANDIDATE RETRIEVAL")
        print("-" * 80)

        retrieved_candidates = retrieve_documents(
            question,
            embedding_model,
            index,
            chunks,
            top_k=RETRIEVAL_K,
        )

        print(
            f"FAISS returned "
            f"{len(retrieved_candidates)} candidates."
        )

        for rank, doc in enumerate(
            retrieved_candidates,
            start=1,
        ):

            print(f"\nFAISS Rank {rank}")
            print(
                f"Chunk ID     : "
                f"{doc['chunk_id']}"
            )
            print(
                f"Similarity   : "
                f"{doc['similarity']:.4f}"
            )
            print(
                f"Page         : "
                f"{doc['metadata']['page']}"
            )
            print(
                f"Relevant?    : "
                f"{doc['chunk_id'] in relevant_ids}"
            )

        # ====================================================
        # STAGE 2: CROSS-ENCODER
        # ====================================================

        print("\n" + "-" * 80)
        print("STAGE 2: CROSS-ENCODER RERANKING")
        print("-" * 80)

        print(
            "CrossEncoder is scoring all FAISS candidates..."
        )

        # IMPORTANT:
        # Do NOT truncate to Top-4 here.
        reranked_documents = rerank_documents(
            question,
            retrieved_candidates,
            reranker,
            top_k=None,
        )

        print(
            f"\nReranked candidate pool: "
            f"{len(reranked_documents)}"
        )

        for rank, doc in enumerate(
            reranked_documents,
            start=1,
        ):

            print(f"\nRerank {rank}")
            print(
                f"Chunk ID      : "
                f"{doc['chunk_id']}"
            )
            print(
                f"FAISS Score   : "
                f"{doc['similarity']:.4f}"
            )
            print(
                f"Rerank Score  : "
                f"{doc['rerank_score']:.4f}"
            )
            print(
                f"Page          : "
                f"{doc['metadata']['page']}"
            )
            print(
                f"Relevant?     : "
                f"{doc['chunk_id'] in relevant_ids}"
            )

        # ====================================================
        # STAGE 3: TOKEN BUDGET
        # ====================================================

        print("\n" + "-" * 80)
        print("STAGE 3: TOKEN-AWARE CONTEXT SELECTION")
        print("-" * 80)

        context_result = select_documents_by_token_budget(
            question,
            reranked_documents,
        )

        selected_documents = context_result[
            "selected_chunk_data"
        ]

        print("\nSelected chunks for LLM:")

        for rank, doc in enumerate(
            selected_documents,
            start=1,
        ):

            print(
                f"  {rank}. {doc['chunk_id']} | "
                f"Rerank={doc.get('rerank_score', 0.0):.4f}"
            )

        print(
            f"\nSelected chunk count : "
            f"{len(selected_documents)}"
        )

        print(
            f"Total input tokens  : "
            f"{context_result['total_input_tokens']}"
        )

        print(
            f"Input budget        : "
            f"{context_result['input_budget']}"
        )

        print(
            f"Remaining tokens    : "
            f"{context_result['remaining_tokens']}"
        )

        # ====================================================
        # STAGE 4: RETRIEVAL METRICS
        # ====================================================

        print("\n" + "-" * 80)
        print(
            f"STAGE 4: RETRIEVAL METRICS @ {EVAL_K}"
        )
        print("-" * 80)

        # Metrics intentionally evaluate the reranked Top-4.
        # Token-budget selection is evaluated separately above.
        evaluation_documents = reranked_documents[:EVAL_K]

        retrieved_ids = [
            doc["chunk_id"]
            for doc in evaluation_documents
        ]

        precision = precision_at_k(
            retrieved_ids,
            relevant_ids,
            EVAL_K,
        )

        recall = recall_at_k(
            retrieved_ids,
            relevant_ids,
            EVAL_K,
        )

        rr = reciprocal_rank(
            retrieved_ids,
            relevant_ids,
        )

        # Negative / no-answer queries have no ground-truth chunk, so
        # precision/recall/MRR are unconditionally 0 for them no matter
        # how good retrieval is. Excluding them keeps the metric meaningful
        # for the 15 answerable questions instead of being dragged down
        # by one question it was never designed to score.
        if relevant_ids:
            all_precision.append(precision)
            all_recall.append(recall)
            all_rr.append(rr)

        print("\nEvaluation Top-4 IDs:")

        for rank, chunk_id in enumerate(
            retrieved_ids,
            start=1,
        ):
            print(f"  {rank}. {chunk_id}")

        print(
            f"\nPrecision@{EVAL_K} : "
            f"{precision:.4f} "
            f"({precision * 100:.2f}%)"
        )

        print(
            f"Recall@{EVAL_K}    : "
            f"{recall:.4f} "
            f"({recall * 100:.2f}%)"
        )

        print(
            f"Reciprocal Rank    : "
            f"{rr:.4f}"
        )

        question_results.append({
            "question_number": question_number,
            "question": question,
            "expected_relevant_chunks": sorted(
                relevant_ids
            ),
            "faiss_candidates": [
                {
                    "rank": rank,
                    "chunk_id": doc["chunk_id"],
                    "similarity": doc["similarity"],
                    "page": doc["metadata"]["page"],
                    "relevant": (
                        doc["chunk_id"]
                        in relevant_ids
                    ),
                }
                for rank, doc in enumerate(
                    retrieved_candidates,
                    start=1,
                )
            ],
            "reranked_candidates": [
                {
                    "rank": rank,
                    "chunk_id": doc["chunk_id"],
                    "faiss_similarity": doc["similarity"],
                    "rerank_score": doc["rerank_score"],
                    "page": doc["metadata"]["page"],
                    "relevant": (
                        doc["chunk_id"]
                        in relevant_ids
                    ),
                }
                for rank, doc in enumerate(
                    reranked_documents,
                    start=1,
                )
            ],
            "token_selected_chunks": [
                doc["chunk_id"]
                for doc in selected_documents
            ],
            "token_selected_count": len(
                selected_documents
            ),
            "total_input_tokens": context_result[
                "total_input_tokens"
            ],
            "input_budget": context_result[
                "input_budget"
            ],
            "remaining_tokens": context_result[
                "remaining_tokens"
            ],
            "precision_at_4": precision,
            "recall_at_4": recall,
            "reciprocal_rank": rr,
        })

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    avg_precision = (
        sum(all_precision)
        / len(all_precision)
    )

    avg_recall = (
        sum(all_recall)
        / len(all_recall)
    )

    mrr = (
        sum(all_rr)
        / len(all_rr)
    )

    print("\n\n" + "=" * 80)
    print("FINAL RESULTS - EXPERIMENT 4")
    print("=" * 80)

    print(
        f"Precision@{EVAL_K} : "
        f"{avg_precision:.4f} "
        f"({avg_precision * 100:.2f}%)"
    )

    print(
        f"Recall@{EVAL_K}    : "
        f"{avg_recall:.4f} "
        f"({avg_recall * 100:.2f}%)"
    )

    print(
        f"MRR                : "
        f"{mrr:.4f} "
        f"({mrr * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    exp2_precision = 0.2500
    exp2_recall = 0.8438
    exp2_mrr = 0.6458

    exp3_precision = 0.2500
    exp3_recall = 0.8438
    exp3_mrr = 0.7500

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)

    print(
        "Experiment 2: "
        "P@4=25.00% | R@4=84.38% | MRR=64.58%"
    )

    print(
        "Experiment 3: "
        "P@4=25.00% | R@4=84.38% | MRR=75.00%"
    )

    print(
        f"Experiment 4: "
        f"P@4={avg_precision * 100:.2f}% | "
        f"R@4={avg_recall * 100:.2f}% | "
        f"MRR={mrr * 100:.2f}%"
    )

    print("\nExperiment 4 - Experiment 3:")

    print(
        f"Precision change: "
        f"{(avg_precision - exp3_precision) * 100:+.2f} "
        f"percentage points"
    )

    print(
        f"Recall change   : "
        f"{(avg_recall - exp3_recall) * 100:+.2f} "
        f"percentage points"
    )

    print(
        f"MRR change      : "
        f"{(mrr - exp3_mrr) * 100:+.2f} "
        f"percentage points"
    )

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    tolerance = 1e-6

    if avg_precision > exp3_precision + tolerance:
        print(
            "✓ Precision improved over Experiment 3."
        )
    elif avg_precision < exp3_precision - tolerance:
        print(
            "✗ Precision decreased compared with Experiment 3."
        )
    else:
        print(
            "→ Precision remained unchanged."
        )

    if avg_recall > exp3_recall + tolerance:
        print(
            "✓ Recall improved over Experiment 3."
        )
    elif avg_recall < exp3_recall - tolerance:
        print(
            "⚠ Recall decreased compared with Experiment 3."
        )
    else:
        print(
            "→ Recall remained unchanged."
        )

    if mrr > exp3_mrr + tolerance:
        print(
            "✓ MRR improved over Experiment 3."
        )
    elif mrr < exp3_mrr - tolerance:
        print(
            "✗ MRR decreased compared with Experiment 3."
        )
    else:
        print(
            "→ MRR remained unchanged."
        )

    print("\nToken-budget observation:")
    average_selected = sum(
        result["token_selected_count"]
        for result in question_results
    ) / len(question_results)

    print(
        f"Average chunks selected for LLM: "
        f"{average_selected:.2f}"
    )

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    return {
        "experiment": "Experiment 4",
        "retrieval_k": RETRIEVAL_K,
        "evaluation_k": EVAL_K,
        "precision_at_4": avg_precision,
        "recall_at_4": avg_recall,
        "mrr": mrr,
        "average_token_selected_chunks": average_selected,
        "question_results": question_results,
    }


def main():
    """Run evaluation and save complete results to logs/."""

    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_path = os.path.join(
        logs_dir,
        f"experiment4_{timestamp}.log",
    )

    json_path = os.path.join(
        logs_dir,
        f"experiment4_{timestamp}.json",
    )

    print("=" * 72)
    print("RAG Experiment 4 evaluation started")
    print(f"Complete log: {log_path}")
    print(f"Structured results: {json_path}")
    print("=" * 72)

    try:
        with open(
            log_path,
            "w",
            encoding="utf-8",
        ) as log_file:

            with redirect_stdout(log_file):
                results = run_evaluation()

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                results,
                file,
                indent=2,
            )

        print(
            "Evaluation completed successfully."
        )

        print(
            f"Precision@4 : "
            f"{results['precision_at_4'] * 100:.2f}%"
        )

        print(
            f"Recall@4    : "
            f"{results['recall_at_4'] * 100:.2f}%"
        )

        print(
            f"MRR         : "
            f"{results['mrr'] * 100:.2f}%"
        )

        print(
            f"Avg selected chunks: "
            f"{results['average_token_selected_chunks']:.2f}"
        )

        print(f"Full log    : {log_path}")
        print(f"JSON results: {json_path}")

    except Exception:

        with open(
            log_path,
            "a",
            encoding="utf-8",
        ) as log_file:

            log_file.write(
                "\n\nEVALUATION FAILED\n"
            )

            import traceback
            traceback.print_exc(
                file=log_file
            )

        print(
            f"Evaluation failed. "
            f"Check the log: {log_path}"
        )

        raise


if __name__ == "__main__":
    main()
