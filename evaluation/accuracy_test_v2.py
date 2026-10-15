import argparse
import json
from datetime import datetime
from pathlib import Path

from src.rag_pipeline import (
    build_faiss_index,
    create_chunks,
    extract_pdf_documents,
    load_embedding_model,
    load_reranker,
    rag_answer,
    retrieve_documents,
    rerank_documents,
    build_prompt,
    RETRIEVAL_K,
    LLM_CONTEXT_K,
)
from evaluation.accuracy_dataset_v2 import ACCURACY_DATASET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pdf",
        nargs="?",
        default="sample_docs/PBI_midterm_report.pdf",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    model = load_embedding_model()
    reranker = load_reranker()

    documents = extract_pdf_documents(
        str(pdf_path),
        pdf_path.name,
    )
    chunks = create_chunks(documents, chunk_size=700, overlap=100)
    index = build_faiss_index(chunks, model)

    dataset = ACCURACY_DATASET[:args.limit] if args.limit else ACCURACY_DATASET
    output = []

    print(f"PDF: {pdf_path}")
    print(f"Pages: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"FAISS K: {RETRIEVAL_K}")
    print(f"LLM context K: {LLM_CONTEXT_K}")
    print()

    for item in dataset:
        q = item["question"]
        print("=" * 90)
        print(f"Q{item['id']}: {q}")
        print("=" * 90)

        retrieved = retrieve_documents(
            q, model, index, chunks, top_k=RETRIEVAL_K
        )
        reranked = rerank_documents(q, retrieved, reranker)

        print("\nRERANKED:")
        for i, d in enumerate(reranked, 1):
            print(
                f"{i}. {d['chunk_id']} | "
                f"page={d['metadata']['page']} | "
                f"FAISS={d.get('similarity', 0):.4f} | "
                f"rerank={d.get('rerank_score', 0):.4f}"
            )

        prompt, stats = build_prompt(q, reranked, document_header=chunks[0] if chunks else None)

        # Generate using the exact prompt used by the pipeline.
        from src.rag_pipeline import generate_answer
        answer = generate_answer(prompt, system_prompt=stats.get("generation_system_prompt"))
        import src.rag_pipeline as rag_pipeline
        model_used = rag_pipeline.LAST_MODEL_USED
        print(f"\nMODEL USED: {model_used}")

        print("\nSELECTED FOR LLM:")
        for d in stats["selected_chunk_data"]:
            print(
                f"- {d['chunk_id']} | "
                f"page={d['metadata']['page']}"
            )

        print(
            f"\nPrompt tokens: {stats['total_input_tokens']} / "
            f"{stats['input_budget']}"
        )
        print("\nANSWER:")
        print(answer)

        output.append({
            "id": item["id"],
            "question": q,
            "gold_answer": item["gold_answer"],
            "supporting_chunks": item["supporting_chunks"],
            "supporting_pages": item["supporting_pages"],
            "question_type": item["question_type"],
            "expected_abstention": item["expected_abstention"],
            "reranked_chunks": [
                {
                    "chunk_id": d["chunk_id"],
                    "page": d["metadata"]["page"],
                    "similarity": d.get("similarity"),
                    "rerank_score": d.get("rerank_score"),
                }
                for d in reranked
            ],
            "selected_chunks": [
                d["chunk_id"] for d in stats["selected_chunk_data"]
            ],
            "selected_pages": [
                d["metadata"]["page"]
                for d in stats["selected_chunk_data"]
            ],
            "has_header": stats["has_header"],
            "evidence_chunks": stats["evidence_chunks"],
            "prompt_tokens": stats["total_input_tokens"],
            "model_used": model_used,
            "answer": answer,
        })

    filename = (
        "accuracy_results_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".json"
    )
    Path(filename).write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("\nSaved:", filename)


if __name__ == "__main__":
    main()
