import argparse
import json
import os
from datetime import datetime

from evaluation.evaluation_dataset import evaluation_dataset
from src.rag_pipeline import (
    build_faiss_index,
    create_chunks,
    extract_pdf_documents,
    load_embedding_model,
    load_reranker,
    rag_answer,
)


def main():
    parser = argparse.ArgumentParser(
        description="Single-turn RAG accuracy test. Answers are reviewed manually; this script does not replace human evaluation."
    )
    parser.add_argument("--pdf", default="sample_docs/PBI_midterm_report.pdf")
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()

    model = load_embedding_model()
    reranker = load_reranker()
    documents = extract_pdf_documents(args.pdf, os.path.basename(args.pdf))
    chunks = create_chunks(documents, chunk_size=700, overlap=100)
    index = build_faiss_index(chunks, model)

    results = []
    cases = evaluation_dataset[: max(0, args.limit)]

    print("\n" + "=" * 80)
    print("SINGLE-TURN RAG ACCURACY TEST")
    print("=" * 80)
    print(f"PDF: {args.pdf}")
    print(f"Questions: {len(cases)}")
    print("Retrieval: FAISS Top-6 -> CrossEncoder -> token-aware context")
    print("Generation: grounded prompt, temperature controlled by LLM_TEMPERATURE")

    for number, case in enumerate(cases, start=1):
        question = case["question"]
        print("\n" + "-" * 80)
        print(f"QUESTION {number}: {question}")

        try:
            answer, selected = rag_answer(
                query=question,
                model=model,
                index=index,
                chunks=chunks,
                top_k=4,
                reranker=reranker,
                retrieval_k=6,
            )

            selected_ids = [item["chunk_id"] for item in selected]
            selected_pages = sorted({
                item["metadata"].get("page")
                for item in selected
            })

            print("\nANSWER:")
            print(answer)
            print("\nSELECTED CHUNKS:")
            for item in selected:
                print(
                    f"- {item['chunk_id']} | page={item['metadata'].get('page')} | "
                    f"rerank={item.get('rerank_score')}"
                )

            results.append({
                "question_number": number,
                "question": question,
                "expected_relevant_chunk_ids": sorted(case["relevant_chunk_ids"]),
                "selected_chunk_ids": selected_ids,
                "selected_pages": selected_pages,
                "answer": answer,
            })

        except Exception as exc:
            print(f"ERROR: {exc}")
            results.append({
                "question_number": number,
                "question": question,
                "expected_relevant_chunk_ids": sorted(case["relevant_chunk_ids"]),
                "error": str(exc),
            })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("logs", f"accuracy_test_{timestamp}.json")
    os.makedirs("logs", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"Saved accuracy test results: {output_path}")
    print("Review each answer for: correctness, faithfulness, completeness, and source correctness.")


if __name__ == "__main__":
    main()
