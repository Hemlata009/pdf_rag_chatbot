"""
Scripted multi-turn chat test (Round 0).

Concept: instead of typing questions by hand, replay the SAME conversation
under different memory settings and score each turn with simple checks, so a
memory change becomes a before/after number instead of a feeling.

Usage:  python run_chat_test.py            # runs baseline + round1
        python run_chat_test.py round1     # runs only one config
"""
import json
import os
import sys
from datetime import datetime

import src.rag_pipeline as rp
from src.terminal_app import (
    CHAT_LOG_DIR, DEFAULT_PDF, FINAL_K, RETRIEVAL_K, build_rag_system, log_turn,
)

# The 6 questions from the real chat of 2026-10-10 (typos kept on purpose).
# must_rewrite: words the rewritten search question has to contain.
# must_answer : words the answer has to contain (any one is enough).
# bad_rewrite : words that show the rewrite went wrong.
SCRIPT = [
    {"q": "what is the role of hemlata?",
     "must_answer": ["intern"]},
    {"q": "what are the skills she used in her itern?",
     "must_rewrite": ["hemlata"], "must_answer": ["excel", "data"]},
    {"q": "tell me what exactly the use case mentioned for the product being built in this report?"},
    {"q": "how many characters are there in the document?",
     "must_answer": ["could not find"]},  # correct behaviour = refuse
    {"q": "now tell me what I asked you in 2nd question?",
     "must_answer": ["skill"]},           # 2nd question was about skills
    {"q": "what are her skills?",
     "must_rewrite": ["hemlata"], "bad_rewrite": ["assistant"]},
]

# Each config = the memory settings for one run.
CONFIGS = {
    "baseline": dict(HISTORY_MAX_TURNS=2, HISTORY_TOKEN_BUDGET=200,
                     NUMBER_TURNS=False, USE_TOPIC_NOTE=False),
    "round1": dict(HISTORY_MAX_TURNS=5, HISTORY_TOKEN_BUDGET=350,
                   NUMBER_TURNS=True, USE_TOPIC_NOTE=True),
}


def check(step, rewritten, answer):
    """Return list of (check_name, passed) for one turn."""
    out = []
    for w in step.get("must_rewrite", []):
        out.append((f"rewrite has '{w}'", w in rewritten.lower()))
    for w in step.get("bad_rewrite", []):
        out.append((f"rewrite lacks '{w}'", w not in rewritten.lower()))
    if step.get("must_answer"):
        ok = any(w in answer.lower() for w in step["must_answer"])
        out.append((f"answer has any of {step['must_answer']}", ok))
    return out


def run_config(name, settings, model, reranker, chunks, index):
    for key, value in settings.items():  # apply settings to the pipeline module
        setattr(rp, key, value)

    os.makedirs(CHAT_LOG_DIR, exist_ok=True)
    log_path = os.path.join(CHAT_LOG_DIR, f"test_{name}_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    history, topic, rows = [], "", []

    for step in SCRIPT:
        trace = {}
        answer, results = rp.rag_answer(
            query=step["q"], model=model, index=index, chunks=chunks,
            top_k=FINAL_K, reranker=reranker, retrieval_k=RETRIEVAL_K,
            history=history, trace=trace, topic=topic,
        )
        topic = trace.get("topic", topic)
        history.append({"question": step["q"], "answer": answer})
        log_turn(log_path, len(history), step["q"], trace, answer, results)
        rows.append((step, trace.get("search_query", step["q"]), topic, answer))

    return log_path, rows


def main():
    wanted = sys.argv[1:] or list(CONFIGS)
    model, reranker, chunks, index = build_rag_system(DEFAULT_PDF)

    summary = {}
    for name in wanted:
        log_path, rows = run_config(name, CONFIGS[name], model, reranker, chunks, index)
        passed = total = 0
        print("\n" + "#" * 80 + f"\nCONFIG: {name}  {CONFIGS[name]}\nlog: {log_path}\n" + "#" * 80)
        for i, (step, rewritten, topic, answer) in enumerate(rows, start=1):
            results = check(step, rewritten, answer)
            passed += sum(ok for _, ok in results)
            total += len(results)
            print(f"\nT{i} Q : {step['q']}\n    RW: {rewritten}\n    TOPIC: {topic or '-'}")
            print(f"    A : {answer[:160]!r}")
            for label, ok in results:
                print(f"    [{'PASS' if ok else 'FAIL'}] {label}")
        summary[name] = (passed, total)

    print("\n" + "=" * 80 + "\nSCORE")
    for name, (p, t) in summary.items():
        print(f"  {name:10s} {p}/{t} checks passed")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
