"""Lightweight BM25 retrieval and reciprocal-rank fusion helpers.

This module deliberately has no third-party dependencies so lexical retrieval
can be unit-tested independently of FAISS, transformer models, and the LLM.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Sequence

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> List[str]:
    """Normalize text into lowercase alphanumeric terms."""
    return _TOKEN_PATTERN.findall((text or "").lower())


def bm25_retrieve(
    query: str,
    chunks: Sequence[Dict],
    top_k: int = 6,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> List[Dict]:
    """Return chunks ranked with Okapi BM25.

    The index is built in memory per call to keep this helper stateless and
    ensure results always reflect the currently selected document collection.
    The document collection is already de-duplicated by DocumentStore.
    """
    if not query or not query.strip() or not chunks or top_k <= 0:
        return []
    if k1 <= 0 or not 0 <= b <= 1:
        raise ValueError("BM25 requires k1 > 0 and b between 0 and 1")

    query_terms = tokenize(query)
    if not query_terms:
        return []

    tokenized_docs = [tokenize(str(chunk.get("text", ""))) for chunk in chunks]
    doc_count = len(tokenized_docs)
    lengths = [len(tokens) for tokens in tokenized_docs]
    average_length = (sum(lengths) / doc_count) if doc_count else 0.0
    if average_length == 0:
        return []

    document_frequency: Counter = Counter()
    term_frequencies: List[Counter] = []
    for tokens in tokenized_docs:
        frequencies = Counter(tokens)
        term_frequencies.append(frequencies)
        document_frequency.update(frequencies.keys())

    scored = []
    for index, (chunk, frequencies) in enumerate(zip(chunks, term_frequencies)):
        score = 0.0
        doc_length = lengths[index]
        for term in query_terms:
            tf = frequencies.get(term, 0)
            if tf == 0:
                continue
            df = document_frequency[term]
            inverse_document_frequency = math.log(
                1.0 + (doc_count - df + 0.5) / (df + 0.5)
            )
            denominator = tf + k1 * (1.0 - b + b * doc_length / average_length)
            score += inverse_document_frequency * (tf * (k1 + 1.0) / denominator)

        if score > 0:
            result = dict(chunk)
            result["bm25_score"] = float(score)
            scored.append((score, index, result))

    scored.sort(key=lambda row: (-row[0], row[1]))
    return [row[2] for row in scored[: min(top_k, len(scored))]]


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Sequence[Dict]],
    top_k: int | None = None,
    rrf_k: int = 60,
) -> List[Dict]:
    """Fuse ranked lists using Reciprocal Rank Fusion (RRF).

    RRF combines rankings rather than incompatible raw similarity/BM25 scores.
    Duplicate chunk IDs are represented once. Existing score metadata is
    retained and ``fusion_score`` records the fused ranking score.
    """
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    fused: Dict[str, Dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            chunk_id = str(item.get("chunk_id", ""))
            if not chunk_id:
                continue
            if chunk_id not in fused:
                fused[chunk_id] = dict(item)
                fused[chunk_id]["fusion_score"] = 0.0
            fused[chunk_id]["fusion_score"] += 1.0 / (rrf_k + rank)

    ordered = sorted(
        fused.values(),
        key=lambda item: (-item["fusion_score"], str(item.get("chunk_id", ""))),
    )
    if top_k is None:
        return ordered
    if top_k <= 0:
        return []
    return ordered[:top_k]


def hybrid_retrieve(
    query: str,
    dense_results: Sequence[Dict],
    chunks: Sequence[Dict],
    top_k: int = 6,
    rrf_k: int = 60,
) -> List[Dict]:
    """Combine pre-ranked dense candidates with BM25 candidates via RRF."""
    if top_k <= 0:
        return []
    lexical_results = bm25_retrieve(query, chunks, top_k=top_k)
    return reciprocal_rank_fusion(
        [dense_results, lexical_results], top_k=top_k, rrf_k=rrf_k
    )
