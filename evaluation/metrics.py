
from typing import List, Set


def precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> float:

    if k <= 0:
        raise ValueError("k must be greater than zero")

    top_k = retrieved_ids[:k]

    if not top_k:
        return 0.0

    relevant_count = sum(
        1 for chunk_id in top_k
        if chunk_id in relevant_ids
    )

    return relevant_count / len(top_k)


def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> float:

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        return 0.0

    top_k = retrieved_ids[:k]

    relevant_count = sum(
        1 for chunk_id in top_k
        if chunk_id in relevant_ids
    )

    return relevant_count / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: Set[str]
) -> float:

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1
    ):

        if chunk_id in relevant_ids:
            return 1 / rank

    return 0.0


def mean_reciprocal_rank(
    retrieved_lists: List[List[str]],
    relevant_lists: List[Set[str]]
) -> float:

    if len(retrieved_lists) != len(relevant_lists):
        raise ValueError(
            "Input lists must have equal lengths"
        )

    if not retrieved_lists:
        return 0.0

    scores = [
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(
            retrieved_lists,
            relevant_lists
        )
    ]

    return sum(scores) / len(scores)
