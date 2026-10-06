"""Retrieval-quality metrics: did we fetch the right document, and how
high did it rank?

All metrics take `retrieved_doc_ids` (ranked, best first) and the single
`gold_doc_id` SQuAD tells us the question was written against. That's a
simplification -- real corpora can have multiple valid source docs -- but
for a controlled ablation study it gives us an unambiguous ground truth.
"""
from __future__ import annotations

import math


def hit_at_k(retrieved_doc_ids: list[str], gold_doc_id: str) -> bool:
    return gold_doc_id in retrieved_doc_ids


def reciprocal_rank(retrieved_doc_ids: list[str], gold_doc_id: str) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == gold_doc_id:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_doc_ids: list[str], gold_doc_id: str) -> float:
    """Binary-relevance nDCG: 1 relevant doc, so IDCG is always 1.0."""
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id == gold_doc_id:
            return 1.0 / math.log2(rank + 1)
    return 0.0


def precision_at_k(retrieved_doc_ids: list[str], gold_doc_id: str) -> float:
    if not retrieved_doc_ids:
        return 0.0
    hits = sum(1 for d in retrieved_doc_ids if d == gold_doc_id)
    return hits / len(retrieved_doc_ids)


def compute_retrieval_metrics(retrieved_doc_ids: list[str], gold_doc_id: str) -> dict:
    return {
        "hit": hit_at_k(retrieved_doc_ids, gold_doc_id),
        "reciprocal_rank": reciprocal_rank(retrieved_doc_ids, gold_doc_id),
        "ndcg": ndcg_at_k(retrieved_doc_ids, gold_doc_id),
        "precision": precision_at_k(retrieved_doc_ids, gold_doc_id),
    }
