"""Glue a RetrievalIndex + generation model together to answer one
question under a given RAGConfig, and score the result.
"""
from __future__ import annotations

from ragbench.config import RAGConfig
from ragbench.generation import generate_answer
from ragbench.metrics_generation import compute_generation_metrics
from ragbench.metrics_retrieval import compute_retrieval_metrics
from ragbench.retrieval import Chunk, RetrievalIndex


def answer_question(index: RetrievalIndex, cfg: RAGConfig, question: str) -> tuple[list[Chunk], str]:
    """Retrieve (+ optionally rerank) then generate. Returns the final
    chunks used as context and the generated answer text.
    """
    fetch_k = cfg.top_k * 3 if cfg.use_reranker else cfg.top_k
    candidates = index.search(question, method=cfg.retrieval_method, top_k=fetch_k, alpha=cfg.hybrid_alpha)

    if cfg.use_reranker:
        candidates = index.rerank(question, candidates)[: cfg.top_k]
    else:
        candidates = candidates[: cfg.top_k]

    answer = generate_answer(question, [c.text for c in candidates])
    return candidates, answer


def run_single(index: RetrievalIndex, cfg: RAGConfig, qa: dict) -> dict:
    """Run one QA pair through one RAGConfig, returning a flat result
    record ready to append to the experiment results table.
    """
    chunks, answer = answer_question(index, cfg, qa["question"])
    retrieved_doc_ids = [c.doc_id for c in chunks]

    retrieval_metrics = compute_retrieval_metrics(retrieved_doc_ids, qa["gold_doc_id"])
    generation_metrics = compute_generation_metrics(answer, qa["gold_answers_all"], [c.text for c in chunks])

    record = {
        "config_name": cfg.name,
        "qa_id": qa["qa_id"],
        "question": qa["question"],
        "gold_answer": qa["gold_answer"],
        "generated_answer": answer,
        "gold_doc_id": qa["gold_doc_id"],
        "retrieved_doc_ids": ",".join(retrieved_doc_ids),
    }
    record.update({f"retrieval_{k}": v for k, v in retrieval_metrics.items()})
    record.update({f"gen_{k}": v for k, v in generation_metrics.items()})
    return record
