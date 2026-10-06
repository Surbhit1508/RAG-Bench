"""Generation-quality metrics: correctness vs gold answer, and a
groundedness/faithfulness heuristic vs the retrieved context.

EM/F1 follow the standard SQuAD evaluation normalization (lowercase,
strip punctuation/articles, collapse whitespace) so our numbers are
comparable to published SQuAD results people have seen before.
"""
from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold_answers: list[str]) -> bool:
    pred_norm = normalize_answer(prediction)
    return any(pred_norm == normalize_answer(g) for g in gold_answers)


def _f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction: str, gold_answers: list[str]) -> float:
    return max(_f1(prediction, g) for g in gold_answers)


_ABSTENTION_PHRASES = ("i don't know", "i do not know", "cannot be determined", "not contained in the context")


def is_abstention(prediction: str) -> bool:
    norm = prediction.lower()
    return any(phrase in norm for phrase in _ABSTENTION_PHRASES)


def groundedness_score(prediction: str, context_chunks: list[str]) -> float:
    """Heuristic faithfulness proxy: fraction of the answer's content
    words that literally appear somewhere in the retrieved context.

    This is NOT a substitute for NLI-based entailment or an LLM-judge --
    it's cheap, fast, and explainable, and it catches the common failure
    mode where a small model just makes something up that shares no
    vocabulary with what it was given. We're upfront about that
    limitation in the report rather than overselling it as "hallucination
    detection."
    """
    pred_tokens = set(normalize_answer(prediction).split())
    if not pred_tokens:
        return 0.0
    context_tokens = set(normalize_answer(" ".join(context_chunks)).split())
    overlap = pred_tokens & context_tokens
    return len(overlap) / len(pred_tokens)


def compute_generation_metrics(prediction: str, gold_answers: list[str], context_chunks: list[str]) -> dict:
    return {
        "exact_match": exact_match(prediction, gold_answers),
        "f1": f1_score(prediction, gold_answers),
        "is_abstention": is_abstention(prediction),
        "groundedness": groundedness_score(prediction, context_chunks),
    }
