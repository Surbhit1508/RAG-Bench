from ragbench.metrics_generation import (
    normalize_answer, exact_match, f1_score, is_abstention, groundedness_score,
)
from ragbench.metrics_retrieval import hit_at_k, reciprocal_rank, ndcg_at_k, precision_at_k


def test_normalize_answer_strips_articles_and_punct():
    assert normalize_answer("The Quick, Fox!") == "quick fox"


def test_exact_match_case_and_article_insensitive():
    assert exact_match("the Linear form", ["linear form"])
    assert not exact_match("circular", ["linear"])


def test_f1_score_partial_overlap():
    score = f1_score("New York City", ["New York"])
    assert 0.5 < score < 1.0


def test_f1_score_no_overlap_is_zero():
    assert f1_score("banana", ["apple"]) == 0.0


def test_is_abstention_detects_dont_know():
    assert is_abstention("I don't know.")
    assert not is_abstention("Paris")


def test_groundedness_score_full_overlap():
    assert groundedness_score("New York", ["The city of New York is large."]) == 1.0


def test_groundedness_score_no_overlap():
    assert groundedness_score("Tokyo", ["Paris is the capital of France."]) == 0.0


def test_retrieval_metrics_hit_and_rank():
    retrieved = ["doc_b", "doc_a", "doc_c"]
    assert hit_at_k(retrieved, "doc_a")
    assert not hit_at_k(retrieved, "doc_z")
    assert reciprocal_rank(retrieved, "doc_a") == 0.5
    assert reciprocal_rank(retrieved, "doc_z") == 0.0
    assert precision_at_k(retrieved, "doc_a") == 1 / 3


def test_ndcg_rewards_higher_rank():
    assert ndcg_at_k(["doc_a", "doc_b"], "doc_a") > ndcg_at_k(["doc_b", "doc_a"], "doc_a")
