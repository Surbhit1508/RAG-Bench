"""Central configuration for RAGBench.

Keeping every path, model name, and tunable constant in one place means
the rest of the codebase never hardcodes a magic string -- if we want to
swap the generation model or the corpus size, we change it here once.
(DRY: one source of truth for config.)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- Corporate network plumbing -------------------------------------------
# HuggingFace Hub downloads (datasets + model weights) need to go through
# Walmart's proxy to reach the public internet. We set this once, here,
# so every module that touches `datasets`/`transformers` gets it for free
# just by importing ragbench.config before doing any downloads.
_PROXY = "http://sysproxy.wal-mart.com:8080"
os.environ.setdefault("HTTP_PROXY", _PROXY)
os.environ.setdefault("HTTPS_PROXY", _PROXY)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")  # keep transport simple/robust
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # Xet's cas-server host doesn't like our corp proxy; plain HTTPS does

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report"

for _dir in (DATA_DIR, RESULTS_DIR, REPORT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QA_PATH = DATA_DIR / "qa_pairs.jsonl"
RESULTS_CSV = RESULTS_DIR / "results.csv"
SUMMARY_JSON = RESULTS_DIR / "summary.json"

# --- Dataset scope -------------------------------------------------------
N_QA_PAIRS = 150          # CPU-only inference budget; easy to bump once pipeline is validated
RANDOM_SEED = 42

# --- Models ---------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "google/flan-t5-base"        # instruction-tuned seq2seq, fast on CPU, available via our Artifactory mirror
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

MAX_NEW_TOKENS = 64
GENERATION_TEMPERATURE = 0.0  # deterministic -- we want reproducible eval, not creativity


@dataclass(frozen=True)
class RAGConfig:
    """One point in the ablation grid: a specific RAG design choice.

    Every field is something we deliberately vary and measure the effect
    of. Keep it immutable (frozen) so a config can be hashed/used as a
    dict key or dataframe row without surprises.
    """

    name: str
    chunk_strategy: str = "fixed"       # "fixed" | "sentence"
    chunk_size: int = 200               # tokens (approx, whitespace-split)
    chunk_overlap: int = 40
    retrieval_method: str = "dense"     # "dense" | "bm25" | "hybrid"
    top_k: int = 3
    use_reranker: bool = False
    hybrid_alpha: float = 0.5           # weight for dense score in hybrid (0=bm25 only, 1=dense only)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "chunk_strategy": self.chunk_strategy,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "retrieval_method": self.retrieval_method,
            "top_k": self.top_k,
            "use_reranker": self.use_reranker,
            "hybrid_alpha": self.hybrid_alpha,
        }


def default_ablation_grid() -> list[RAGConfig]:
    """The set of configs we compare against each other.

    Each one changes ONE thing relative to a sensible baseline so we can
    attribute performance differences to a specific design choice instead
    of a tangle of confounded variables. That's the whole point of an
    ablation study.
    """
    return [
        RAGConfig(name="baseline_dense_fixed200", retrieval_method="dense", chunk_strategy="fixed", chunk_size=200),
        RAGConfig(name="bm25_fixed200", retrieval_method="bm25", chunk_strategy="fixed", chunk_size=200),
        RAGConfig(name="hybrid_fixed200", retrieval_method="hybrid", chunk_strategy="fixed", chunk_size=200),
        RAGConfig(name="dense_sentence_chunks", retrieval_method="dense", chunk_strategy="sentence", chunk_size=200),
        RAGConfig(name="dense_small_chunks100", retrieval_method="dense", chunk_strategy="fixed", chunk_size=100),
        RAGConfig(name="dense_large_chunks400", retrieval_method="dense", chunk_strategy="fixed", chunk_size=400),
        RAGConfig(name="dense_topk1", retrieval_method="dense", chunk_strategy="fixed", chunk_size=200, top_k=1),
        RAGConfig(name="dense_topk5", retrieval_method="dense", chunk_strategy="fixed", chunk_size=200, top_k=5),
        RAGConfig(name="dense_reranked", retrieval_method="dense", chunk_strategy="fixed", chunk_size=200, top_k=3, use_reranker=True),
    ]
