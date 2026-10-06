"""Run the full ablation grid over the QA set and save results.

Groups RAGConfigs by their (chunk_strategy, chunk_size, chunk_overlap)
signature so we build each distinct RetrievalIndex (the expensive part --
embedding every chunk) exactly once, no matter how many configs share
that chunking. Run as: `python -m ragbench.run_experiment`
"""
from __future__ import annotations

import time
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from ragbench import config
from ragbench.config import RAGConfig, default_ablation_grid
from ragbench.data_prep import load_corpus, load_qa_pairs
from ragbench.pipeline import run_single
from ragbench.retrieval import RetrievalIndex, build_chunks


def _chunking_key(cfg: RAGConfig) -> tuple:
    return (cfg.chunk_strategy, cfg.chunk_size, cfg.chunk_overlap)


def run_all(configs: list[RAGConfig] | None = None, qa_limit: int | None = None) -> pd.DataFrame:
    configs = configs or default_ablation_grid()
    corpus = load_corpus()
    qa_pairs = load_qa_pairs()
    if qa_limit:
        qa_pairs = qa_pairs[:qa_limit]

    by_chunking: dict[tuple, list[RAGConfig]] = defaultdict(list)
    for cfg in configs:
        by_chunking[_chunking_key(cfg)].append(cfg)

    all_records = []
    for chunking_key, cfgs_sharing_index in by_chunking.items():
        strategy, size, overlap = chunking_key
        print(f"\n=== Building index for chunking={strategy}, size={size}, overlap={overlap} "
              f"(shared by {len(cfgs_sharing_index)} config(s)) ===")
        t0 = time.time()
        chunks = build_chunks(corpus, strategy, size, overlap)
        index = RetrievalIndex(chunks)
        index._ensure_dense()  # build once, up front, so tqdm timing below reflects generation cost
        index._ensure_bm25()
        print(f"  {len(chunks)} chunks indexed in {time.time() - t0:.1f}s")

        for cfg in cfgs_sharing_index:
            print(f"\n--- Running config: {cfg.name} ---")
            for qa in tqdm(qa_pairs, desc=cfg.name):
                record = run_single(index, cfg, qa)
                all_records.append(record)

    df = pd.DataFrame(all_records)
    config.RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_CSV, index=False)
    print(f"\nWrote {len(df)} result rows -> {config.RESULTS_CSV}")
    return df


if __name__ == "__main__":
    run_all()
