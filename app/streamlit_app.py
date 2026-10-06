"""RAGBench interactive demo: ask a question, see retrieval + answer for
the best-performing config from the ablation study (falls back to the
baseline config if results haven't been analyzed yet).

Run with: `streamlit run app/streamlit_app.py`
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json

import streamlit as st

from ragbench import config
from ragbench.config import default_ablation_grid
from ragbench.data_prep import load_corpus
from ragbench.pipeline import answer_question
from ragbench.retrieval import RetrievalIndex, build_chunks

st.set_page_config(page_title="RAGBench Demo", layout="wide")
st.title("RAGBench — Retrieval-Augmented QA Demo")
st.caption(
    "A small, fully local RAG pipeline (Flan-T5 + MiniLM embeddings) over a SQuAD-derived "
    "Wikipedia corpus. This is the interactive companion to the RAGBench ablation study -- "
    "see `report/REPORT.md` for the full rigor (statistical significance tests, retrieval "
    "metrics, ablations over chunking/retrieval-method/top-k/reranking)."
)


@st.cache_resource(show_spinner="Loading best config + building index (first run only)...")
def load_best_config_and_index():
    best_name = None
    if config.SUMMARY_JSON.exists():
        summary = json.loads(config.SUMMARY_JSON.read_text(encoding="utf-8"))
        best_name = max(summary, key=lambda name: summary[name]["gen_f1"]["mean"])

    grid = {c.name: c for c in default_ablation_grid()}
    cfg = grid.get(best_name) or grid["baseline_dense_fixed200"]

    corpus = load_corpus()
    chunks = build_chunks(corpus, cfg.chunk_strategy, cfg.chunk_size, cfg.chunk_overlap)
    index = RetrievalIndex(chunks)
    return cfg, index


cfg, index = load_best_config_and_index()

with st.sidebar:
    st.subheader("Active config")
    st.json(cfg.as_dict())
    st.caption("Automatically picked as the highest-F1 config from results/summary.json, "
               "or the baseline if you haven't run the ablation study yet.")

question = st.text_input("Ask a question about... whatever's in a random SQuAD Wikipedia paragraph:",
                          placeholder="e.g. What is usually formed when oxygen binds to a metal?")

if question:
    with st.spinner("Retrieving + generating..."):
        chunks, answer = answer_question(index, cfg, question)

    st.subheader("Answer")
    st.success(answer)

    st.subheader(f"Retrieved context (top {len(chunks)})")
    for i, c in enumerate(chunks, start=1):
        with st.expander(f"[{i}] from {c.doc_id}"):
            st.write(c.text)
