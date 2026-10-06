# Changelog

All notable changes to RAGBench are documented here.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-08-25

Initial release: a full rigorous RAG ablation study, end to end.

### Added
- **Project scaffold**: `uv`-managed venv, `pyproject.toml` packaging, central
  `config.py` as single source of truth for paths/models/the ablation grid.
- **Data pipeline** (`data_prep.py`, `fetch_assets.py`): builds a 2,000-document
  corpus + 150-question QA benchmark from the SQuAD v1.1 dev set, with gold
  answer spans and gold source documents for ground-truth retrieval + generation
  evaluation.
- **Chunking strategies** (`chunking.py`): fixed-size sliding window with
  overlap, and sentence-boundary-aware packing.
- **Retrieval backends** (`retrieval.py`): dense (MiniLM embeddings + FAISS
  flat index), BM25 (sparse/lexical), and a weighted hybrid of the two, plus
  optional cross-encoder reranking.
- **Generation** (`generation.py`): local Flan-T5-base inference, deterministic
  greedy decoding for reproducible ablation comparisons.
- **Retrieval metrics** (`metrics_retrieval.py`): Hit@k, Mean Reciprocal Rank,
  nDCG, Precision@k.
- **Generation metrics** (`metrics_generation.py`): SQuAD-style Exact Match
  and F1, abstention detection, and a lexical-overlap groundedness heuristic.
- **Statistical rigor** (`stats_testing.py`): bootstrap confidence intervals
  and paired bootstrap significance testing between any two configs, run
  against every config vs. baseline.
- **Full ablation experiment runner** (`run_experiment.py`): 9 configs spanning
  retrieval method, chunk size, chunk strategy, top-k, and reranking, with
  chunking-index reuse across configs that share the same chunking (avoids
  redundant embedding computation).
- **Analysis pipeline** (`analyze_results.py`): per-config summary stats with
  bootstrap CIs, significance tests vs. baseline, and comparison plots.
- **Streamlit demo app** (`app/streamlit_app.py`): interactive QA over the
  best-performing config found by the ablation study.
- **Executed walkthrough notebook** (`notebooks/01_pipeline_walkthrough.ipynb`)
  with real outputs on live examples.
- **18 unit tests** covering chunking, retrieval/generation metrics, and the
  statistical testing module — all pure-logic, no model downloads required.
- **Full written report** (`report/REPORT.md`) with methodology, results
  table, and per-finding discussion.
- **Corporate-network workaround** (`fetch_assets.py`): downloads shell out to
  `curl` instead of `httpx`/`requests` to work around NTLM proxy auth, and
  route model weight fetches through an internal Artifactory HuggingFaceML
  mirror to bypass HuggingFace's Xet CDN backend being unreachable through the
  proxy entirely. Documented in the README as a real infra debugging story.

### Fixed
- `dense_reranked` ablation config originally changed **two** variables at
  once vs. baseline (top_k 3→5 *and* enabling the reranker), which would have
  confounded the reranking-effect measurement. Fixed to isolate reranking as
  the sole variable (top_k held at 3, matching baseline) before the affected
  config was re-run and merged back into the results table — the other 8
  configs were unaffected and not re-run.
- `datasets.load_dataset("squad", ...)` failing on newer `datasets`/
  `huggingface_hub` versions (requires namespaced `"rajpurkar/squad"`, and
  even that path couldn't authenticate through the corporate proxy) — replaced
  with direct parsing of the raw SQuAD JSON, removing the `datasets` library
  dependency entirely for this project.
- Windows console `UnicodeEncodeError` on non-ASCII corpus text — fixed by
  running with `PYTHONIOENCODING=utf-8`.

### Known limitations (see README for full discussion)
- Groundedness metric is a lexical-overlap heuristic, not true
  NLI-based faithfulness/hallucination detection.
- Single-hop QA only (SQuAD); multi-hop retrieval not evaluated.
- 150-question benchmark keeps CPU-only iteration fast at the cost of wider
  confidence intervals — `hybrid_fixed200`'s F1 improvement over baseline
  (p=0.071) is a good example of a result that's promising but not yet
  statistically confirmed at this sample size.
