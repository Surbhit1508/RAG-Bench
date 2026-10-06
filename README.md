# RAGBench

**A rigorous evaluation framework for Retrieval-Augmented Generation — because "it works on my three test questions" isn't science.**

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Tests](https://img.shields.io/badge/tests-18%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Runs](https://img.shields.io/badge/inference-100%25%20local%20CPU-orange)

RAGBench answers a question most RAG demos never ask: **which of your
design choices actually matter, and how do you know?**

Instead of building one RAG pipeline and showing off cherry-picked
answers, this project builds **nine RAG configurations** that each
change exactly one design decision, runs all of them against the same
150-question benchmark, and uses **paired bootstrap significance
testing** — not eyeballed bar charts — to say which differences are
real and which are noise.

Everything runs **100% locally on CPU**. No API keys, no per-call cost,
fully reproducible by anyone who clones this repo.

---

## Table of contents

- [Headline findings](#headline-findings)
- [What's being tested](#whats-being-tested)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Project layout](#project-layout)
- [Why SQuAD / why these models](#why-squad--why-these-models)
- [A real engineering war story](#a-real-engineering-war-story-corporate-networks-vs-model-downloads)
- [Limitations](#limitations)
- [Changelog](#changelog)

---

Want the full, warts-and-all chronological build story -- every decision,
gotcha, and dead-end, written for future-you to understand five years
from now? See [`BUILD_LOG.md`](BUILD_LOG.md).

---

## Headline findings

Full write-up with statistical detail: [`report/REPORT.md`](report/REPORT.md).

| # | Finding | Evidence |
|---|---|---|
| 1 | **Retrieving more context can make the *generator* worse, even when retrieval itself doesn't get worse.** Top-5 retrieval has statistically indistinguishable hit-rate from top-3 (p=0.26), but its answer F1 drops significantly (p<0.0001). A small-scale "lost in the middle" effect. | `dense_topk5` vs baseline |
| 2 | **Hybrid retrieval looks like the best config, but we can't claim significance yet.** Highest hit rate, MRR, and F1 of any config — but the F1 gain narrowly misses p<0.05 (p=0.071) at n=150. Directionally promising, not proven. | `hybrid_fixed200` vs baseline |
| 3 | **BM25 alone is a clear, significant downgrade** from dense retrieval (p=0.008 on hit rate) and its abstention rate more than triples — when it retrieves the wrong thing, the model actually notices. | `bm25_fixed200` vs baseline |
| 4 | **Reranking gave a small, non-significant lift.** Given it roughly doubles latency, the data says it isn't worth it for this corpus/model combo. | `dense_reranked` vs baseline |

This is the actual point of the project: #2 and #4 are results a plot-only
analysis would happily oversell. Bootstrap testing keeps the claims honest.

## What's being tested

| Axis varied | Configs |
|---|---|
| Retrieval method | dense (embeddings), BM25 (sparse/lexical), hybrid (weighted combo) |
| Chunking strategy | fixed-size sliding window vs. sentence-boundary-aware |
| Chunk size | 100 / 200 / 400 tokens |
| Top-k retrieved | 1 / 3 / 5 |
| Reranking | dense retrieval + cross-encoder reranker vs. none |

Every config is scored on **both halves** of a RAG system:

- **Retrieval quality**: Hit@k, MRR, nDCG, Precision@k — did we fetch the
  paragraph the question was actually written against?
- **Generation quality**: Exact Match, F1 (SQuAD-style), abstention rate,
  and a groundedness heuristic — did the answer come from the retrieved
  context, or did the model make something up?

### Results at a glance

| Config | Retrieval Hit@k | Gen F1 | Abstention rate |
|---|---|---|---|
| baseline (dense, top-3) | 0.820 | 0.743 | 0.053 |
| bm25 | 0.700 &nbsp;▼sig | 0.655 &nbsp;▼sig | 0.180 |
| **hybrid** | **0.853** | **0.796** | 0.047 |
| sentence chunks | 0.807 | 0.724 &nbsp;▼sig | 0.060 |
| chunk size 100 | 0.800 | 0.711 | 0.040 |
| chunk size 400 | 0.813 | 0.726 | 0.067 |
| top-k=1 | 0.620 &nbsp;▼sig | 0.589 &nbsp;▼sig | 0.127 |
| top-k=5 | 0.833 | 0.608 &nbsp;▼sig | 0.107 |
| reranked | 0.840 | 0.759 | 0.053 |

▼sig = statistically significant vs. baseline, paired bootstrap p<0.05 (10,000 resamples).
See [`results/summary.json`](results/summary.json) for full 95% CIs on every metric.

![Retrieval comparison](results/retrieval_comparison.png)
![Generation comparison](results/generation_comparison.png)

## Architecture

```
                     ┌─────────────────────┐
   SQuAD v1.1  ──▶   │   data_prep.py       │──▶  corpus.jsonl + qa_pairs.jsonl
                     └─────────────────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │   chunking.py         │  (fixed-window | sentence-aware)
                     └─────────────────────┘
                                │
                                ▼
   ┌─────────────────────────────────────────────────┐
   │                 retrieval.py                       │
   │   dense (MiniLM + FAISS) │ BM25 │ hybrid │ rerank   │
   └─────────────────────────────────────────────────┘
                                │  top-k chunks
                                ▼
                     ┌─────────────────────┐
                     │   generation.py       │  Flan-T5-base, greedy decode
                     └─────────────────────┘
                                │  answer
                                ▼
          ┌──────────────────────────────────────┐
          │  metrics_retrieval.py + metrics_generation.py │
          └──────────────────────────────────────┘
                                │
                                ▼
          ┌──────────────────────────────────────┐
          │   stats_testing.py — paired bootstrap    │
          │   significance testing across 9 configs  │
          └──────────────────────────────────────┘
                                │
                                ▼
              results/  (CSV, plots, summary.json)
              report/REPORT.md  (full write-up)
              app/streamlit_app.py  (interactive demo)
```

## Quickstart

```bash
git clone <this-repo>
cd ragbench

uv venv
uv pip install -r requirements.txt
uv pip install -e .

python -m ragbench.data_prep        # builds data/corpus.jsonl + data/qa_pairs.jsonl
python -m ragbench.run_experiment   # runs all 9 configs x 150 questions (~25-30 min on CPU)
python -m ragbench.analyze_results  # produces results/summary.json + comparison plots

streamlit run app/streamlit_app.py  # interactive demo of the best config found
pytest tests/                       # fast unit tests, no downloads required (<1s)
```

Model weights and the SQuAD JSON are fetched on first run and cached
locally (`models/`, `data/`) — nothing re-downloads on subsequent runs.

## Project layout

```
src/ragbench/
  config.py            # all paths, model names, the ablation grid (single source of truth)
  fetch_assets.py       # downloads dataset + model weights
  data_prep.py           # builds the corpus + QA benchmark from SQuAD
  chunking.py             # fixed-window and sentence-aware chunking strategies
  retrieval.py             # dense (FAISS) / BM25 / hybrid retrieval + reranking
  generation.py             # Flan-T5 answer generation
  pipeline.py                # wires retrieval + generation together for one question
  metrics_retrieval.py        # Hit@k, MRR, nDCG, Precision@k
  metrics_generation.py        # EM, F1, abstention detection, groundedness heuristic
  stats_testing.py               # bootstrap CIs + paired significance tests
  run_experiment.py               # orchestrates the full ablation grid
  analyze_results.py               # aggregates results, runs significance tests, makes plots

app/streamlit_app.py    # interactive demo using the best config found by the ablation study
notebooks/                # executed walkthrough notebook with real outputs
tests/                     # unit tests for chunking/metrics/stats (no model downloads needed)
results/                    # results.csv, summary.json, comparison plots (generated)
report/                      # written ablation report with findings (generated)
BUILD_LOG.md                  # chronological build diary: every decision, gotcha, and dead-end
```

## Why SQuAD / why these models

**SQuAD**: every question ships with the exact gold paragraph it was
written against *and* a short gold answer span — ground truth for both
retrieval and generation quality, with zero manual labeling.

**Models** (all fetched once, run fully offline afterward):
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` — the
  standard lightweight baseline for dense retrieval.
- **Generation**: `google/flan-t5-base` — instruction-tuned
  encoder-decoder, small enough to run a 1,350-generation ablation
  sweep on a laptop CPU in well under an hour, deterministic (greedy
  decoding) for reproducible comparisons.
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` — standard
  cross-encoder reranking baseline.

## A real engineering war story: corporate networks vs. model downloads

Worth calling out because it's a legitimate production concern, not
just a "worked on my machine" footnote.

This project was built behind a corporate proxy requiring NTLM
authentication. Python's `httpx`/`requests` stacks (used internally by
`datasets` and `huggingface_hub`) don't negotiate NTLM automatically —
every download failed with a `407 authenticationrequired`. Windows
`curl`, on the other hand, negotiates NTLM transparently via SSPI using
the logged-in user's credentials, and worked immediately.

`fetch_assets.py` works around this by shelling out to `curl` for every
external download (SQuAD JSON, model weight files), landing everything
in plain local folders that `transformers`/`sentence-transformers` load
directly via `from_pretrained(local_dir)` — no HF Hub cache internals,
no fighting NTLM auth into three different HTTP libraries.

That still wasn't the whole story: HuggingFace's newer **Xet CDN**
backend (`*.xethub.hf.co`) turned out to be unreachable through the
proxy at all — even `curl` got a `407` there, confirmed via `curl -v`
tracing the redirect chain. The fix was routing model weight downloads
through an **internal Artifactory HuggingFaceML mirror repo** instead,
which fetches the bytes server-side and hands them back over a domain
the proxy actually allows. Not every model repo is available through
that mirror (several popular ones 404 — `gpt2`, `Qwen2.5`, `Phi-3`,
`TinyLlama` all did), which is exactly why `google/flan-t5-base` — a
somewhat less trendy but perfectly capable choice — ended up as the
generation model here, after probing several ungated alternatives.

This is the kind of "why did this data pipeline silently fail" incident
that shows up in real ML infra work, so it's documented here instead of
hidden. See `fetch_assets.py` for the working implementation.

## Limitations

- **Groundedness metric is a lexical-overlap heuristic**, not true
  faithfulness/hallucination detection. A production system would want
  NLI-based entailment scoring or an LLM-judge here.
- **Single-hop QA only** — SQuAD questions are answerable from one
  paragraph. Multi-hop retrieval (HotpotQA-style) is a different,
  harder problem this study doesn't cover.
- **150 questions** keeps CPU-only iteration fast, at the cost of wider
  confidence intervals than a several-thousand-question eval would
  give — the bootstrap CIs reported are honest about that uncertainty.
  Bump `N_QA_PAIRS` in `config.py` and rerun to tighten them.
- **One generation model.** Findings about chunking/retrieval may not
  transfer identically to a larger, more context-robust generator.

## Changelog

See [`CHANGELOG.md`](CHANGELOG.md).

## License

MIT — see [`LICENSE`](LICENSE).
