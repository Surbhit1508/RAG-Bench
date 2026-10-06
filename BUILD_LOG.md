# Build Log -- RAGBench

**Purpose of this file**: so five-years-from-now I can open this
repo and reconstruct not just *what* it does, but *why every piece looks
the way it does*, including the dead ends. This is not the README
(recruiter pitch) and not `report/REPORT.md` (methodology/results). This
is the messy, honest, chronological build diary.

---

## 1. The original ask, and why this shape

RAGBench predates Hospital Ops Suite and is, in a real sense, its
methodological ancestor -- the paired-bootstrap-significance-testing
approach used everywhere in Hospital Ops Suite's `src/stats/bootstrap.py`
is a direct descendant of `src/ragbench/stats_testing.py`, written here
first.

The core decision that shapes this whole project: **don't build one RAG
pipeline and show off cherry-picked good answers -- build several RAG
configurations that each change exactly one design decision, and use
statistics, not eyeballed bar charts, to say which differences are real.**
That's an ablation study, not a demo, and it's a deliberately harder
thing to build than "here's a chatbot over some PDFs." The value of the
project is entirely in the rigor, not in the RAG pipeline itself (a
basic dense-retrieval RAG pipeline is not novel or hard to build).

Two more constraints locked in early:
- **100% local, CPU-only inference.** No API keys, no per-call cost, no
  "trust me, it worked when I ran it." Fully reproducible by anyone who
  clones the repo, including five-years-from-now you on a laptop with no
  internet access to a paid LLM API.
- **Every design axis varies exactly one thing relative to a baseline.**
  This is the actual definition of an ablation study, and it's why
  `config.py`'s `RAGConfig` dataclass and `default_ablation_grid()` exist
  as a single source of truth -- see section 8 for the confound bug this
  discipline almost didn't catch.

## 2. Why SQuAD, specifically

SQuAD v1.1 was chosen (over building a custom QA set) for one concrete
reason: every question ships with **both** (a) the exact gold paragraph
it was written against and (b) a short gold answer span. That's ground
truth for *both halves* of RAG evaluation --

- retrieval quality: did we fetch the paragraph the question came from?
- generation quality: does the generated answer match the gold span?

-- for free, with zero manual labeling. Building a custom QA benchmark
with hand-labeled gold answers would have taken longer than the entire
rest of the project and introduced its own labeling-quality questions.
This is the same reasoning that later led to picking datasets with
built-in ground truth for Hospital Ops Suite (UCI's real readmission
labels, MTSamples' real specialty labels) rather than anything requiring
manual annotation.

## 3. The corporate-network saga (read this before touching `fetch_assets.py`)

This took real, non-trivial debugging time and is worth documenting
precisely because it will happen again on the next project (it did --
see Hospital Ops Suite's own `BUILD_LOG.md` section 2, which is this
story's sequel with a twist).

**Problem 1 -- NTLM auth**: Python's `requests`/`huggingface_hub`
download stacks fail with `407 authenticationrequired` through Walmart's
corporate proxy. Windows `curl` negotiates NTLM automatically via SSPI
using the logged-in user's credentials and just works. Fix: shell out to
`curl` for every download (`fetch_assets.py`'s `_curl()` helper) instead
of fighting NTLM auth into three different HTTP libraries
(`requests`/`httpx`/`huggingface_hub` all handle auth differently).

**Problem 2 -- the Xet CDN wall**: HuggingFace's actual model weight
files redirect to an S3/Xet CDN backend (`us.aws.cdn.hf.co`) that the
corporate proxy blocks outright with a 407, even via `curl`. Two things
were tried:
1. First attempt (still in `config.py` today, `os.environ.setdefault
   ("HF_HUB_DISABLE_XET", "1")` plus `HF_HUB_ENABLE_HF_TRANSFER=0`) --
   telling `huggingface_hub`'s own downloader to avoid Xet and use plain
   HTTPS. Left in the code as defensive belt-and-suspenders even though
   it's not actually load-bearing for the final approach, since some
   `transformers`/`sentence-transformers` internals still probe
   `huggingface_hub` even when the actual bytes are loaded from a local
   directory.
2. **The fix that actually worked**: Walmart's Artifactory has a native
   HuggingFaceML remote repo (`hub-hf-release-remote`) that fetches
   model files server-side and hands them back over a domain the proxy
   allows. Metadata (the file listing) comes from `huggingface.co`'s API
   directly (that host is reachable); the actual file *bytes* come from
   Artifactory. Two different hosts for two different purposes, both
   necessary.

**A nuance worth flagging for continuity with the *next* project**:
this project only ever needed **model repos** through Artifactory
(`hub-hf-release-remote/{repo_id}/resolve/main/{file}`). Hospital Ops
Suite, built later, needed **dataset repos**, and discovered the mirror
path is different there -- `hub-huggingfaceml-release-remote` (longer
name) with a mandatory `datasets/` path segment before `repo_id`. The
shorter model-repo name 404s on dataset repos. If a future project needs
both model AND dataset files from the same mirror, expect to need both
path patterns, not one.

**The file-selection allowlist** (`_ALLOWED_SUFFIXES`,
`_SKIP_DIR_MARKERS` in `fetch_assets.py`): a HF model repo often ships
the same weights in five formats (PyTorch, safetensors, ONNX, OpenVINO,
TensorFlow, Flax, CoreML) plus training scripts and docs. Only
`.json`/`.safetensors`/`.bin`/`.txt`/`.model` files are actually needed
to call `from_pretrained()`. An **allowlist** was chosen over a
blocklist deliberately: new repos keep adding new export formats nobody
has seen yet, but the set of files actually required to load a model via
`transformers` has been stable for years. Blocklisting would mean
discovering a new junk format only after it silently got downloaded.

**Idempotent caching**: `fetch_hf_repo()` writes a `.fetched_ok` marker
file after a successful fetch and short-circuits on it next time. Model
weights are hundreds of MB+; without this, every script invocation
during development would have re-downloaded gigabytes for no reason.

## 4. Abandoning the `datasets` library entirely (not just working around it)

The original plan was to load SQuAD via `datasets.load_dataset("squad")`.
Two real problems, documented in `CHANGELOG.md`'s Fixed section:
1. Newer `datasets`/`huggingface_hub` versions require the namespaced
   `"rajpurkar/squad"` repo id instead of the old bare `"squad"` alias.
2. Even the namespaced path still goes through `huggingface_hub`'s own
   HTTP stack internally, which can't authenticate through the corporate
   proxy (same NTLM problem as section 3, but this time baked inside a
   library instead of something `curl` could route around).

Rather than keep patching around `datasets`, the fix was to **remove
the dependency entirely**: `fetch_assets.fetch_squad_dev()` downloads
the raw SQuAD JSON directly via `curl` from
`rajpurkar.github.io/SQuAD-explorer` (a plain GitHub Pages host, no
proxy drama at all), and `data_prep.py` parses the nested
`title -> paragraphs -> qas` JSON structure by hand. Net effect: one
fewer dependency in `requirements.txt`, and one fewer place where a
library upgrade could silently break the pipeline months later.

## 5. Corpus/QA construction nuances

- **Dedup by context text, not by (title, question)**: SQuAD asks many
  questions per paragraph. `_dedupe_contexts()` keys on the actual
  context string so the corpus doesn't end up with dozens of duplicate
  copies of the same paragraph just because multiple questions reference
  it.
- **Distractor documents are the actual source of retrieval difficulty**:
  `target_size = max(2000, len(seen_docs_for_qa) * 4)` pads the corpus
  with extra real SQuAD paragraphs beyond the ones any QA pair is
  actually about. Difficulty in retrieval comes from having many
  plausible-but-wrong documents to be tempted by, not from corpus size
  for its own sake -- a 150-question benchmark against a 150-document
  corpus (one doc per question) would make retrieval trivially easy and
  tell you nothing about a retriever's actual precision.
- **`N_QA_PAIRS = 150`** is a documented CPU-inference-budget tradeoff,
  not an ideal number -- it's explicitly called out in `config.py`'s
  comment as "easy to bump once pipeline is validated," and in the
  README's limitations as the reason some findings (hybrid retrieval's
  F1 gain, p=0.071) are directionally promising but not statistically
  confirmed. A wider study would rerun with more QA pairs specifically
  to narrow that one confidence interval, not to redo anything else.
- **`RANDOM_SEED = 42`** everywhere (QA sampling, bootstrap resampling)
  -- the *point* of an ablation study is that results are reproducible;
  an unseeded random shuffle would make "did config X really beat
  config Y" partly a question of which run you happened to look at.

## 6. Chunking nuances

Two chunking strategies, deliberately simple ones ("simple enough to
explain in an interview" was an explicit design filter used throughout
this project):
- **Fixed-window**: whitespace-token sliding window with overlap. Cheap,
  predictable, ignores sentence boundaries -- can and does cut sentences
  in half.
- **Sentence-aware**: greedily packs whole sentences into a window up to
  `chunk_size` tokens, never splitting a sentence. `chunk_overlap` here
  means something subtly different than in the fixed strategy: it's "how
  many trailing sentences (by token count) carry into the next chunk,"
  not "shift the window by this many tokens." Both strategies are
  parameterized with the same field names in `RAGConfig` for a uniform
  ablation grid, even though the semantics differ slightly underneath --
  documented in the chunking module's docstring specifically so nobody
  assumes `chunk_overlap` means identically the same arithmetic in both
  branches.

## 7. Retrieval nuances

- **FAISS cosine similarity via L2-normalize + inner product**:
  `faiss.normalize_L2(embeddings)` before building an `IndexFlatIP` is
  the standard trick that makes inner product equivalent to cosine
  similarity -- avoids needing a separate cosine-specific index type.
- **RetrievalIndex is built once per unique chunking, not once per
  config**: `run_experiment.py` groups the 9 ablation configs by their
  `(chunk_strategy, chunk_size, chunk_overlap)` signature before running
  anything, because building embeddings is the expensive part of the
  whole pipeline. Without this grouping, a 9-config grid where 6 configs
  share the same `fixed/200/40` chunking would recompute the identical
  embedding set 6 times for zero benefit. This one design decision is
  most of the difference between the ablation study finishing in
  ~25-30 minutes on CPU vs. taking multiple hours.
- **Hybrid retrieval pulls a wider pool (20) from each method before
  combining**: `hybrid_search(..., pool=20)` retrieves 20 candidates from
  dense AND 20 from BM25 (min-max normalized independently), then
  combines and truncates to the real `top_k`. Without the wider pool, a
  chunk that BM25 ranks #1 but dense retrieval doesn't even place in its
  naive top-3 would never get a chance to compete in the combined
  ranking -- the wider pool is what makes "hybrid" actually consider
  both signals rather than just re-ranking whatever dense already found.
- **Reranking fetches 3x the final top_k before reranking down**:
  `pipeline.answer_question()`'s `fetch_k = cfg.top_k * 3 if
  cfg.use_reranker else cfg.top_k` exists specifically so a reranker has
  real candidates to rerank *among* -- reranking only 3 candidates down
  to 3 candidates would just reorder them with no chance to surface a
  4th-or-later candidate that the cross-encoder actually prefers. This
  exact line of code is the fix for the confound bug in section 8 below.

## 8. The confound bug (worth understanding fully, not just knowing it happened)

`CHANGELOG.md` documents this tersely; here's the full story for
context. The `dense_reranked` ablation config originally changed **two**
variables relative to baseline at once: it bumped `top_k` from 3 to 5
*and* turned on the reranker. That's a direct violation of the entire
premise of an ablation study -- if `dense_reranked` had come out
significantly better (or worse) than baseline, there would have been no
way to attribute that difference to reranking specifically vs. simply
retrieving more context (which section 9's headline finding shows has
its own, separate, real effect). The fix was to hold `top_k` at 3
(matching baseline) for the reranked config, and instead give the
reranker extra *candidates to choose from* via the `fetch_k = top_k * 3`
mechanism in `pipeline.py` -- so the reranker gets real headroom to do
its job without changing the one variable the ablation grid claims to be
isolating. Only the affected config was rerun and merged back into the
results table; the other 8 were unaffected and left alone (rerunning
everything "just to be safe" would have wasted CPU time without changing
any other config's already-correct methodology).

**The general lesson**: an ablation grid's entire scientific validity
rests on each config changing exactly one thing. This is easy to state
and easy to accidentally violate once "let's also give the reranker a
fair shot with more candidates" sounds like a reasonable, harmless
tweak. It isn't harmless -- it's a second independent variable. Catch
this class of bug by explicitly diffing each config's fields against
baseline before trusting a result, not by eyeballing whether a change
"seems reasonable."

## 9. Generation and metrics nuances

- **Deterministic greedy decoding (`do_sample=False, num_beams=1`,
  temperature effectively 0)**: this is an evaluation harness, not a
  chatbot. The same config must produce the same answer every run, or
  "config A beats config B" partly becomes a question of sampling luck
  rather than a real property of the configuration.
- **The prompt explicitly instructs abstention**: `_PROMPT_TEMPLATE`
  tells the model to say "I don't know" if the answer isn't in the
  context, specifically so `is_abstention()` has something real to
  detect. Without that instruction, a small model like Flan-T5-base
  would likely just hallucinate a plausible-sounding wrong answer
  instead of abstaining, and the abstention-rate metric (used to explain
  *why* BM25's F1 drops so much -- it triples its abstention rate) would
  have nothing to measure.
- **EM/F1 use the exact standard SQuAD normalization** (lowercase, strip
  punctuation, drop articles a/an/the, collapse whitespace) specifically
  so the numbers are comparable to published SQuAD results anyone
  reading the report might already have some intuition for -- inventing
  a custom normalization would make every number un-anchored to any
  external reference point.
- **Groundedness is explicitly NOT hallucination detection**: it's a
  lexical-overlap heuristic (fraction of the answer's content words that
  literally appear in the retrieved context), chosen because it's cheap,
  fast, and fully explainable, and documented everywhere as a proxy, not
  a real NLI-based faithfulness measure. This is the same "label your
  heuristics as heuristics" discipline later applied to Hospital Ops
  Suite's NLP urgency scorer.
- **Retrieval metrics assume exactly one gold document**, which is a
  real simplification (some real corpora have multiple valid source
  documents for one question) but is what SQuAD's ground truth actually
  provides, and it keeps nDCG's IDCG term trivially 1.0 (binary
  relevance, one relevant doc) instead of needing graded relevance
  judgments nobody has for this data.

## 10. Statistics nuances (the part that makes this project not just a demo)

- **Bootstrap over t-tests, deliberately**: with ~150 QA pairs and
  metrics that are often binary/bounded (hit@k, exact match), assuming
  normality for a t-test is shaky. Bootstrap resampling makes no
  distributional assumption and, per the module's own docstring, is
  "easy to explain in an interview: resample the results with
  replacement 10,000 times, see how much the mean metric wobbles."
- **Paired resampling preserves pairing on purpose**: `paired_bootstrap
  _test()` resamples *question indices* (the same random index applied
  to both config A's and config B's results), not each config's
  differences independently. Since every config runs on the *same* 150
  questions, this is the bootstrap equivalent of a paired t-test --
  resampling independently would throw away real information (that
  question #47 being hard for config A likely also makes it hard for
  config B) and would inflate the estimated variance of the difference,
  making real effects look less significant than they are.
- **Significance is only tested against ONE baseline, not all pairs**:
  `analyze_results.significance_vs_baseline()` deliberately compares
  every config to `baseline_dense_fixed200` only, not every config to
  every other config. Testing all C(9,2)=36 pairs would both be less
  interpretable (what story does "BM25 vs sentence-chunks" even tell?)
  and would raise real multiple-comparisons concerns that a single
  ablation-grid-vs-baseline framing avoids by construction.

## 11. Ablation grid design (`config.py`)

`RAGConfig` is a frozen dataclass specifically so a config instance can
be hashed or used as a dict key without surprises (mutable dataclasses
aren't hashable by default, and nothing in this pipeline should ever
need to mutate a config after it's defined -- if you need a different
config, make a new one). `default_ablation_grid()`'s nine entries each
change exactly one field relative to `baseline_dense_fixed200`:

| Config | Varies |
|---|---|
| `baseline_dense_fixed200` | (the baseline itself) |
| `bm25_fixed200` | retrieval_method |
| `hybrid_fixed200` | retrieval_method |
| `dense_sentence_chunks` | chunk_strategy |
| `dense_small_chunks100` | chunk_size |
| `dense_large_chunks400` | chunk_size |
| `dense_topk1` | top_k |
| `dense_topk5` | top_k |
| `dense_reranked` | use_reranker (see section 8 for how this almost varied two things) |

## 12. Headline results (as shipped -- these are the numbers that matter)

| Config | Retrieval Hit@k | Gen F1 | Abstention rate |
|---|---|---|---|
| baseline (dense, top-3) | 0.820 | 0.743 | 0.053 |
| bm25 | 0.700 (sig. worse) | 0.655 (sig. worse) | 0.180 |
| **hybrid** | **0.853** | **0.796** | 0.047 |
| sentence chunks | 0.807 | 0.724 (sig. worse) | 0.060 |
| chunk size 100 | 0.800 | 0.711 | 0.040 |
| chunk size 400 | 0.813 | 0.726 | 0.067 |
| top-k=1 | 0.620 (sig. worse) | 0.589 (sig. worse) | 0.127 |
| top-k=5 | 0.833 | 0.608 (sig. worse) | 0.107 |
| reranked | 0.840 | 0.759 | 0.053 |

Four findings that survive statistical scrutiny (paired bootstrap,
10,000 resamples, p<0.05 threshold):

1. **top_k=5 hurts generation even though retrieval doesn't get worse**
   -- hit-rate is statistically indistinguishable from baseline (p=0.26)
   but F1 drops significantly (p<0.0001). A small-scale "lost in the
   middle" effect: retrieving more context doesn't help the generator if
   the generator can't use it well.
2. **Hybrid retrieval looks best on every metric but isn't provably
   better at this sample size** -- F1 gain over baseline just misses
   significance (p=0.071). Directionally promising, not proven; the
   honest call is "worth a bigger study," not "hybrid wins."
3. **BM25 alone is a clear, significant downgrade** (p=0.008 on hit
   rate) and its abstention rate more than triples -- when it retrieves
   the wrong thing, the *generator* actually notices and (correctly)
   declines to answer rather than hallucinating.
4. **Reranking gives a small, non-significant lift** while roughly
   doubling latency (cross-encoder scoring is expensive relative to a
   dot product) -- a legitimate "not worth it for this corpus/model
   combo" finding, not a failure of reranking in general.

## 13. Streamlit demo nuances

`app/streamlit_app.py` auto-selects "the best config" by reading
`results/summary.json` and picking whichever config has the highest mean
`gen_f1`, falling back to the hardcoded baseline config if the ablation
study hasn't been run yet (so the demo never crashes on a fresh clone,
it just runs a slightly worse config until you run the real experiment).
`@st.cache_resource` wraps the index-building step so navigating the app
doesn't rebuild embeddings on every interaction -- only on first load per
session.

## 14. Testing philosophy

18 tests, all pure-logic, zero network access or model downloads
required -- they run in under a second. `test_stats.py` is a good
example of the pattern used everywhere in this project (and later
copied into Hospital Ops Suite almost verbatim): verify the *statistical
machinery itself* against synthetic data with known properties (constant
values should produce a zero-width CI; two clearly-separated
distributions should be flagged significant; identical values should
never be flagged significant) rather than depending on real model
inference to validate that bootstrap resampling code is correct.

## 15. Packaging: this project IS an installable package (unlike Hospital Ops Suite)

`pyproject.toml` + `src/ragbench/` layout + `uv pip install -e .` is a
real, if lightweight, Python packaging setup -- every module is imported
as `from ragbench import config`, not via relative path hacks. This was
a deliberate choice at the time (probably worth carrying forward): it
makes `import ragbench` work identically whether you're in a test file,
a notebook, or the Streamlit app, without every entrypoint needing its
own `sys.path.insert()` hack. (The one exception, `streamlit_app.py`,
still has a `sys.path.insert()` at the top -- Streamlit's execution
model doesn't reliably respect an editable install's path resolution
the same way pytest/plain scripts do, so the belt-and-suspenders path
insert stayed in as a defensive measure.) Hospital Ops Suite later
skipped packaging entirely (`python -m src.module` everywhere, no
`pyproject.toml`) -- a reasonable simplification for a project that
never needed a Streamlit-style app importing across a package boundary
the same way.

## 16. Git/GitHub nuances (the lesson that shaped Hospital Ops Suite's push process)

The git history here is short (4 commits) but one of them is
instructive: `merge: reconcile with remote initial commit (keep our
LICENSE)`. This happened because the remote GitHub Enterprise repo
already had its own initial commit (almost certainly auto-created
LICENSE/README from the repo-creation UI) by the time this project's
local history was pushed, producing divergent histories that needed an
explicit merge instead of a clean fast-forward push. **This is exactly
why, on Hospital Ops Suite, the instruction to the user was "create the
remote repo but leave it completely empty"** -- that merge-conflict
detour here is the lived experience that turned into an explicit
guardrail on the next project. If you're setting up a new repo's remote
in the future: empty remote, always, no auto-initialized files.

## 17. Docs and public/internal separation nuances

The README's corporate-proxy war story deliberately says
"corporate-style proxy" rather than naming Walmart's proxy hostname
directly in the *public-facing* framing, while the *internal* technical
deep-dive (this file, and the code comments in `fetch_assets.py`) keeps
the real hostnames and Artifactory paths -- because this repo's intent
was to eventually also exist as a public/portfolio mirror, and internal
infrastructure hostnames don't need to travel to that audience even
though they're not actually secret. The LinkedIn blurb written for this
project followed the same rule (softened "corporate NTLM proxy" language,
included the standard "views are my own, not affiliated with my
employer" disclaimer since the post explicitly mentions working at
Walmart).

## 18. Known limitations (as documented, not forgotten)

- Groundedness metric is a lexical-overlap heuristic, not real
  NLI-based faithfulness/hallucination detection.
- Single-hop QA only (SQuAD); multi-hop retrieval (HotpotQA-style, where
  the answer requires combining facts from two+ documents) is a
  different, harder problem this study never attempts.
- 150 QA pairs keeps CPU-only iteration fast (~25-30 min for the full
  9-config grid) at the real cost of wider confidence intervals --
  hybrid retrieval's F1 improvement (p=0.071) is the clearest example of
  a result this sample size genuinely cannot confirm or deny cleanly.
  `N_QA_PAIRS` in `config.py` is the one-line change to rerun with more
  data if that CI ever needs to be tightened.
- One generation model (Flan-T5-base) -- findings about chunking and
  retrieval may not transfer identically to a larger, more
  context-robust generator; the "lost in the middle" effect in
  particular is known in the literature to be model-size-dependent.

## 19. If you're reading this before touching the code again

```bash
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

Numbers will drift slightly on rerun (library versions change, and
generation is greedy-deterministic but retrieval embeddings/model
weights could differ if a dependency version bumps) -- what shouldn't
drift without triggering a closer look is *which config wins* and
*whether a difference is significant*. If those change, something real
changed, not just noise.
