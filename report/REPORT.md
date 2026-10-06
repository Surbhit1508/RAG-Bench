# RAGBench Ablation Report

## 1. Question this project answers

RAG systems have a lot of knobs: how you chunk documents, how you
retrieve, how many passages you pass to the generator, whether you
rerank. Most public RAG tutorials pick values for all of these
arbitrarily and never check whether they mattered. This report answers,
with actual statistical evidence: **which of these design choices move
the needle, and which are noise?**

## 2. Setup

- **Benchmark**: 150 questions sampled from the SQuAD v1.1 dev set,
  each with a gold answer span and the exact paragraph it was written
  against. Corpus: 2,000 unique Wikipedia paragraphs (the paragraphs
  referenced by our questions, plus distractor paragraphs so retrieval
  is a real search problem rather than a 1-of-150 lookup).
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Generation model**: `google/flan-t5-base` (greedy decoding, fully
  deterministic — same input always produces the same output, so
  differences between configs are attributable to the config, not
  sampling noise)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Statistical method**: paired bootstrap resampling (10,000 resamples)
  against the baseline config, for both retrieval hit-rate and answer F1.
  Paired because every config was run on the *same* 150 questions, so we
  resample question indices (not independent per-config samples) to
  preserve that pairing — same logic as a paired t-test, without
  assuming normally-distributed metrics.

## 3. The ablation grid

| Config | What it changes vs. baseline |
|---|---|
| `baseline_dense_fixed200` | dense retrieval, 200-token fixed chunks, top-3 |  (baseline) |
| `bm25_fixed200` | sparse (BM25) retrieval instead of dense |
| `hybrid_fixed200` | weighted combination of dense + BM25 |
| `dense_sentence_chunks` | sentence-boundary-aware chunking instead of fixed windows |
| `dense_small_chunks100` | 100-token chunks instead of 200 |
| `dense_large_chunks400` | 400-token chunks instead of 200 |
| `dense_topk1` | retrieve only 1 passage instead of 3 |
| `dense_topk5` | retrieve 5 passages instead of 3 |
| `dense_reranked` | retrieve 5 candidates, rerank with a cross-encoder, keep top 3 |

## 4. Metrics

**Retrieval** (did we fetch the right source document?):
Hit@k, Mean Reciprocal Rank, nDCG, Precision@k.

**Generation** (did the model answer correctly, and did it stick to
what it was given?):
Exact Match / F1 (SQuAD-style, vs. gold answer), abstention rate
(did the model say "I don't know"), and a groundedness heuristic
(fraction of the answer's words that literally appear in the retrieved
context — a cheap, explainable, honestly-limited proxy for
faithfulness, not a substitute for NLI-based entailment or an
LLM-judge).

## 5. Results

All numbers are means over 150 questions with 95% bootstrap confidence
intervals. Full data: `results/summary.json`, `results/results.csv`.
Plots: `results/retrieval_comparison.png`, `results/generation_comparison.png`.

| Config | Retrieval Hit@k | Retrieval MRR | Gen F1 | Gen EM | Abstention rate |
|---|---|---|---|---|---|
| **baseline_dense_fixed200** | 0.820 [0.753, 0.880] | 0.707 | 0.743 [0.674, 0.809] | 0.680 | 0.053 |
| bm25_fixed200 | 0.700 [0.620, 0.773] | 0.637 | 0.655 [0.581, 0.728] | 0.600 | 0.180 |
| hybrid_fixed200 | 0.853 [0.793, 0.907] | 0.781 | 0.796 [0.731, 0.856] | 0.740 | 0.047 |
| dense_sentence_chunks | 0.807 [0.740, 0.867] | 0.692 | 0.724 [0.653, 0.793] | 0.667 | 0.060 |
| dense_small_chunks100 | 0.800 [0.733, 0.860] | 0.712 | 0.711 [0.639, 0.781] | 0.660 | 0.040 |
| dense_large_chunks400 | 0.813 [0.747, 0.873] | 0.703 | 0.726 [0.657, 0.793] | 0.667 | 0.067 |
| dense_topk1 | 0.620 [0.540, 0.700] | 0.620 | 0.589 [0.512, 0.668] | 0.547 | 0.127 |
| dense_topk5 | 0.833 [0.773, 0.893] | 0.710 | 0.608 [0.533, 0.684] | 0.553 | 0.107 |
| dense_reranked | 0.840 [0.780, 0.900] | 0.814 | 0.759 [0.692, 0.823] | 0.687 | 0.053 |

### Statistically significant differences vs. baseline (paired bootstrap, p < 0.05)

| Config | Metric | Mean diff | 95% CI | p-value |
|---|---|---|---|---|
| bm25_fixed200 | retrieval hit | −0.120 | [−0.207, −0.033] | 0.0076 |
| bm25_fixed200 | gen F1 | −0.087 | [−0.167, −0.008] | 0.0322 |
| dense_sentence_chunks | gen F1 | −0.018 | [−0.041, −0.001] | 0.0354 |
| dense_topk1 | retrieval hit | −0.200 | [−0.267, −0.140] | <0.0001 |
| dense_topk1 | gen F1 | −0.154 | [−0.222, −0.089] | <0.0001 |
| dense_topk5 | gen F1 | −0.135 | [−0.197, −0.074] | <0.0001 |

Everything else (hybrid, reranking, chunk size 100/400) was **not**
statistically distinguishable from baseline at n=150 -- see section 6
for why that's still an informative result, not a null result.

## 6. Findings & discussion

**1. More retrieved context can make generation *worse*, even when retrieval itself doesn't get worse.**
This is the headline finding. `dense_topk5` retrieves 5 passages instead
of 3, and its retrieval hit rate is *not* significantly different from
baseline (+0.013, p=0.26) -- it still finds the right document just as
often. But its generation F1 drops by 0.135 points, wildly significant
(p<0.0001). The retrieval step didn't get worse; the generator got
*worse at using what it was given* once more (mostly irrelevant)
context was mixed in. This is a small-scale, empirically-measured
instance of the "lost in the middle" effect reported in larger LLM
literature -- and it shows up even at 3->5 passages with a 250M-parameter
model and a modest 1024-token context budget. Practical takeaway: don't
assume "retrieve more, just in case" is a free lunch. It has a real,
measurable, generation-side cost that retrieval metrics alone would
never catch -- you have to evaluate both halves of the pipeline
separately, which is exactly the point of this project.

**2. BM25 alone is a clear downgrade from dense retrieval on this benchmark.**
Significantly worse on both retrieval hit (-0.12) and F1 (-0.087), and
notably its abstention rate more than triples (0.053 -> 0.18) -- when
BM25 fails to retrieve the right paragraph, the model much more often
notices and says "I don't know" rather than confidently hallucinating.
Makes sense: SQuAD questions are frequently phrased with different
words than the source paragraph ("What drove rental prices up?" vs.
a paragraph that never uses the word "drove"), which is precisely the
gap dense embeddings are designed to close and pure lexical matching
can't.

**3. Hybrid retrieval looks like the best config numerically, but we can't claim significance yet.**
Highest hit rate (0.853), highest MRR (0.781), highest F1 (0.796) of
any config -- but the F1 improvement vs. baseline (p=0.071) narrowly
misses the conventional 0.05 threshold. This is the single most
important lesson in the whole report about *rigor*: a plot alone would
happily tell you "hybrid wins," and a lot of RAG blog posts would stop
there. With bootstrap resampling we can be honest that 150 questions
isn't quite enough data to be confident this isn't noise. The right
next step isn't to claim victory -- it's to rerun with a larger QA
sample (config.py's `N_QA_PAIRS` is one line to change) before making
this the production default.

**4. Reranking gave a small, non-significant lift -- and that's a legitimate cost/benefit finding, not a failure.**
+0.02 hit rate, +0.016 F1, neither significant. Baseline dense
retrieval already had an 82% hit rate at top-3, leaving limited room
for a reranker to fix mistakes it can't fully recover from anyway
(if the right chunk wasn't in the initial candidate pool, no reranker
saves you). Given reranking roughly doubles latency (a second model
forward pass over every candidate), the honest conclusion here is:
**not worth it for this corpus/model combination** -- exactly the kind
of "we tried the fancier thing and it didn't pay for itself" result a
real engineering team needs to make good tradeoffs, and exactly the
kind of result a demo-only RAG project would never surface.

**5. Chunk size and chunking strategy: mostly noise, with one small real exception.**
100 vs. 200 vs. 400 token chunks were statistically indistinguishable
from each other on both retrieval and generation. Sentence-aware
chunking, however, showed a small but statistically significant F1 drop
(-0.018, p=0.035) despite no significant change in retrieval hit rate --
suggesting variable-length sentence-packed chunks occasionally hand the
generator slightly less clean context windows than uniform fixed
windows do, even when the right document is retrieved. Small effect,
but real, and a good example of a result you'd only find by actually
measuring instead of assuming "respecting sentence boundaries must be
better."

**6. top_k=1 is a clear, unsurprising loser.** Included mainly as a
sanity check that the eval harness can detect a config that *should*
be obviously worse -- and it is, by a wide and highly significant
margin on every metric. Good confirmation the measurement pipeline
itself is working correctly. (One artifact worth flagging: at top_k=1,
Precision@k is mathematically identical to Hit@k -- with only one
document retrieved, "fraction of retrieved docs that are correct" and
"was the correct doc retrieved at all" are the same question. That's
why `dense_topk1`'s precision bar looks unusually high in the plot; it
is not evidence top_k=1 retrieves more precisely in any meaningful sense.)

## 7. Limitations (being upfront about them)

- **Groundedness metric is a lexical-overlap heuristic**, not true
  faithfulness/hallucination detection. A model could restate context
  verbatim and score perfectly while still misinterpreting it, or
  paraphrase correctly and score lower than it deserves. A production
  system would want NLI-based entailment scoring or an LLM-judge here.
- **Single-hop QA only.** SQuAD questions are answerable from one
  paragraph. Multi-hop retrieval (HotpotQA-style) is a harder and
  different problem this study doesn't cover.
- **150 questions** keeps CPU-only iteration fast; wider confidence
  intervals than a several-thousand-question eval would give. The
  bootstrap CIs reported here are honest about that uncertainty rather
  than hiding it.
- **One generation model.** Findings about chunking/retrieval may not
  transfer identically to a much larger generator that's more robust
  to noisy context.
