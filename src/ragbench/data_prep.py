"""Build the RAGBench corpus + evaluation QA set from SQuAD v1.1.

Why SQuAD: every question comes with (a) the exact gold paragraph it was
written against and (b) a short gold answer span. That gives us ground
truth for BOTH halves of RAG evaluation for free:
  - retrieval quality: did we fetch the paragraph the question came from?
  - generation quality: does the generated answer match the gold answer?

We parse the raw SQuAD JSON directly (via fetch_assets.fetch_squad_dev)
rather than going through the `datasets` library, since `datasets`
insists on talking to the HuggingFace Hub API which our corporate proxy
can't authenticate against (see fetch_assets.py for the full story).

Run as a script: `python -m ragbench.data_prep`
"""
from __future__ import annotations

import json
import random

from ragbench import config
from ragbench.fetch_assets import fetch_squad_dev


def _iter_squad_rows(squad_json: dict):
    """Flatten SQuAD's nested title -> paragraphs -> qas structure."""
    for article in squad_json["data"]:
        title = article["title"]
        for para in article["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                if qa.get("is_impossible"):
                    continue
                answers = [a["text"] for a in qa["answers"]]
                if not answers:
                    continue
                yield {"title": title, "context": context, "question": qa["question"], "answers": answers}


def _dedupe_contexts(rows) -> dict[str, dict]:
    """Collapse SQuAD rows down to unique (title, context) documents.

    SQuAD asks many questions per paragraph, so naively treating every
    row as its own document would massively inflate the corpus with
    duplicates. We key on the context text itself.
    """
    docs: dict[str, dict] = {}
    for row in rows:
        key = row["context"]
        if key not in docs:
            doc_id = f"doc_{len(docs):05d}"
            docs[key] = {"doc_id": doc_id, "title": row["title"], "text": row["context"]}
    return docs


def build_corpus_and_qa(n_qa_pairs: int = config.N_QA_PAIRS, seed: int = config.RANDOM_SEED) -> None:
    squad_path = fetch_squad_dev()
    print(f"Parsing SQuAD dev set from {squad_path} ...")
    with open(squad_path, "r", encoding="utf-8") as f:
        squad_json = json.load(f)

    rows = list(_iter_squad_rows(squad_json))
    docs_by_context = _dedupe_contexts(rows)
    context_to_doc_id = {ctx: d["doc_id"] for ctx, d in docs_by_context.items()}

    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)

    qa_pairs = []
    seen_docs_for_qa = set()
    for i in order:
        if len(qa_pairs) >= n_qa_pairs:
            break
        row = rows[i]
        doc_id = context_to_doc_id[row["context"]]
        qa_pairs.append(
            {
                "qa_id": f"qa_{len(qa_pairs):05d}",
                "question": row["question"],
                "gold_answer": row["answers"][0],
                "gold_answers_all": row["answers"],
                "gold_doc_id": doc_id,
            }
        )
        seen_docs_for_qa.add(doc_id)

    # Keep the docs referenced by our QA sample, plus enough extra
    # "distractor" docs that retrieval is a real challenge rather than
    # a trivial 1-doc lookup. Difficulty comes from having many
    # plausible-but-wrong documents, not from corpus size for its own sake.
    all_docs = list(docs_by_context.values())
    rng.shuffle(all_docs)
    corpus_docs = {d["doc_id"]: d for d in all_docs if d["doc_id"] in seen_docs_for_qa}
    target_size = max(2000, len(seen_docs_for_qa) * 4)
    for d in all_docs:
        if len(corpus_docs) >= target_size:
            break
        corpus_docs.setdefault(d["doc_id"], d)

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CORPUS_PATH, "w", encoding="utf-8") as f:
        for d in corpus_docs.values():
            f.write(json.dumps(d) + "\n")

    with open(config.QA_PATH, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa) + "\n")

    print(f"Wrote {len(corpus_docs)} corpus docs -> {config.CORPUS_PATH}")
    print(f"Wrote {len(qa_pairs)} QA pairs -> {config.QA_PATH}")


def load_corpus() -> list[dict]:
    with open(config.CORPUS_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_qa_pairs() -> list[dict]:
    with open(config.QA_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


if __name__ == "__main__":
    build_corpus_and_qa()
