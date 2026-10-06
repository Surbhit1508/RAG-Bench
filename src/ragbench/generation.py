"""Answer generation via a small local instruction-tuned model (Flan-T5).

Deterministic generation (greedy decoding, temperature effectively 0)
because this is an evaluation harness, not a chatbot -- we want the
same config to produce the same answer every run so ablation results
are reproducible and comparisons are fair.
"""
from __future__ import annotations

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ragbench import config
from ragbench.fetch_assets import local_model_dir

_PROMPT_TEMPLATE = (
    "Answer the question using only the context below. "
    "If the answer is not contained in the context, say \"I don't know\".\n\n"
    "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
)

_model = None
_tokenizer = None


def _load():
    global _model, _tokenizer
    if _model is None:
        model_dir = str(local_model_dir(config.GENERATION_MODEL))
        _tokenizer = AutoTokenizer.from_pretrained(model_dir)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        _model.eval()
    return _model, _tokenizer


def build_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    return _PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(question: str, context_chunks: list[str], max_new_tokens: int = config.MAX_NEW_TOKENS) -> str:
    model, tokenizer = _load()
    prompt = build_prompt(question, context_chunks)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,   # greedy -- deterministic, reproducible eval
        num_beams=1,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
