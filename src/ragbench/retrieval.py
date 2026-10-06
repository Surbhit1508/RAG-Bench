"""Retrieval backends: dense (embeddings + FAISS), BM25 (sparse), and a
weighted hybrid of the two, plus an optional cross-encoder reranker.

This is the module the ablation study leans on hardest -- retrieval
method is one of the axes we vary and measure.
"""
from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from ragbench import config
from ragbench.chunking import chunk_document
from ragbench.fetch_assets import local_model_dir


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def build_chunks(corpus: list[dict], chunk_strategy: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in corpus:
        pieces = chunk_document(doc["text"], chunk_strategy, chunk_size, chunk_overlap)
        for i, piece in enumerate(pieces):
            chunks.append(Chunk(chunk_id=f"{doc['doc_id']}_c{i}", doc_id=doc["doc_id"], text=piece))
    return chunks


_embedder_cache: dict[str, SentenceTransformer] = {}
_reranker_cache: dict[str, CrossEncoder] = {}


def get_embedder(model_name: str = config.EMBEDDING_MODEL) -> SentenceTransformer:
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = SentenceTransformer(str(local_model_dir(model_name)))
    return _embedder_cache[model_name]


def get_reranker(model_name: str = config.RERANKER_MODEL) -> CrossEncoder:
    if model_name not in _reranker_cache:
        _reranker_cache[model_name] = CrossEncoder(str(local_model_dir(model_name)))
    return _reranker_cache[model_name]


class RetrievalIndex:
    """Holds one chunking of the corpus plus its dense + sparse indices.

    Building embeddings is the expensive part, so this object is built
    ONCE per unique (chunk_strategy, chunk_size, chunk_overlap) triple
    and reused across every RAGConfig that shares that chunking --
    otherwise a 9-config ablation grid would recompute the same
    embeddings 6+ times for no reason (see run_experiment.py).
    """

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._dense_embeddings: np.ndarray | None = None
        self._faiss_index: faiss.Index | None = None
        self._bm25: BM25Okapi | None = None

    def _ensure_dense(self) -> None:
        if self._faiss_index is not None:
            return
        embedder = get_embedder()
        texts = [c.text for c in self.chunks]
        embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)  # so inner product == cosine similarity
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        self._dense_embeddings = embeddings
        self._faiss_index = index

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        tokenized = [_tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)

    def dense_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        self._ensure_dense()
        embedder = get_embedder()
        q_emb = embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, idxs = self._faiss_index.search(q_emb, top_k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]

    def bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        self._ensure_bm25()
        scores = self._bm25.get_scores(_tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_idx]

    @staticmethod
    def _min_max_normalize(pairs: list[tuple[int, float]]) -> dict[int, float]:
        if not pairs:
            return {}
        scores = np.array([s for _, s in pairs])
        lo, hi = scores.min(), scores.max()
        if hi - lo < 1e-9:
            return {i: 1.0 for i, _ in pairs}
        return {i: (s - lo) / (hi - lo) for i, s in pairs}

    def hybrid_search(self, query: str, top_k: int, alpha: float, pool: int = 20) -> list[tuple[int, float]]:
        """Weighted combination of normalized dense + BM25 scores.

        `alpha` = weight on the dense score (1-alpha goes to BM25).
        We pull a wider `pool` from each method before combining so a
        chunk that's #1 on BM25 but outside dense's naive top_k still
        gets a fair shot at the final ranking.
        """
        dense_pairs = self.dense_search(query, pool)
        bm25_pairs = self.bm25_search(query, pool)
        dense_norm = self._min_max_normalize(dense_pairs)
        bm25_norm = self._min_max_normalize(bm25_pairs)

        all_ids = set(dense_norm) | set(bm25_norm)
        combined = {i: alpha * dense_norm.get(i, 0.0) + (1 - alpha) * bm25_norm.get(i, 0.0) for i in all_ids}
        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return ranked

    def search(self, query: str, method: str, top_k: int, alpha: float = 0.5) -> list[Chunk]:
        if method == "dense":
            pairs = self.dense_search(query, top_k)
        elif method == "bm25":
            pairs = self.bm25_search(query, top_k)
        elif method == "hybrid":
            pairs = self.hybrid_search(query, top_k, alpha)
        else:
            raise ValueError(f"Unknown retrieval_method: {method!r}")
        return [self.chunks[i] for i, _ in pairs]

    def rerank(self, query: str, candidates: list[Chunk]) -> list[Chunk]:
        if not candidates:
            return candidates
        reranker = get_reranker()
        pairs = [(query, c.text) for c in candidates]
        scores = reranker.predict(pairs)
        order = np.argsort(scores)[::-1]
        return [candidates[i] for i in order]
