"""BM25 关键词检索（rank_bm25 后端，自研分词器）。"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from ..ingest.chunker import Chunk
from .tokenizer import tokenize


class BM25Retriever:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks: list[Chunk] = chunks or []
        self._bm25: BM25Okapi | None = None
        if self.chunks:
            self._bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] > 0]

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Retriever":
        with open(Path(path) / "bm25.pkl", "rb") as f:
            return pickle.load(f)
