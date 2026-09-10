"""混合检索（多路召回）：FAISS 向量检索 + BM25 关键词检索 → RRF 倒数排序融合。"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..ingest.chunker import Chunk
from .bm25 import BM25Retriever
from .embedder import BaseEmbedder
from .vector_store import FaissVectorStore


@dataclass
class Candidate:
    chunk: Chunk
    scores: dict = field(default_factory=dict)   # {"vector": s, "bm25": s, "rrf": r, "rerank": r}
    rank: dict = field(default_factory=dict)     # 各路排名，用于调试输出


def rrf_fuse(*ranked_lists: list[tuple[Chunk, float]], k: int = 60) -> dict[str, Candidate]:
    """RRF: score(d) = Σ 1/(k + rank_i(d))。k=60 抑制头部排名噪声。"""
    pool: dict[str, Candidate] = {}
    for li, ranked in enumerate(ranked_lists):
        for rank, (chunk, raw) in enumerate(ranked, start=1):
            key = chunk.chunk_id
            if key not in pool:
                pool[key] = Candidate(chunk=chunk)
            pool[key].scores[f"raw_{li}"] = raw
            pool[key].scores["rrf"] = pool[key].scores.get("rrf", 0.0) + 1.0 / (k + rank)
            pool[key].rank[f"list_{li}"] = rank
    return pool


class HybridRetriever:
    def __init__(self, vector_store: FaissVectorStore, bm25: BM25Retriever,
                 embedder: BaseEmbedder, vector_top_k: int = 20,
                 bm25_top_k: int = 20, rrf_k: int = 60):
        self.vs = vector_store
        self.bm25 = bm25
        self.embedder = embedder
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.rrf_k = rrf_k

    def retrieve(self, query: str) -> list[Candidate]:
        q_vec = self.embedder.encode_one(query)
        vec_hits = self.vs.search(q_vec, self.vector_top_k)
        bm25_hits = self.bm25.search(query, self.bm25_top_k)
        pool = rrf_fuse(vec_hits, bm25_hits, k=self.rrf_k)
        pool["__stats__"] = None  # 占位防误用
        pool.pop("__stats__")
        # 记录向量最高分，供拒答阈值判断
        if vec_hits:
            best = max(pool.values(), key=lambda c: c.chunk.chunk_id == vec_hits[0][0].chunk_id)
            for c in pool.values():
                if c.chunk.chunk_id == vec_hits[0][0].chunk_id:
                    c.scores["vector_best"] = vec_hits[0][1]
        return sorted(pool.values(), key=lambda c: -c.scores["rrf"])
