"""FAISS 向量库：IndexFlatIP（内积 = 余弦，向量已 L2 归一化），支持持久化与 numpy 回退。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..ingest.chunker import Chunk


class FaissVectorStore:
    def __init__(self, dim: int):
        import faiss  # 延迟导入，便于无 faiss 环境回退
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]):
        self.chunks.extend(chunks)
        self.index.add(np.asarray(vectors, dtype="float32"))

    def search(self, query_vec: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        k = min(top_k, self.index.ntotal)
        scores, ids = self.index.search(np.asarray([query_vec], dtype="float32"), k)
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]

    def save(self, path: str | Path):
        import faiss
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        meta = [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "source": c.source,
                 "heading_path": c.heading_path, "text": c.text, "index": c.index,
                 "metadata": c.metadata} for c in self.chunks]
        (path / "chunks.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, dim: int) -> "FaissVectorStore":
        import faiss
        path = Path(path)
        store = cls.__new__(cls)
        store.dim = dim
        store.index = faiss.read_index(str(path / "index.faiss"))
        raw = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        store.chunks = [Chunk(**r) for r in raw]
        return store


class NumpyVectorStore(FaissVectorStore):
    """无 faiss 时的暴力检索回退（数据量 < 1e5 时性能可接受）。"""

    def __init__(self, dim: int):
        self.dim = dim
        self.index = None
        self._matrix: np.ndarray | None = None
        self.chunks = []

    def add(self, chunks, vectors):
        self.chunks.extend(chunks)
        mat = np.asarray(vectors, dtype="float32")
        self._matrix = mat if self._matrix is None else np.vstack([self._matrix, mat])

    def search(self, query_vec, top_k):
        if self._matrix is None:
            return []
        q = np.asarray([query_vec], dtype="float32")
        sims = (self._matrix @ q.T).ravel()
        k = min(top_k, len(sims))
        ids = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in ids]

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "index.npy", self._matrix)
        import json as _json
        meta = [{"chunk_id": c.chunk_id, "doc_id": c.doc_id, "source": c.source,
                 "heading_path": c.heading_path, "text": c.text, "index": c.index,
                 "metadata": c.metadata} for c in self.chunks]
        (path / "chunks.json").write_text(_json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path, dim):
        path = Path(path)
        store = cls.__new__(cls)
        store.dim = dim
        store.index = None
        store._matrix = np.load(path / "index.npy")
        import json as _json
        raw = _json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        from ..ingest.chunker import Chunk as C
        store.chunks = [C(**r) for r in raw]
        return store


def build_vector_store(cfg: dict, dim: int):
    try:
        import faiss  # noqa: F401
        return FaissVectorStore(dim)
    except ImportError:
        print("[vector_store] faiss 不可用，回退 numpy 暴力检索")
        return NumpyVectorStore(dim)


def load_vector_store(path: str | Path, dim: int):
    try:
        import faiss  # noqa: F401
        return FaissVectorStore.load(path, dim)
    except ImportError:
        return NumpyVectorStore.load(path, dim)
