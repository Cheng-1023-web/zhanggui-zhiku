"""应用配置加载与索引构建/加载入口。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .ingest.chunker import build_chunks
from .ingest.parsers import parse_file
from .retrieval.bm25 import BM25Retriever
from .retrieval.embedder import build_embedder
from .retrieval.hybrid import HybridRetriever
from .retrieval.vector_store import build_vector_store, load_vector_store


@dataclass
class App:
    cfg: dict
    retriever: HybridRetriever | None = None
    graph_store: object | None = None
    stats: dict = field(default_factory=dict)

    # ---------- 索引构建 ----------
    def ingest(self) -> dict:
        icfg = self.cfg
        docs_dir = Path(icfg["ingest"]["docs_dir"])
        index_dir = Path(icfg["ingest"]["index_dir"])
        docs = [parse_file(p) for p in sorted(docs_dir.glob("*"))
                if p.suffix.lower() in {".md", ".txt", ".pdf", ".docx"}]
        chunks = build_chunks(
            docs,
            chunk_size=icfg["chunk"]["chunk_size"],
            chunk_overlap=icfg["chunk"]["chunk_overlap"],
            min_chunk_size=icfg["chunk"]["min_chunk_size"],
        )
        embedder = build_embedder(icfg["embedding"])
        vectors = embedder.encode([c.text for c in chunks])

        store = build_vector_store(icfg, embedder.dim)
        store.add(chunks, vectors)
        store.save(index_dir)

        bm25 = BM25Retriever(chunks)
        bm25.save(index_dir)
        (index_dir / "meta.json").write_text(
            yaml.safe_dump({"dim": embedder.dim,
                            "embedding_provider": icfg["embedding"]["provider"]},
                           allow_unicode=True), encoding="utf-8")

        # GraphRAG：三元组抽取入图
        from .graphrag.extractor import TripleExtractor, build_graph
        from .graphrag.graph_store import build_graph_store
        self.graph_store = build_graph_store(icfg["graphrag"])
        n_triples = build_graph(chunks, self.graph_store, TripleExtractor())
        if hasattr(self.graph_store, "save"):
            self.graph_store.save(index_dir / "graph.pkl")

        self.stats = {"docs": len(docs), "chunks": len(chunks),
                      "triples": n_triples, "dim": embedder.dim}
        self._build_retriever(store, bm25, embedder)
        return self.stats

    def _build_retriever(self, store, bm25, embedder):
        rc = self.cfg["retrieval"]
        self.retriever = HybridRetriever(
            store, bm25, embedder,
            vector_top_k=rc["vector_top_k"], bm25_top_k=rc["bm25_top_k"],
            rrf_k=rc["rrf_k"])

    # ---------- 索引加载 ----------
    def load(self) -> "App":
        index_dir = Path(self.cfg["ingest"]["index_dir"])
        meta = yaml.safe_load((index_dir / "meta.json").read_text(encoding="utf-8"))
        embedder = build_embedder({**self.cfg["embedding"], "provider": meta["embedding_provider"],
                                   "dim": meta["dim"]})
        store = load_vector_store(index_dir, meta["dim"])
        bm25 = BM25Retriever.load(index_dir)
        self._build_retriever(store, bm25, embedder)
        gp = index_dir / "graph.pkl"
        if gp.exists():
            from .graphrag.graph_store import NetworkXGraphStore
            self.graph_store = NetworkXGraphStore.load(gp)
        return self


def load_config(path: str | Path = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
