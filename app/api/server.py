"""FastAPI 服务：/api/ask（问答）、/api/search（仅检索）、/api/stats。"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import App, load_config
from ..pipeline.nodes import run_query
from ..retrieval.reranker import build_reranker

_app: App | None = None


def get_app_instance() -> App:
    global _app
    if _app is None:
        _app = App(cfg=load_config()).load()
    return _app


class AskRequest(BaseModel):
    query: str


def create_app() -> FastAPI:
    app = FastAPI(title="掌柜智库 RAG API", version="1.0.0")
    inst = get_app_instance()

    @app.get("/")
    def index():
        return FileResponse(Path(__file__).resolve().parents[2] / "web" / "index.html")

    @app.post("/api/ask")
    def ask(req: AskRequest):
        state = run_query(inst.retriever, build_reranker(inst.cfg.get("rerank", {})),
                          inst.cfg["generation"], inst.cfg["answer"],
                          inst.graph_store, query=req.query)
        ans = state["answer"]
        return {
            "answer": ans.text,
            "refused": ans.refused,
            "citations": ans.citations,
            "graph_lines": ans.graph_lines,
            "timings_ms": state["timings"],
        }

    @app.post("/api/search")
    def search(req: AskRequest):
        cands = inst.retriever.retrieve(req.query)
        reranked = build_reranker(inst.cfg.get("rerank", {})).rerank(
            req.query, cands, top_k=inst.cfg["retrieval"]["rerank_top_k"])
        return [{"chunk_id": c.chunk.chunk_id, "heading": c.chunk.metadata.get("heading", ""),
                 "snippet": c.chunk.text[:160],
                 "scores": {k: round(v, 4) for k, v in c.scores.items()}}
                for c in reranked]

    @app.get("/api/stats")
    def stats():
        s = dict(inst.stats or {})
        if inst.graph_store is not None and hasattr(inst.graph_store, "stats"):
            s["graph"] = inst.graph_store.stats()
        return s

    return app
