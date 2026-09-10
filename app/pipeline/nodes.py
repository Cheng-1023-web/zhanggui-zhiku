"""RAG Pipeline 节点与编排：retrieve → rerank → (refuse | generate)。

状态流（LangGraph 语义）：
  retrieve ──▶ rerank ──▶ guard ──(拒答)──▶ END
                          │
                       (有依据)
                          ▼
                       generate ──▶ END
"""
from __future__ import annotations

import time

from ..generation.answer import generate_answer
from ..graphrag.extractor import graph_context
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.reranker import build_reranker
from .graph import END, START, StateGraph


def build_rag_pipeline(retriever: HybridRetriever, reranker, llm_cfg: dict,
                       answer_cfg: dict, graph_store=None) -> "StateGraph":
    def node_retrieve(state: dict):
        t0 = time.time()
        state["candidates"] = retriever.retrieve(state["query"])
        state["timings"]["retrieve_ms"] = round((time.time() - t0) * 1000, 1)

    def node_rerank(state: dict):
        t0 = time.time()
        state["candidates"] = reranker.rerank(
            state["query"], state["candidates"],
            top_k=llm_cfg.get("rerank_top_k", 5))
        state["timings"]["rerank_ms"] = round((time.time() - t0) * 1000, 1)

    def node_generate(state: dict):
        t0 = time.time()
        glines: list[str] = []
        if graph_store is not None:
            glines = graph_context(state["query"], graph_store)
        ans = generate_answer(state["query"], state["candidates"],
                              llm_cfg, answer_cfg, glines)
        state["answer"] = ans
        state["timings"]["generate_ms"] = round((time.time() - t0) * 1000, 1)

    def route_after_rerank(state: dict) -> str:
        # 拒答判定放在 generate 内部完成（需结合阈值与向量分），统一走 generate
        return "generate"

    g = StateGraph()
    g.add_node("retrieve", node_retrieve)
    g.add_node("rerank", node_rerank)
    g.add_node("generate", node_generate)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", END)
    return g.compile()


def run_query(retriever: HybridRetriever, reranker, llm_cfg: dict,
              answer_cfg: dict, graph_store=None, query: str = "") -> dict:
    state = {"query": query, "candidates": [], "answer": None, "timings": {}}
    pipeline = build_rag_pipeline(retriever, reranker, llm_cfg, answer_cfg, graph_store)
    pipeline.invoke(state)
    return state
