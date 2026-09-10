"""CLI：ingest（建库）/ ask（问答）/ eval（评测）/ serve（启动 API）。

用法：
  python -m app.cli ingest
  python -m app.cli ask "企业版的计费方式是什么"
  python -m app.cli eval
  python -m app.cli serve
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import App, load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser("zhanggui-zhiku")
    ap.add_argument("cmd", choices=["ingest", "ask", "eval", "serve"])
    ap.add_argument("-q", "--query", default="")
    ap.add_argument("-c", "--config", default="config.yaml")
    args = ap.parse_args()

    app = App(cfg=load_config(args.config))

    if args.cmd == "ingest":
        print(json.dumps(app.ingest(), ensure_ascii=False, indent=2))
    elif args.cmd == "ask":
        app.load()
        from app.pipeline.nodes import run_query
        from app.retrieval.reranker import build_reranker
        state = run_query(app.retriever, build_reranker(app.cfg.get("rerank", {})),
                          app.cfg["generation"], app.cfg["answer"],
                          app.graph_store, query=args.query)
        ans = state["answer"]
        print(json.dumps({
            "answer": ans.text,
            "refused": ans.refused,
            "citations": ans.citations,
            "graph_lines": ans.graph_lines,
            "timings_ms": state["timings"],
        }, ensure_ascii=False, indent=2))
    elif args.cmd == "eval":
        app.load()
        from app.evaluation.metrics import run_retrieval_eval
        result = run_retrieval_eval(app.retriever, app.cfg.get("rerank", {}),
                                    "app/evaluation/eval_set.json",
                                    k=app.cfg["retrieval"]["rerank_top_k"])
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.cmd == "serve":
        import uvicorn
        from app.api.server import create_app
        uvicorn.run(create_app(), host=app.cfg["api"]["host"],
                    port=app.cfg["api"]["port"])


if __name__ == "__main__":
    main()
