"""评测指标：Recall@K、MRR、答案引用覆盖率（LLM-as-a-Judge 接口预留）。"""
from __future__ import annotations

import json
from pathlib import Path

from ..retrieval.hybrid import HybridRetriever
from ..retrieval.reranker import build_reranker


def recall_at_k(ranked_chunk_ids: list[str], gold_ids: list[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    hit = len(set(ranked_chunk_ids[:k]) & set(gold_ids))
    return hit / len(set(gold_ids))


def mrr(ranked_chunk_ids: list[str], gold_ids: list[str]) -> float:
    gold = set(gold_ids)
    for i, cid in enumerate(ranked_chunk_ids, start=1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def run_retrieval_eval(retriever: HybridRetriever, rerank_cfg: dict,
                       eval_set_path: str | Path, k: int = 5) -> dict:
    eval_set = json.loads(Path(eval_set_path).read_text(encoding="utf-8"))
    reranker = build_reranker(rerank_cfg)
    recalls, mrrs, rerank_recalls, rows = [], [], [], []
    for item in eval_set:
        cands = retriever.retrieve(item["question"])
        pre_ids = [c.chunk.chunk_id for c in cands]
        r_cands = reranker.rerank(item["question"], cands, top_k=k)
        post_ids = [c.chunk.chunk_id for c in r_cands]
        gold = item["gold_chunk_ids"]
        r1 = recall_at_k(pre_ids, gold, k)
        r2 = recall_at_k(post_ids, gold, k)
        m = mrr(post_ids, gold)
        recalls.append(r1); rerank_recalls.append(r2); mrrs.append(m)
        rows.append({"question": item["question"], f"recall@{k}_pre": round(r1, 3),
                     f"recall@{k}_post": round(r2, 3), "mrr": round(m, 3)})
    n = len(eval_set) or 1
    return {
        "n_queries": len(eval_set),
        f"recall@{k}_pre_rerank": round(sum(recalls) / n, 4),
        f"recall@{k}_post_rerank": round(sum(rerank_recalls) / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
        "detail": rows,
    }


def judge_answer_relevance(question: str, answer_text: str, llm) -> float | None:
    """LLM-as-a-Judge 答案相关性评分（1-5）。需配置真实 LLM，未配置返回 None。"""
    prompt = (
        "请对以下问答对的相关性打 1-5 分（5 为完全切题且依据充分），只输出数字。\n\n"
        f"问题：{question}\n回答：{answer_text}"
    )
    try:
        out = llm.chat(prompt)
        return float(out.strip().rstrip("分"))
    except Exception:  # noqa: BLE001
        return None
