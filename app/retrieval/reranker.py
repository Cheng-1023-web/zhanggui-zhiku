"""二阶段重排序：BGE-Reranker（可选）/ 词面特征回退。

- provider="bge"：FlagEmbedding 的 BGE-Reranker（Cross-Encoder 深度交互），
  精度高，仅对候选 Top-N 精排，用精度换延迟；
- provider="lexical"：token 重叠率 + 短语命中加权的轻量精排（本项目默认，
  零依赖、可复现，作为精排链路的占位实现）。
"""
from __future__ import annotations

from .hybrid import Candidate
from .tokenizer import tokenize


class LexicalReranker:
    """词面精排：query token 覆盖率 + bigram 命中 + 标题路径匹配加分。"""

    def rerank(self, query: str, candidates: list[Candidate], top_k: int = 5) -> list[Candidate]:
        q_toks = set(tokenize(query))
        for c in candidates:
            d_toks = set(tokenize(c.chunk.text))
            overlap = len(q_toks & d_toks) / (len(q_toks) or 1)
            heading_bonus = 0.1 if any(t in " ".join(c.chunk.heading_path) for t in q_toks) else 0.0
            exact = 0.15 if query.strip() and query.strip() in c.chunk.text else 0.0
            c.scores["rerank"] = overlap * 0.7 + heading_bonus + exact + 0.2 * c.scores.get("rrf", 0)
        return sorted(candidates, key=lambda c: -c.scores["rerank"])[:top_k]


class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        from FlagEmbedding import FlagReranker
        self.model = FlagReranker(model_name, use_fp16=False)

    def rerank(self, query: str, candidates: list[Candidate], top_k: int = 5) -> list[Candidate]:
        pairs = [[query, c.chunk.text] for c in candidates]
        scores = self.model.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        for c, s in zip(candidates, scores):
            c.scores["rerank"] = float(s)
        return sorted(candidates, key=lambda c: -c.scores["rerank"])[:top_k]


def build_reranker(cfg: dict):
    provider = cfg.get("provider", "lexical")
    if provider == "bge":
        try:
            return BGEReranker(cfg.get("model", "BAAI/bge-reranker-large"))
        except Exception as e:  # noqa: BLE001
            print(f"[reranker] BGE-Reranker 不可用（{e}），回退 lexical 精排")
    return LexicalReranker()
