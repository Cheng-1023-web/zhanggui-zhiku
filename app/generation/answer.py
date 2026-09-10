"""答案生成：上下文构建、引用约束、无依据拒答（幻觉抑制）。"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..retrieval.hybrid import Candidate
from .llm import build_llm


@dataclass
class Answer:
    text: str
    citations: list[dict] = field(default_factory=list)  # 引用的 chunk 元信息
    refused: bool = False
    graph_lines: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)


def build_prompt(question: str, candidates: list[Candidate],
                 graph_lines: list[str], max_context_chars: int = 3000) -> str:
    parts, total = [], 0
    for i, c in enumerate(candidates, 1):
        block = f"[{i}] （{c.chunk.metadata.get('heading', '')} | {c.chunk.source}）\n{c.chunk.text}"
        if total + len(block) > max_context_chars:
            break
        parts.append(block)
        total += len(block)
    ctx = "\n\n".join(parts)
    graph = "\n".join(graph_lines)
    return (
        "你是企业知识库问答助手。请严格基于以下知识库上下文回答问题。\n"
        "要求：\n"
        "1. 每条结论末尾必须标注依据编号，如 [1]；无依据的结论不得输出；\n"
        "2. 若上下文不足以回答，直接回答「知识库中未找到足够依据」；\n"
        "3. 不要编造上下文之外的信息。\n\n"
        f"知识库上下文：\n{ctx}\n"
        + (f"\n知识图谱关联（可辅助组织答案，同样需挂引用）：\n{graph}\n" if graph else "")
        + f"\n用户问题：{question}"
    )


def check_refusal(query: str, candidates: list[Candidate], threshold: float) -> bool:
    """拒答判定：候选为空，或最高向量相似度低于阈值 → 拒绝作答。"""
    if not candidates:
        return True
    best_vec = max((c.scores.get("vector_best", 0) for c in candidates), default=0)
    return best_vec < threshold


def generate_answer(query: str, candidates: list[Candidate], llm_cfg: dict,
                    answer_cfg: dict, graph_lines: list[str] | None = None) -> Answer:
    graph_lines = graph_lines or []
    if check_refusal(query, candidates, answer_cfg.get("refusal_threshold", 0.30)):
        return Answer(
            text="知识库中未找到足够依据，无法可靠回答该问题。"
                 "建议补充相关文档或换个问法（检索增强拒答，用于抑制幻觉）。",
            refused=True, graph_lines=graph_lines,
        )
    prompt = build_prompt(query, candidates, graph_lines,
                          llm_cfg.get("max_context_chars", 3000))
    llm = build_llm(llm_cfg)
    text = llm.chat(prompt)
    citations = [
        {
            "n": i + 1,
            "chunk_id": c.chunk.chunk_id,
            "source": c.chunk.source,
            "heading": c.chunk.metadata.get("heading", ""),
            "snippet": c.chunk.text[:120],
            "score": round(c.scores.get("rerank", c.scores.get("rrf", 0)), 4),
        }
        for i, c in enumerate(candidates)
    ]
    return Answer(text=text, citations=citations, graph_lines=graph_lines)
