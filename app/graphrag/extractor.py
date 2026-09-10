"""GraphRAG：实体/关系抽取 → 图存储 → 查询期子图上下文增强。

抽取策略（可插拔）：
- 规则抽取（默认，零依赖）：基于句式模板「X 是 Y」「X 包括 A、B、C」
  「X 支持 Y」「X 适用于 Y」等中文业务文档高频句式；
- LLM 抽取（可选）：配置 generation.api_base 后可用结构化 Prompt 抽取三元组。
"""
from __future__ import annotations

import json
import re

from ..ingest.chunker import Chunk
from .graph_store import BaseGraphStore

_TRIPLE_PATTERNS = [
    re.compile(r"(?P<s>[\u4e00-\u9fffA-Za-z0-9_\-]{2,20})(?:支持|提供|内置|包含)(?:了)?(?P<o>[\u4e00-\u9fffA-Za-z0-9_\-]{2,20})(?:功能|能力|模块)?"),
    re.compile(r"(?P<s>[\u4e00-\u9fffA-Za-z0-9_\-]{2,20})(?:是|为|指)(?:一?种)?(?P<o>[\u4e00-\u9fffA-Za-z0-9_\-]{2,20})"),
    re.compile(r"(?P<s>[\u4e00-\u9fffA-Za-z0-9_\-]{2,20})(?:适用于|面向)(?P<o>[\u4e00-\u9fffA-Za-z0-9_\-]{2,20})"),
    re.compile(r"(?P<s>[\u4e00-\u9fffA-Za-z0-9_\-]{2,20})(?:分为|包括|包括以下)(?P<o>[^。；\n]{2,60})"),
    re.compile(r"(?P<s>[\u4e00-\u9fffA-Za-z0-9_\-]{2,20})(?:依赖|需要|配合)(?P<o>[\u4e00-\u9fffA-Za-z0-9_\-]{2,20})"),
]
_VERBS = ["支持", "是", "适用于", "包括", "依赖"]


class TripleExtractor:
    def extract(self, chunks: list[Chunk]) -> list[tuple[str, str, str]]:
        triples: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for chunk in chunks:
            for sent in re.split(r"[。；\n]", chunk.text):
                for pat in _TRIPLE_PATTERNS:
                    for m in pat.finditer(sent):
                        s, o = m.group("s").strip(), m.group("o").strip()
                        verb = next((v for v in _VERBS if v in sent), "关联")
                        if s and o and s != o and (s, verb, o) not in seen:
                            seen.add((s, verb, o))
                            triples.append((s, verb, o))
        return triples

    def extract_with_llm(self, chunks: list[Chunk], llm) -> list[tuple[str, str, str]]:
        """LLM 结构化三元组抽取（配置了 api_base 时启用）。"""
        triples: list[tuple[str, str, str]] = []
        prompt = ("从以下文本抽取实体关系三元组，仅输出 JSON 数组，"
                  "每项格式 {\"s\":实体,\"r\":关系,\"o\":实体}，不要输出其他内容：\n\n{text}")
        for chunk in chunks:
            try:
                out = llm.chat(prompt.format(text=chunk.text[:1500]))
                for item in json.loads(out):
                    triples.append((item["s"], item["r"], item["o"]))
            except Exception:  # noqa: BLE001 单条失败不阻断
                continue
        return triples


def build_graph(chunks: list[Chunk], store: BaseGraphStore,
                extractor: TripleExtractor | None = None) -> int:
    extractor = extractor or TripleExtractor()
    triples = extractor.extract(chunks)
    for s, r, o in triples:
        store.add_triple(s, r, o)
    return len(triples)


def graph_context(query: str, store: BaseGraphStore, max_edges: int = 8) -> list[str]:
    """查询期：命中实体的一跳子图 → 文本化上下文，供答案生成引用。"""
    hits = store.match_entities(query)
    lines: list[str] = []
    for ent in hits:
        for s, r, o in store.neighbors(ent, max_edges=max_edges):
            lines.append(f"{s} --[{r}]--> {o}")
    return lines
