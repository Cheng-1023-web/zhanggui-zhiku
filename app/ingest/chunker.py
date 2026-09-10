"""文本切分：结构化语义切分（按标题/段落边界）+ 递归字符切分相结合。

策略：
1. ParsedDoc 的每个 Section 先按段落边界聚合；
2. 超过 chunk_size 的长段落，用 RecursiveCharacterTextSplitter 的
   递归分隔符策略（\n\n → \n → 。→ ；→ ，→ 空格 → 硬切）二次切分；
3. 相邻 chunk 保留 overlap 重叠，避免语义被切断。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .parsers import ParsedDoc


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    heading_path: list[str]
    text: str
    index: int = 0
    metadata: dict = field(default_factory=dict)


_SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""]


class RecursiveCharacterTextSplitter:
    """LangChain RecursiveCharacterTextSplitter 的最小实现（中文分隔符增强版）。"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        assert chunk_overlap < chunk_size, "overlap 必须小于 chunk_size"
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text] if text else []
        chunks: list[str] = []
        for seg in self._recursive_split(text, 0):
            self._pack_with_overlap(seg, chunks)
        return [c.strip() for c in chunks if c.strip()]

    def _recursive_split(self, text: str, depth: int) -> list[str]:
        if depth >= len(_SEPARATORS):
            return [text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
        sep = _SEPARATORS[depth]
        if sep == "":
            return [text[i:i + self.chunk_size]
                    for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]
        parts = [p for p in text.split(sep) if p.strip()]
        if len(parts) == 1:
            return self._recursive_split(text, depth + 1)
        out: list[str] = []
        for i, part in enumerate(parts):
            piece = part + (sep if sep in "。\n" and i < len(parts) - 1 else "")
            out.extend(self._recursive_split(piece, depth + 1) if len(piece) > self.chunk_size else [piece])
        return out

    def _pack_with_overlap(self, seg: str, chunks: list[str]):
        """将分段按 chunk_size 打包，并保留 chunk_overlap 尾部重叠。"""
        start = 0
        while start < len(seg):
            end = min(start + self.chunk_size, len(seg))
            piece = seg[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(seg):
                break
            start = max(end - self.chunk_overlap, start + 1)


def build_chunks(docs: list[ParsedDoc], chunk_size: int = 500,
                 chunk_overlap: int = 100, min_chunk_size: int = 50) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    for doc in docs:
        idx = 0
        for sec in doc.sections:
            # 结构化切分：段落边界优先聚合
            paragraphs = [p.strip() for p in sec.text.split("\n") if p.strip()]
            buf = ""
            for para in paragraphs:
                if len(para) > chunk_size:
                    if buf:
                        chunks.append(_make(doc, sec, buf, idx)); idx += 1; buf = ""
                    for sub in splitter.split_text(para):
                        chunks.append(_make(doc, sec, sub, idx)); idx += 1
                elif len(buf) + len(para) + 1 <= chunk_size:
                    buf = f"{buf}\n{para}".strip()
                else:
                    chunks.append(_make(doc, sec, buf, idx)); idx += 1
                    buf = para
            if buf:
                chunks.append(_make(doc, sec, buf, idx)); idx += 1
    # 过滤过短的 chunk
    return [c for c in chunks if len(c.text) >= min_chunk_size or c.index == 0]


def _make(doc: ParsedDoc, sec, text: str, index: int) -> Chunk:
    return Chunk(
        chunk_id=f"{doc.doc_id}#c{index}",
        doc_id=doc.doc_id,
        source=doc.source,
        heading_path=list(sec.heading_path),
        text=text.strip(),
        index=index,
        metadata={"heading": " > ".join(sec.heading_path)},
    )
