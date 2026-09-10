"""文档解析：PDF / Word / Markdown / TXT → 结构化文本（保留标题层级）。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedDoc:
    """解析后的文档：以 (标题路径, 正文) 为单元组织。"""
    doc_id: str
    source: str                      # 原始文件路径
    sections: list["Section"] = field(default_factory=list)


@dataclass
class Section:
    """一个标题层级下的正文段落，heading_path 如 ['产品手册', '计费说明']。"""
    heading_path: list[str]
    text: str


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")


def parse_file(path: str | Path) -> ParsedDoc:
    path = Path(path)
    suffix = path.suffix.lower()
    doc_id = path.stem
    if suffix == ".pdf":
        text = _parse_pdf(path)
    elif suffix in {".docx", ".doc"}:
        text = _parse_docx(path)
    else:  # .md / .txt 及其他文本
        text = path.read_text(encoding="utf-8", errors="ignore")
    return _parse_markdown_text(doc_id, str(path), text)


def _parse_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _parse_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    lines: list[str] = []
    for para in doc.paragraphs:
        style = (para.style.name or "").lower() if para.style is not None else ""
        text = para.text.strip()
        if not text:
            continue
        # Word 标题样式转 Markdown 标题，统一走 md 解析
        if style.startswith("heading"):
            try:
                level = int(style.replace("heading", "").strip() or "1")
            except ValueError:
                level = 1
            lines.append("#" * level + " " + text)
        else:
            lines.append(text)
    return "\n".join(lines)


def _parse_markdown_text(doc_id: str, source: str, text: str) -> ParsedDoc:
    """将 Markdown 文本切为 Section 列表，保留标题路径（结构化语义切分的第一步）。"""
    doc = ParsedDoc(doc_id=doc_id, source=source)
    heading_stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush():
        if buf and any(l.strip() for l in buf):
            path = [h for _, h in heading_stack] or [doc_id]
            doc.sections.append(Section(heading_path=list(path), text="\n".join(buf).strip()))
        buf.clear()

    for line in text.splitlines():
        m = _MD_HEADING.match(line.strip())
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            # 维护标题栈：弹出层级 >= 当前层级的标题
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
        else:
            buf.append(line)
    flush()
    if not doc.sections and text.strip():
        doc.sections.append(Section(heading_path=[doc_id], text=text.strip()))
    return doc
