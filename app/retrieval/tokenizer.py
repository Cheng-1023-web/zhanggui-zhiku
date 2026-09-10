"""中文/英文混合分词器（jieba 优先，自研 bigram 回退）。

- 有 jieba：jieba 精确模式切中文词 + 英文/数字整词，检索精度最佳；
- 无 jieba：字符 bigram 回退（零依赖，专名召回稳定）。
接口对齐 jieba.lcut，可在不改动上层的情况下替换实现。
"""
from __future__ import annotations

import re

_EN = re.compile(r"[a-zA-Z0-9_+#.]+ ")
_CN = re.compile(r"[\u4e00-\u9fff]")

_STOP = {"的", "了", "和", "是", "在", "与", "或", "及", "对", "为", "等", "中", "有", "个",
         "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is", "are"}


def _cn_bigram(text: str) -> list[str]:
    cn_chars = [c for c in text if _CN.match(c)]
    return [c[i] + c[i + 1] for i in range(len(cn_chars) - 1)] or cn_chars


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    # 英文/数字整词
    for en in _EN.findall(text):
        low = en.lower().strip("._")
        if low:
            tokens.append(low)
    # 中文部分
    cn_text = _EN.sub(" ", text)
    try:
        import jieba  # jieba 懒加载，失败则回退 bigram
        tokens.extend(t for t in jieba.lcut(cn_text)
                      if t.strip() and _CN.search(t))
    except Exception:  # noqa: BLE001
        tokens.extend(_cn_bigram(cn_text))
    return [t for t in tokens if t and t not in _STOP]


if __name__ == "__main__":
    print(tokenize("FAISS 向量检索与 BM25 关键词检索的混合召回"))
