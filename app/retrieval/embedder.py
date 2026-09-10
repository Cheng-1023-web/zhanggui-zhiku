"""Embedding 层：BGE（可选）+ 零依赖哈希回退。

- provider="bge"：加载 BAAI/bge-large-zh-v1.5（需 sentence-transformers），
  语义质量高，但需要下载约 1.3GB 模型权重；
- provider="hash"：字符 bigram 哈希 + TF 加权的确定性向量（本项目默认），
  零下载、可复现，足以支撑检索链路跑通与评测。
"""
from __future__ import annotations

import hashlib
import math
import re

from .tokenizer import tokenize


class BaseEmbedder:
    dim: int

    def encode(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


class HashEmbedder(BaseEmbedder):
    """哈希嵌入：token → 桶 +1，频率加权，L2 归一化（余弦相似度可用）。"""

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _bucket(self, token: str) -> int:
        return int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "big") % self.dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        vecs = []
        for text in texts:
            v = [0.0] * self.dim
            toks = tokenize(text)
            for t in toks:
                v[self._bucket(t)] += 1.0
                # 一阶哈希冲突缓解：符号位 token 哈希决定
                sign = 1 if int.from_bytes(hashlib.md5(("s" + t).encode()).digest()[:1], "big") % 2 else -1
                v[self._bucket("s" + t)] += sign * 0.5
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            vecs.append([x / norm for x in v])
        return vecs


class BGEEmbedder(BaseEmbedder):
    """BGE 中文嵌入（sentence-transformers 后端）。"""

    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._instruction = "为这个句子生成表示以用于检索相关文章："

    def encode(self, texts: list[str]) -> list[list[float]]:
        emb = self.model.encode([self._instruction + t for t in texts],
                                normalize_embeddings=True, show_progress_bar=False)
        return [e.tolist() for e in emb]


def build_embedder(cfg: dict) -> BaseEmbedder:
    provider = cfg.get("provider", "hash")
    if provider == "bge":
        try:
            return BGEEmbedder(cfg.get("bge_model", "BAAI/bge-large-zh-v1.5"))
        except Exception as e:  # noqa: BLE001 模型缺失时回退
            print(f"[embedder] BGE 不可用（{e}），回退到 hash 嵌入")
    return HashEmbedder(cfg.get("dim", 1024))
