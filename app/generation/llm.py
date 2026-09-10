"""LLM 客户端：OpenAI 兼容接口 + 离线回退。

- 配置 generation.api_base / api_key 后走真实 LLM（任何 OpenAI 兼容服务）；
- 未配置时用 ExtractiveFallback：抽取式生成，保证项目离线可完整跑通。
"""
from __future__ import annotations

import json
import urllib.request


class OpenAICompatLLM:
    def __init__(self, api_base: str, api_key: str, model: str, temperature: float = 0.1):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def chat(self, prompt: str, system: str = "") -> str:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]


class ExtractiveFallback:
    """离线抽取式生成：按句子与 query 的 token 重叠度排序拼装答案。

    每句都带来源标注，天然满足「每条结论挂引用」的要求。
    """

    def chat(self, prompt: str, system: str = "") -> str:
        # prompt 结构约定见 answer.build_prompt：--- 上下文 --- 之后是问题
        try:
            ctx_block, question = prompt.rsplit("用户问题：", 1)
        except ValueError:
            return "（生成器配置缺失）"
        ctx = ctx_block.split("知识库上下文：")[-1]
        # 解析 [n] 开头的证据段
        import re
        evidences: list[tuple[int, str]] = []
        for block in re.split(r"\n(?=\[\d+])", ctx.strip()):
            m = re.match(r"\[(\d+)]\s*(.+)", block, re.S)
            if m:
                evidences.append((int(m.group(1)), m.group(2).strip()))
        from ..retrieval.tokenizer import tokenize
        q_toks = set(tokenize(question))
        scored = []
        for n, text in evidences:
            for sent in re.split(r"(?<=[。！？；])", text):
                sent = sent.strip()
                if not sent:
                    continue
                s_toks = set(tokenize(sent))
                score = len(q_toks & s_toks) / (len(q_toks) or 1)
                scored.append((score, n, sent))
        scored.sort(key=lambda x: -x[0])
        picked, used = [], set()
        for score, n, sent in scored:
            if score <= 0 or len(picked) >= 4 or n in used and len(picked) >= 2:
                continue
            picked.append(f"{sent}[{n}]")
            used.add(n)
        if not picked:
            return "（未从依据中抽取到相关内容）"
        return "根据知识库：\n" + "\n".join(f"- {p}" for p in picked)


def build_llm(cfg: dict):
    if cfg.get("api_base") and cfg.get("api_key"):
        return OpenAICompatLLM(cfg["api_base"], cfg["api_key"],
                               cfg.get("model", "gpt-4o-mini"),
                               cfg.get("temperature", 0.1))
    return ExtractiveFallback()
