"""轻量 LangGraph 风格状态图引擎。

API 对齐 langgraph：StateGraph / add_node / add_edge / add_conditional_edges /
compile().invoke(state)。未引入 langgraph 依赖以保证零网络环境可跑通；
如需切换官方实现，仅需替换本文件（节点函数签名一致）。
"""
from __future__ import annotations

from typing import Callable


class StateGraph:
    def __init__(self, state_schema: type | None = None):
        self._nodes: dict[str, Callable] = {}
        self._edges: dict[str, str] = {}
        self._cond: dict[str, Callable] = {}
        self._entry: str | None = None

    def add_node(self, name: str, fn: Callable) -> "StateGraph":
        self._nodes[name] = fn
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        self._entry = name
        return self

    def add_edge(self, a: str, b: str) -> "StateGraph":
        self._edges[a] = b
        return self

    def add_conditional_edges(self, a: str, router: Callable) -> "StateGraph":
        self._cond[a] = router
        return self

    def compile(self):
        return CompiledGraph(self._nodes, self._edges, self._cond, self._entry)


class CompiledGraph:
    def __init__(self, nodes, edges, cond, entry):
        self.nodes, self.edges, self.cond, self.entry = nodes, edges, cond, entry

    def invoke(self, state: dict) -> dict:
        cur = self.entry
        steps = 0
        while cur and cur != "END" and steps < 50:
            fn = self.nodes.get(cur)
            if fn:
                fn(state)
            cur = self.cond[cur](state) if cur in self.cond else self.edges.get(cur)
            steps += 1
        return state


START = "__start__"
END = "END"
