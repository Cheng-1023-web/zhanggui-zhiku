"""图存储适配层：NetworkX（默认）与 Neo4j（可选）统一接口。"""
from __future__ import annotations


class BaseGraphStore:
    def add_triple(self, s: str, r: str, o: str): ...
    def match_entities(self, query: str) -> list[str]:
        raise NotImplementedError
    def neighbors(self, entity: str, max_edges: int = 8) -> list[tuple[str, str, str]]:
        raise NotImplementedError
    def stats(self) -> dict:
        raise NotImplementedError


class NetworkXGraphStore(BaseGraphStore):
    """零依赖图存储；实体索引：实体名 → 节点，查询时对 query 分词做实体匹配。"""

    def __init__(self):
        import networkx as nx
        self.g = nx.MultiDiGraph()

    def add_triple(self, s, r, o):
        if self.g.has_edge(s, o, key=r):
            return
        self.g.add_node(s); self.g.add_node(o)
        self.g.add_edge(s, o, key=r, relation=r)

    def _all_entities(self) -> list[str]:
        return list(self.g.nodes)

    def match_entities(self, query: str, top_n: int = 5) -> list[str]:
        q = query.lower()
        hits = [e for e in self._all_entities() if e.lower() in q]
        # 实体名较长时优先
        hits.sort(key=len, reverse=True)
        return hits[:top_n]

    def neighbors(self, entity, max_edges=8):
        out = []
        for _, o, data in list(self.g.out_edges(entity, data=True))[:max_edges]:
            out.append((entity, data.get("relation", "关联"), o))
        for s, _, data in list(self.g.in_edges(entity, data=True))[:max(0, max_edges - len(out))]:
            out.append((s, data.get("relation", "关联"), entity))
        return out

    def stats(self):
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges()}

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.g, f)

    @classmethod
    def load(cls, path):
        import pickle
        obj = cls.__new__(cls)
        import networkx as nx
        with open(path, "rb") as f:
            obj.g = pickle.load(f)
        return obj


class Neo4jGraphStore(BaseGraphStore):
    """Neo4j 后端：docker 起一个 Neo4j 后把 config.graphrag.backend 改为 neo4j 即用。"""

    def __init__(self, uri: str, user: str, password: str):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def add_triple(self, s, r, o):
        r_safe = re.sub(r"\W", "_", r) if (re := __import__("re")) else r
        with self.driver.session() as sess:
            sess.run(
                f"MERGE (a:Entity {{name:$s}}) MERGE (b:Entity {{name:$o}}) "
                f"MERGE (a)-[rel:`{r_safe}`]->(b)",
                s=s, o=o,
            )

    def match_entities(self, query, top_n=5):
        q = query.lower()
        with self.driver.session() as sess:
            rows = sess.run(
                "MATCH (e:Entity) WHERE $q CONTAINS toLower(e.name) "
                "RETURN e.name AS name ORDER BY size(e.name) DESC LIMIT $n",
                q=q, n=top_n,
            )
            return [r["name"] for r in rows]

    def neighbors(self, entity, max_edges=8):
        with self.driver.session() as sess:
            rows = sess.run(
                "MATCH (a:Entity {name:$e})-[r]-(b:Entity) "
                "RETURN a.name AS s, type(r) AS r, b.name AS o LIMIT $n",
                e=entity, n=max_edges,
            )
            return [(r["s"], r["r"], r["o"]) for r in rows]

    def stats(self):
        with self.driver.session() as sess:
            row = sess.run(
                "MATCH (n:Entity) WITH count(n) AS nodes "
                "MATCH ()-[r]->() RETURN nodes, count(r) AS edges").single()
            return {"nodes": row["nodes"], "edges": row["edges"]}


def build_graph_store(cfg: dict):
    if cfg.get("backend") == "neo4j":
        try:
            n = cfg["neo4j"]
            return Neo4jGraphStore(n["uri"], n["user"], n["password"])
        except Exception as e:  # noqa: BLE001
            print(f"[graph] Neo4j 不可用（{e}），回退 NetworkX")
    return NetworkXGraphStore()
