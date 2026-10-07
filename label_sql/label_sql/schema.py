import json
from collections import defaultdict, deque
from pathlib import Path

JSON_CASTS = {
    "text": "text", "integer": "integer", "bigint": "bigint",
    "numeric": "numeric", "decimal": "numeric", "real": "real",
    "double precision": "double precision", "boolean": "boolean",
    "date": "date", "timestamp": "timestamp", "timestamptz": "timestamptz",
    "uuid": "uuid",
}


class QueryError(ValueError):
    """A schema, plan, or generated prediction cannot be used safely."""


def norm(value):
    return " ".join(value.casefold().split())


def named(value):
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise QueryError("Names and labels must be nonempty strings without NUL")
    return value


class Schema:
    def __init__(self, data):
        if not isinstance(data, dict) or data.get("version") != 1:
            raise QueryError("Expected schema JSON version 1")
        self.data = data
        self.tables = {}
        self.columns = {}
        self.labels = defaultdict(list)
        self.table_labels = defaultdict(list)
        for table in data.get("tables", []):
            tid = named(table["id"])
            if tid in self.tables:
                raise QueryError(f"Duplicate table id: {tid}")
            named(table["schema"]); named(table["name"]); named(table["label"])
            if "." in table["label"]:
                raise QueryError("Table labels cannot contain '.' (qualification separator)")
            self.tables[tid] = table
            self.table_labels[norm(table["label"])].append(tid)
            seen = set()
            for col in table["columns"]:
                name, label = named(col["name"]), named(col["label"])
                if name in seen or "." in label:
                    raise QueryError("Duplicate column name or '.' in column label")
                seen.add(name)
                self.columns[(tid, name)] = col
                aliases = col.get("aliases", [])
                if not isinstance(aliases, list):
                    raise QueryError("Column aliases must be an array")
                for text in set([label] + [named(a) for a in aliases]):
                    ref = (tid, name)
                    for key in [norm(text), norm(table["label"] + "." + text)]:
                        if ref not in self.labels[key]:
                            self.labels[key].append(ref)
            # Check after indexing all columns, so declaration order is immaterial.
            for col in table["columns"]:
                if "json" not in col:
                    continue
                spec = col["json"]
                if not isinstance(spec, dict) or set(spec) != {"column", "path"}:
                    raise QueryError("JSON column requires json: {column: physical_name, path: [...]} only")
                source = self.columns.get((tid, named(spec["column"])))
                if source is None or "json" in source:
                    raise QueryError("JSON source must be a physical column in the same table")
                if str(source.get("type", "")).strip().lower() not in {"json", "jsonb"}:
                    raise QueryError("JSON source column type must be JSON or JSONB")
                path = spec["path"]
                if not isinstance(path, list) or not 1 <= len(path) <= 32:
                    raise QueryError("JSON path must contain 1 to 32 keys/indexes")
                if any(type(p) not in (str, int) or (isinstance(p, str) and "\x00" in p) for p in path):
                    raise QueryError("JSON path elements must be string keys or integer indexes without NUL")
                if str(col.get("type", "")).strip().lower() not in JSON_CASTS:
                    raise QueryError("JSON virtual column requires a supported scalar type: " + ", ".join(JSON_CASTS))
        if not self.tables:
            raise QueryError("Schema contains no tables")
        if any(len(v) > 1 for v in self.table_labels.values()):
            raise QueryError("Table labels must be unique; edit exported labels")
        self.relations = {}
        self.graph = defaultdict(list)
        for rel in data.get("relations", []):
            rid = named(rel["id"])
            a, b = rel["from_table"], rel["to_table"]
            if rid in self.relations or a not in self.tables or b not in self.tables:
                raise QueryError(f"Invalid relation: {rid}")
            left, right = rel["from_columns"], rel["to_columns"]
            if not left or len(left) != len(right):
                raise QueryError(f"Invalid composite key: {rid}")
            for tid, cols in [(a, left), (b, right)]:
                existing = {c["name"] for c in self.tables[tid]["columns"] if "json" not in c}
                if any(c not in existing for c in cols):
                    raise QueryError(f"Unknown or nonphysical relation column: {rid}")
            self.relations[rid] = rel
            if a != b:  # A self-join requires role aliases, outside this plan grammar.
                self.graph[a].append((b, rid))
                self.graph[b].append((a, rid))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def resolve(self, label):
        if not isinstance(label, str):
            raise QueryError("Column reference must be a label string")
        hits = self.labels.get(norm(label), [])
        if not hits:
            raise QueryError(f"Unknown column label: {label}")
        if len(hits) != 1:
            raise QueryError(f"Ambiguous label '{label}'; use Table label.Column label")
        return hits[0]

    def table(self, label):
        hits = self.table_labels.get(norm(named(label)), [])
        if len(hits) != 1:
            raise QueryError(f"Unknown or ambiguous table label: {label}")
        return hits[0]

    def join_tree(self, root, required):
        """Unique shortest FK path per target. Reject cycles and ambiguous paths."""
        distance, ways, parent = {root: 0}, {root: 1}, {}
        queue = deque([root])
        while queue:
            here = queue.popleft()
            for nxt, rid in sorted(self.graph[here]):
                d = distance[here] + 1
                if nxt not in distance:
                    distance[nxt], ways[nxt], parent[nxt] = d, ways[here], (here, rid)
                    queue.append(nxt)
                elif distance[nxt] == d:
                    ways[nxt] = min(2, ways[nxt] + ways[here])
        edges = set()
        for target in required:
            if target not in distance:
                raise QueryError("No declared relationship connects the requested tables")
            if ways[target] != 1:
                raise QueryError("Ambiguous join path; supply a curated schema with one intended relationship path")
            current = target
            while current != root:
                before, rid = parent[current]
                edges.add((before, current, rid))
                current = before
        return sorted(edges, key=lambda e: (distance[e[1]], e[1], e[2]))

    def prompt_schema(self):
        """Compact label-only context; physical identifiers stay in the compiler."""
        tables = [{"label": t["label"], "columns": [
            {"label": c["label"], "type": c.get("type", "unknown"), "aliases": c.get("aliases", [])}
            for c in t["columns"]]} for t in self.tables.values()]
        relations = []
        for r in self.relations.values():
            endpoints = []
            for side in ["from", "to"]:
                table = self.tables[r[side + "_table"]]
                lookup = {c["name"]: c["label"] for c in table["columns"]}
                endpoints.append([table["label"] + "." + lookup[c] for c in r[side + "_columns"]])
            relations.append(endpoints)
        return json.dumps({"tables": tables, "relations": relations}, ensure_ascii=False, separators=(",", ":"))
