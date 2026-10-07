"""Closed query-plan grammar. No raw SQL expressions are accepted."""
import math
from .schema import QueryError, JSON_CASTS


def quote(name):
    # SQL is a psycopg template, so literal percent signs are doubled.
    return '"' + name.replace('"', '""').replace('%', '%%') + '"'


def keys(obj, allowed, required=()):
    if not isinstance(obj, dict) or set(obj) - set(allowed) or set(required) - set(obj):
        raise QueryError(f"Expected object with keys {sorted(allowed)}; required {sorted(required)}")


def array(value, title, maximum=100):
    if not isinstance(value, list) or len(value) > maximum:
        raise QueryError(f"{title} must be an array of at most {maximum} items")
    return value


def compile_plan(schema, plan, max_limit=1000):
    keys(plan, {"from", "select", "where", "group_by", "having", "order_by", "limit", "distinct", "join_type"}, {"select"})
    columns = []
    expressions = []

    def expr(value, allow_aggregate=True):
        keys(value, {"column", "aggregate"}, {"column"})
        col = value["column"]
        if not isinstance(col, str):
            raise QueryError("column must be a label")
        agg = value.get("aggregate")
        if agg is not None and (not isinstance(agg, str) or agg not in {"count", "sum", "avg", "min", "max"} or not allow_aggregate):
            raise QueryError("Unsupported aggregate in this clause")
        if col == "*":
            if agg != "count":
                raise QueryError("Only count supports '*'")
            ref = None
        else:
            ref = schema.resolve(col)
            if ref not in columns:
                columns.append(ref)
        result = (ref, agg)
        expressions.append(result)
        return result

    selected = [expr(e) for e in array(plan["select"], "select")]
    if not selected:
        raise QueryError("select cannot be empty")
    group = [expr({"column": c}, False) for c in array(plan.get("group_by", []), "group_by")]
    ordering = []
    for item in array(plan.get("order_by", []), "order_by"):
        keys(item, {"column", "aggregate", "direction"}, {"column"})
        direction = item.get("direction", "asc")
        if direction not in ("asc", "desc"):
            raise QueryError("direction must be asc or desc")
        ordering.append((expr({k: v for k, v in item.items() if k != "direction"}), direction))

    def predicate(node, aggregate=False, depth=0):
        if depth > 12:
            raise QueryError("Filter nesting limit exceeded")
        if not isinstance(node, dict):
            raise QueryError("Filter must be an object")
        if set(node) in ({"and"}, {"or"}):
            op = next(iter(node))
            children = array(node[op], op)
            if not children:
                raise QueryError("and/or cannot be empty")
            return (op, [predicate(c, aggregate, depth + 1) for c in children])
        keys(node, {"column", "aggregate", "op", "value"}, {"column", "op"})
        e = expr({k: node[k] for k in ("column", "aggregate") if k in node}, aggregate)
        op = node["op"]
        if not isinstance(op, str) or op not in {"eq", "ne", "gt", "gte", "lt", "lte", "like", "ilike", "in", "is_null", "is_not_null"}:
            raise QueryError("Unsupported filter operator")
        if op in {"is_null", "is_not_null"}:
            if "value" in node:
                raise QueryError("Null checks do not accept value")
            return ("leaf", e, op, None)
        if "value" not in node:
            raise QueryError("Filter requires value")
        value = node["value"]
        values = array(value, "in values", 500) if op == "in" else [value]
        if not values:
            raise QueryError("in values cannot be empty")
        for v in values:
            if v is None or type(v) not in (str, int, float, bool) or (type(v) is float and not math.isfinite(v)):
                raise QueryError("Use scalar finite values; use is_null for NULL")
        if op in {"like", "ilike"} and not isinstance(value, str):
            raise QueryError("LIKE requires a string")
        return ("leaf", e, op, value)

    before_where = len(expressions)
    where = predicate(plan["where"]) if "where" in plan else None
    after_where = len(expressions)
    having = predicate(plan["having"], True) if "having" in plan else None
    # WHERE may reference columns absent from GROUP BY. Other clauses may not.
    group_checked = expressions[:before_where] + expressions[after_where:]
    grouped = bool(group) or any(agg for _, agg in group_checked)
    if having and not grouped:
        raise QueryError("HAVING requires grouping or aggregates")
    if grouped and any(not agg and (ref, agg) not in group for ref, agg in group_checked):
        raise QueryError("Every non-aggregate SELECT/ORDER/HAVING column must be in group_by")
    distinct = plan.get("distinct", False)
    if type(distinct) is not bool:
        raise QueryError("distinct must be boolean")
    if distinct and any(e not in selected for e, _ in ordering):
        raise QueryError("With DISTINCT, ORDER BY expressions must be selected")
    limit = plan.get("limit", 100)
    if type(limit) is not int or not 1 <= limit <= max_limit:
        raise QueryError(f"limit must be an integer from 1 to {max_limit}")
    join_type = plan.get("join_type", "inner")
    if join_type not in ("inner", "left"):
        raise QueryError("join_type must be inner or left")
    root = schema.table(plan["from"]) if "from" in plan else (columns[0][0] if columns else None)
    if root is None:
        raise QueryError("Specify 'from' table label for count(*)")
    if join_type == "left" and "from" not in plan:
        raise QueryError("LEFT JOIN requires an explicit 'from' table label")
    edges = schema.join_tree(root, {t for t, _ in columns})
    aliases = {root: "t0"}
    for _, target, _ in edges:
        aliases[target] = f"t{len(aliases)}"

    def table_sql(tid):
        table = schema.tables[tid]
        return f'{quote(table["schema"])}.{quote(table["name"])} AS {aliases[tid]}'

    json_expressions = {}

    def render_column(ref):
        tid, name = ref
        col = schema.columns[ref]
        if "json" not in col:
            return f"{aliases[tid]}.{quote(name)}"
        if ref not in json_expressions:
            spec = col["json"]
            # Bind path keys as data, including quotes, braces, dots and indexes.
            path = bind([str(p) for p in spec["path"]])
            sql = f'({aliases[tid]}.{quote(spec["column"])} #>> {path}::text[])'
            cast = JSON_CASTS[col["type"].strip().lower()]
            if cast != "text":
                sql = f"({sql}::{cast})"
            # Reuse the same placeholder in GROUP BY/ORDER BY and projection.
            json_expressions[ref] = sql
        return json_expressions[ref]

    def render(e):
        ref, agg = e
        sql = "*" if ref is None else render_column(ref)
        return f"{agg.upper()}({sql})" if agg else sql

    params = {}

    def bind(value):
        key = f"p{len(params)}"
        params[key] = value
        return f"%({key})s"

    def render_filter(node):
        if node[0] in ("and", "or"):
            return "(" + (" " + node[0].upper() + " ").join(render_filter(c) for c in node[1]) + ")"
        _, e, op, value = node
        left = render(e)
        if op in {"is_null", "is_not_null"}:
            return left + (" IS NULL" if op == "is_null" else " IS NOT NULL")
        if op == "in":
            return left + " IN (" + ", ".join(bind(v) for v in value) + ")"
        operation = {"eq": "=", "ne": "<>", "gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "like": "LIKE", "ilike": "ILIKE"}[op]
        return f"{left} {operation} {bind(value)}"

    parts = ["SELECT " + ("DISTINCT " if distinct else "") + ", ".join(map(render, selected)), "FROM " + table_sql(root)]
    for _, target, rid in edges:
        r = schema.relations[rid]
        clauses = [f'{aliases[r["from_table"]]}.{quote(a)} = {aliases[r["to_table"]]}.{quote(b)}'
                   for a, b in zip(r["from_columns"], r["to_columns"])]
        parts.append(join_type.upper() + " JOIN " + table_sql(target) + " ON " + " AND ".join(clauses))
    if where:
        parts.append("WHERE " + render_filter(where))
    if group:
        parts.append("GROUP BY " + ", ".join(map(render, group)))
    if having:
        parts.append("HAVING " + render_filter(having))
    if ordering:
        parts.append("ORDER BY " + ", ".join(render(e) + " " + d.upper() for e, d in ordering))
    parts.append("LIMIT " + bind(limit))
    return {"sql": "\n".join(parts), "params": params, "plan": plan,
            "tables": list(aliases), "relations": [r for _, _, r in edges]}
