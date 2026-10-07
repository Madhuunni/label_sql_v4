"""Read PostgreSQL catalog metadata; never export row data or credentials."""
import argparse
import copy
import json
import os
from pathlib import Path
from .schema import Schema, QueryError


def human(name):
    return name.replace("_", " ").strip().title()


def comment_label(comment, fallback):
    # Only explicit label: comments are treated as labels, not arbitrary prose.
    if isinstance(comment, str) and comment.lower().startswith("label:"):
        return comment[6:].strip()
    return fallback


def export_schema(connection, schemas, overrides=None, include_views=False):
    """Accept an existing SQLAlchemy Connection; return the portable JSON dict."""
    from sqlalchemy import inspect
    inspector = inspect(connection)
    if connection.dialect.name != "postgresql":
        raise QueryError("This exporter requires PostgreSQL")
    overrides = overrides or {}
    tables, relations, omitted = [], [], []
    fks = {}
    for namespace in schemas:
        names = inspector.get_table_names(schema=namespace)
        views = inspector.get_view_names(schema=namespace) if include_views else []
        for name in sorted(set(names + views)):
            tid = f"{namespace}.{name}"
            custom = overrides.get(tid, {})
            comment = inspector.get_table_comment(name, schema=namespace).get("text")
            label = custom.get("label", comment_label(comment, human(namespace) + " " + human(name)))
            cols = inspector.get_columns(name, schema=namespace)
            known = {c["name"] for c in cols}
            if set(custom.get("columns", {})) - known:
                raise QueryError(f"Unknown column in label overrides for {tid}")
            columns = []
            for col in cols:
                cfg = custom.get("columns", {}).get(col["name"], {})
                if isinstance(cfg, str):
                    cfg = {"label": cfg}
                columns.append({"name": col["name"], "label": cfg.get("label", comment_label(col.get("comment"), human(col["name"]))),
                                "aliases": cfg.get("aliases", []), "type": str(col["type"]),
                                "nullable": bool(col["nullable"]), "comment": col.get("comment")})
            virtuals = custom.get("json_columns", [])
            if not isinstance(virtuals, list) or any(not isinstance(c, dict) or "json" not in c for c in virtuals):
                raise QueryError("json_columns must be an array of virtual column definitions containing json")
            columns.extend(copy.deepcopy(virtuals))
            primary = inspector.get_pk_constraint(name, schema=namespace).get("constrained_columns", []) if name not in views else []
            tables.append({"id": tid, "schema": namespace, "name": name, "label": label,
                           "kind": "view" if name in views else "table", "columns": columns, "primary_key": primary,
                           "comment": comment})
            fks[tid] = inspector.get_foreign_keys(name, schema=namespace, postgresql_ignore_search_path=True) if name not in views else []
    ids = {t["id"] for t in tables}
    if set(overrides) - ids:
        raise QueryError("Label overrides refer to tables outside the exported schema selection")
    for tid, foreign_keys in fks.items():
        for index, fk in enumerate(foreign_keys):
            namespace = fk.get("referred_schema") or inspector.default_schema_name
            target = f'{namespace}.{fk["referred_table"]}'
            rid = f'{tid}:{fk.get("name") or index}'
            if target not in ids:
                omitted.append({"relation": rid, "reason": "Referenced table not in exported selection", "target": target})
                continue
            relations.append({"id": rid, "from_table": tid, "from_columns": fk["constrained_columns"],
                              "to_table": target, "to_columns": fk["referred_columns"]})
    data = {"version": 1, "dialect": "postgresql", "tables": tables, "relations": relations, "omitted_relations": omitted}
    Schema(data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url-env", default="DATABASE_URL", help="Environment variable containing postgresql+psycopg:// URL")
    parser.add_argument("--schemas", nargs="+", default=["public"])
    parser.add_argument("--labels", help="Optional JSON label overrides")
    parser.add_argument("--include-views", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from sqlalchemy import create_engine
    url = os.environ.get(args.url_env)
    if not url:
        parser.error(f"Set {args.url_env} to the database URL")
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    overrides = json.loads(Path(args.labels).read_text()) if args.labels else None
    engine = create_engine(url)
    try:
        with engine.connect() as conn:
            # Catalog reflection only; reinforce the read-only intent.
            conn.exec_driver_sql("SET TRANSACTION READ ONLY")
            data = export_schema(conn, args.schemas, overrides, args.include_views)
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Exported {len(data['tables'])} tables/views and {len(data['relations'])} relations to {output}")
        if data["omitted_relations"]:
            print(f"Warning: {len(data['omitted_relations'])} cross-selection relations omitted; see omitted_relations")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
