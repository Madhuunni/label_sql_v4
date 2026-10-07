"""Generate modest synthetic starters from label metadata. Review before training."""
import argparse
import itertools
import json
from pathlib import Path
from .schema import Schema, QueryError
from .compiler import compile_plan


def generate(schema):
    rows = []
    def add(question, plan):
        try:
            compile_plan(schema, plan)
        except QueryError:
            return
        rows.append({"question": question, "plan": plan, "source": "synthetic; human review required"})
    refs = []
    for tid, table in schema.tables.items():
        for c in table["columns"]:
            ref = table["label"] + "." + c["label"]
            refs.append((tid, ref))
            select = [{"column": ref}]
            for prefix in ["Show", "List", "Display"]:
                add(f"{prefix} {ref}", {"select": select})
            add(f"Show unique {ref}", {"select": select, "distinct": True})
            add(f"Show {ref} in descending order", {"select": select, "order_by": [{"column": ref, "direction": "desc"}]})
            dtype = c.get("type", "").upper()
            if any(t in dtype for t in ("INT", "NUMERIC", "DECIMAL", "REAL", "DOUBLE", "FLOAT")):
                for value in [10, 100, 500]:
                    add(f"Show {ref} greater than {value}", {"select": select, "where": {"column": ref, "op": "gt", "value": value}})
                add(f"What is the total {ref}?", {"select": [{"column": ref, "aggregate": "sum"}]})
            elif any(t in dtype for t in ("CHAR", "TEXT")):
                for value in ["Active", "Pending", "Chennai"]:
                    add(f"Show {ref} equal to '{value}'", {"select": select, "where": {"column": ref, "op": "eq", "value": value}})
            add(f"Show {ref} that is missing", {"select": select, "where": {"column": ref, "op": "is_null"}})
        add(f"How many {table['label']} are there?", {"from": table["label"], "select": [{"column": "*", "aggregate": "count"}]})
    # Include cross-table projections so the model sees label combinations.
    for (a, left), (b, right) in itertools.islice(itertools.combinations(refs, 2), 5000):
        if a != b:
            add(f"List {left} and {right}", {"select": [{"column": left}, {"column": right}]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows = generate(Schema.load(args.schema))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"Wrote {len(rows)} synthetic starter pairs. Add human-authored multi-clause queries and a separate held-out set.")


if __name__ == "__main__":
    main()
