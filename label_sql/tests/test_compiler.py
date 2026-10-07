import copy
import json
import unittest
from pathlib import Path
from label_sql import Schema, QueryError, compile_plan

ROOT = Path(__file__).resolve().parents[1]


class CompilerTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads((ROOT / "data/schema.json").read_text())
        self.schema = Schema(self.raw)

    def test_three_table_join(self):
        plan = json.loads((ROOT / "data/example_plan.json").read_text())
        result = compile_plan(self.schema, plan)
        self.assertEqual(result["sql"].count("INNER JOIN"), 2)
        self.assertEqual(result["params"], {"p0": "Open", "p1": 5000, "p2": 100})
        self.assertIn('t0."full_name"', result["sql"])

    def test_bridge_table_not_selected(self):
        result = compile_plan(self.schema, {"select": [{"column": "Department name"}, {"column": "Case ID"}]})
        self.assertEqual(len(result["tables"]), 3)
        self.assertEqual(len(result["relations"]), 2)

    def test_values_cannot_become_sql(self):
        attack = "x'; DROP TABLE employees; --"
        r = compile_plan(self.schema, {"select": [{"column": "Employee name"}], "where": {"column": "Employee name", "op": "eq", "value": attack}})
        self.assertNotIn(attack, r["sql"])
        self.assertEqual(r["params"]["p0"], attack)

    def test_composite_fk(self):
        self.raw["tables"][0]["columns"].append({"name": "tenant", "label": "Employee tenant"})
        self.raw["tables"][1]["columns"].append({"name": "tenant", "label": "Department tenant"})
        r = self.raw["relations"][0]
        r["from_columns"].append("tenant"); r["to_columns"].append("tenant")
        result = compile_plan(Schema(self.raw), {"select": [{"column": "Employee name"}, {"column": "Department name"}]})
        self.assertIn('t0."tenant" = t1."tenant"', result["sql"])

    def test_duplicate_label_requires_qualification(self):
        self.raw["tables"][0]["columns"][0]["label"] = "ID"
        self.raw["tables"][1]["columns"][0]["label"] = "ID"
        schema = Schema(self.raw)
        with self.assertRaisesRegex(QueryError, "Ambiguous"):
            schema.resolve("ID")
        self.assertEqual(schema.resolve("Employees.ID"), ("public.employees", "employee_id"))

    def test_unknown_label(self):
        with self.assertRaisesRegex(QueryError, "Unknown"):
            compile_plan(self.schema, {"select": [{"column": "invented"}]})

    def test_parallel_foreign_keys_are_ambiguous(self):
        other = copy.deepcopy(self.raw["relations"][0]); other["id"] = "another"
        self.raw["relations"].append(other)
        with self.assertRaisesRegex(QueryError, "Ambiguous join"):
            compile_plan(Schema(self.raw), {"select": [{"column": "Employee name"}, {"column": "Department name"}]})

    def test_disconnected_tables(self):
        self.raw["relations"] = []
        with self.assertRaisesRegex(QueryError, "No declared"):
            compile_plan(Schema(self.raw), {"select": [{"column": "Employee name"}, {"column": "Department name"}]})

    def test_aggregate_group_having(self):
        plan = {"select": [{"column": "Department name"}, {"column": "Salary", "aggregate": "sum"}],
                "group_by": ["Department name"], "having": {"column": "Salary", "aggregate": "sum", "op": "gt", "value": 100},
                "order_by": [{"column": "Salary", "aggregate": "sum", "direction": "desc"}]}
        result = compile_plan(self.schema, plan)
        self.assertIn("GROUP BY", result["sql"])
        self.assertIn("HAVING SUM", result["sql"])
        self.assertIn("ORDER BY SUM", result["sql"])

    def test_where_not_required_in_group(self):
        compile_plan(self.schema, {"select": [{"column": "Salary", "aggregate": "sum"}],
                                  "where": {"column": "Employee name", "op": "eq", "value": "Anita"}})

    def test_invalid_group_rejected(self):
        with self.assertRaisesRegex(QueryError, "group_by"):
            compile_plan(self.schema, {"select": [{"column": "Employee name"}, {"column": "Salary", "aggregate": "sum"}]})

    def test_raw_sql_and_bad_limit_rejected(self):
        for plan in [{"select": [{"column": "Employee name"}], "raw": "DROP"},
                     {"select": [{"column": "Employee name"}], "limit": True},
                     {"select": [{"column": "Employee name"}], "limit": 0},
                     {"select": [{"column": "Employee name"}], "limit": 1001}]:
            with self.assertRaises(QueryError):
                compile_plan(self.schema, plan)

    def test_count_requires_from(self):
        with self.assertRaises(QueryError):
            compile_plan(self.schema, {"select": [{"column": "*", "aggregate": "count"}]})
        result = compile_plan(self.schema, {"from": "Employees", "select": [{"column": "*", "aggregate": "count"}]})
        self.assertIn("COUNT(*)", result["sql"])

    def test_left_join_requires_root(self):
        p = {"select": [{"column": "Employee name"}, {"column": "Department name"}], "join_type": "left"}
        with self.assertRaises(QueryError):
            compile_plan(self.schema, p)
        p["from"] = "Employees"
        self.assertIn("LEFT JOIN", compile_plan(self.schema, p)["sql"])

    def test_nested_filter_and_in(self):
        p = {"select": [{"column": "Employee name"}], "where": {"or": [
            {"column": "Employee ID", "op": "in", "value": [1, 2]},
            {"column": "Employee department ID", "op": "is_null"}]}}
        r = compile_plan(self.schema, p)
        self.assertEqual(r["params"], {"p0": 1, "p1": 2, "p2": 100})
        self.assertIn("IS NULL", r["sql"])

    def test_quoted_identifiers(self):
        self.raw["tables"][0]["columns"][1]["name"] = 'odd"%name'
        r = compile_plan(Schema(self.raw), {"select": [{"column": "Employee name"}]})
        self.assertIn('"odd""%%name"', r["sql"])

    def test_distinct_order_must_be_selected(self):
        with self.assertRaises(QueryError):
            compile_plan(self.schema, {"select": [{"column": "Employee name"}], "distinct": True,
                                      "order_by": [{"column": "Salary"}]})

    def test_all_bundled_examples_compile(self):
        for filename in ("train.jsonl", "validation.jsonl", "test.jsonl"):
            for line in (ROOT / "data" / filename).read_text().splitlines():
                compile_plan(self.schema, json.loads(line)["plan"])


if __name__ == "__main__":
    unittest.main()
