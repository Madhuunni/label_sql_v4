"""Exporter contract checks with a fake inspector; no live database required."""
import sys
import types
import unittest
from unittest.mock import patch
from label_sql.export_schema import export_schema


class Inspector:
    default_schema_name = "public"
    def get_table_names(self, schema): return ["staff", "teams"]
    def get_view_names(self, schema): return []
    def get_table_comment(self, name, schema): return {"text": None}
    def get_columns(self, name, schema):
        names = ["id", "team_id", "name"] if name == "staff" else ["id", "name"]
        return [{"name": n, "type": "INTEGER" if n != "name" else "TEXT", "nullable": False,
                 "comment": "label: Staff name" if name == "staff" and n == "name" else None} for n in names]
    def get_pk_constraint(self, name, schema): return {"constrained_columns": ["id"]}
    def get_foreign_keys(self, name, schema, postgresql_ignore_search_path):
        return [{"name": "fk_team", "constrained_columns": ["team_id"], "referred_schema": "public",
                 "referred_table": "teams", "referred_columns": ["id"]}] if name == "staff" else []


class ExportTests(unittest.TestCase):
    def test_labels_and_foreign_keys(self):
        fake = types.ModuleType("sqlalchemy")
        fake.inspect = lambda connection: Inspector()
        conn = types.SimpleNamespace(dialect=types.SimpleNamespace(name="postgresql"))
        with patch.dict(sys.modules, {"sqlalchemy": fake}):
            result = export_schema(conn, ["public"], {"public.staff": {"label": "Employees", "columns": {"id": "Employee ID"}}})
        self.assertEqual(result["tables"][0]["columns"][0]["label"], "Employee ID")
        self.assertEqual(result["tables"][0]["columns"][2]["label"], "Staff name")
        self.assertEqual(result["relations"][0]["from_columns"], ["team_id"])
        self.assertNotIn("password", str(result))


if __name__ == "__main__":
    unittest.main()
