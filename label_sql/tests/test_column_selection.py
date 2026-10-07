import unittest
from pathlib import Path
from label_sql.schema import Schema, QueryError
from label_sql.generate_training import generate, select_columns
from label_sql.compiler import compile_plan

ROOT=Path(__file__).resolve().parents[1]
class ColumnSelectionTests(unittest.TestCase):
    def test_multiple_tables_and_json_columns(self):
        s=Schema.load(ROOT/'data/schema_json_fields.json')
        selectors=['Employees.Employee city','public.employees.salary','departments.department_name']
        rows,report=generate(s,tables=['employees','departments'],columns=selectors,include_joins=True)
        allowed={'Employees.Employee city','Employees.Salary','Departments.Department name'}
        self.assertEqual(len(rows),3600)
        self.assertEqual({r['metadata']['column_label'] for r in rows},allowed)
        self.assertTrue(any(r['metadata']['family']=='join' for r in rows))
        for r in rows:
            compile_plan(s,r['plan'])
            self.assertTrue({c['column'] for c in r['plan']['select']} <= allowed)
            if 'where' in r['plan']: self.assertIn(r['plan']['where']['column'],allowed)
        self.assertEqual([c['count'] for c in report['coverage']],[1200]*3)

    def test_names_duplicates_and_errors(self):
        s=Schema.load(ROOT/'data/schema_json_fields.json')
        picked=select_columns(s,['public.employees'],['Salary','salary','Employees.Salary'])
        self.assertEqual([c['name'] for c in picked['public.employees']],['salary'])
        for selector in ['missing','Departments.Department name']:
            with self.assertRaises(QueryError): select_columns(s,['public.employees'],[selector])
        with self.assertRaises(QueryError): select_columns(s,list(s.tables),['employee_id'])
        picked=select_columns(s,list(s.tables),['Employees.Employee city'])
        self.assertEqual(len(picked['public.departments']),0)
        with self.assertRaises(QueryError): select_columns(s,list(s.tables),[])
