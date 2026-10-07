import unittest
from pathlib import Path
from collections import Counter
from label_sql.schema import Schema, QueryError
from label_sql.compiler import compile_plan
from label_sql.generate_training import generate, select_tables

ROOT = Path(__file__).resolve().parents[1]

class GeneratorTests(unittest.TestCase):
    def test_json_columns_counts_uniqueness_and_compilation(self):
        s = Schema.load(ROOT/'data/schema_json_fields.json')
        rows,report = generate(s,['employees'])
        self.assertEqual(len(rows),10800)
        self.assertEqual(len({r['question'].casefold() for r in rows}),len(rows))
        self.assertEqual(set(Counter(r['metadata']['column_label'] for r in rows).values()),{1200})
        self.assertEqual({r['metadata']['table_id'] for r in rows},{'public.employees'})
        self.assertEqual(report['total'],len(rows))
        for r in rows: compile_plan(s,r['plan'])
        city = next(r for r in rows if r['metadata']['column_label']=='Employees.Employee city')
        self.assertIn('#>>',compile_plan(s,city['plan'])['sql'])
        self.assertEqual(rows,generate(s,['employees'])[0])

    def test_join_and_samples(self):
        s = Schema.load(ROOT/'data/schema_json_fields.json')
        rows,_ = generate(s,['employees','departments'],samples={'Employees.Employee city':['Chennai']},include_joins=True)
        joins = [r for r in rows if r['metadata']['family']=='join']
        self.assertTrue(joins)
        self.assertIn('JOIN',compile_plan(s,joins[0]['plan'])['sql'])
        values = [r['plan']['where']['value'] for r in rows if r['metadata']['column_label']=='Employees.Employee city' and r['metadata']['family']=='filter']
        self.assertEqual(set(values),{'Chennai'})

    def test_errors_and_table_selection(self):
        s = Schema.load(ROOT/'data/schema.json')
        self.assertEqual(select_tables(s,['Employees','public.employees']),['public.employees'])
        with self.assertRaises(QueryError): generate(s,['missing'])
        with self.assertRaises(QueryError): generate(s,per_column=1199)
        with self.assertRaises(QueryError): generate(s,samples={'bad label':[1]})
        with self.assertRaises(QueryError): generate(s,['employees'],samples={'Employees.Salary':['bad']})

if __name__ == '__main__': unittest.main()
