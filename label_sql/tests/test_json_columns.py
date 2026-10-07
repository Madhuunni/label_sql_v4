import copy
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from label_sql import Schema, QueryError, compile_plan
from label_sql.tokenizer import LocalTokenizer
from label_sql.train import read_examples, target_text
from label_sql.model import prompt
from label_sql.export_schema import export_schema

ROOT = Path(__file__).resolve().parents[1]

class JsonColumnTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads((ROOT / 'data/schema_json_fields.json').read_text())
        self.schema = Schema(self.raw)

    def virtual(self, name):
        return next(c for c in self.raw['tables'][0]['columns'] if c['name'] == name)

    def test_projection_filters_numeric_order(self):
        plan = json.loads((ROOT / 'data/example_json_plan.json').read_text())
        result = compile_plan(self.schema, plan)
        self.assertIn('t0."profile" #>> %(p0)s::text[]', result['sql'])
        self.assertIn('::numeric', result['sql'])
        self.assertNotIn('"profile_city"', result['sql'])
        self.assertEqual(result['params'], {'p0':['address','city'], 'p1':['bonus'], 'p2':'Chennai', 'p3':10000, 'p4':100})

    def test_grouping_reuses_path_placeholder(self):
        result = compile_plan(self.schema, {'select':[{'column':'Employee city'},{'column':'Annual bonus','aggregate':'sum'}], 'group_by':['Employee city'], 'order_by':[{'column':'Employee city'}]})
        self.assertEqual(result['sql'].count('#>> %(p0)s::text[]'),3)
        self.assertIn('SUM(', result['sql'])
        self.assertEqual(result['params']['p1'],['bonus'])

    def test_json_filter_across_join(self):
        result = compile_plan(self.schema, {'select':[{'column':'Department name'}], 'where':{'column':'Employee city','op':'eq','value':'Salem'}})
        self.assertIn('INNER JOIN "public"."employees"',result['sql'])
        self.assertIn('t1."profile" #>>',result['sql'])

    def test_date_boolean_and_null(self):
        result=compile_plan(self.schema, {'select':[{'column':'Joining date'},{'column':'Active employee'}], 'where':{'column':'Employee city','op':'is_null'}})
        self.assertIn('::date',result['sql']); self.assertIn('::boolean',result['sql'])
        self.assertIn('IS NULL',result['sql'])

    def test_special_keys_and_array_index_bound_as_data(self):
        path=['a.b', 'x\'}; DROP TABLE employees; --', 0, '']
        self.virtual('profile_city')['json']['path']=path
        r=compile_plan(Schema(self.raw),{'select':[{'column':'Employee city'}]})
        self.assertEqual(r['params']['p0'],[str(x) for x in path])
        self.assertNotIn('DROP',r['sql'])

    def test_invalid_json_definitions(self):
        variants=[('type','text); DROP TABLE x; --'),('json',{'column':'missing','path':['x']}),
                  ('json',{'column':'salary','path':['x']}),('json',{'column':'profile_bonus','path':['x']}),
                  ('json',{'column':'profile','path':[]}),('json',{'column':'profile','path':[True]}),
                  ('json',{'column':'profile','path':['x'],'sql':'evil'})]
        for field,value in variants:
            raw=copy.deepcopy(self.raw)
            col=next(c for c in raw['tables'][0]['columns'] if c['name']=='profile_city')
            col[field]=value
            with self.subTest(value=value), self.assertRaises(QueryError):Schema(raw)

    def test_virtual_fk_rejected(self):
        self.raw['relations'][0]['from_columns']=['profile_city']
        with self.assertRaisesRegex(QueryError,'nonphysical'):Schema(self.raw)

    def test_training_examples_and_token_lengths(self):
        rows=read_examples(ROOT / 'data/json_fields_train.jsonl',self.schema)
        tokenizer=LocalTokenizer.fit([t for row in rows for t in (prompt(self.schema,row['question']),target_text(row))])
        for name in ['train','validation','test']:
            for row in read_examples(ROOT / f'data/json_fields_{name}.jsonl',self.schema):
                tokenizer.encode(prompt(self.schema,row['question']),2048)
                tokenizer.encode(target_text(row),1024)
        self.assertIn('Employee city',self.schema.prompt_schema())

    def test_exporter_keeps_json_mapping(self):
        class Inspector:
            def get_table_names(self,schema):return ['employees']
            def get_table_comment(self,name,schema):return {'text':None}
            def get_columns(self,name,schema):return [{'name':'profile','type':'JSONB','nullable':True}]
            def get_pk_constraint(self,name,schema):return {'constrained_columns':[]}
            def get_foreign_keys(self,*args,**kwargs):return []
        module=types.ModuleType('sqlalchemy'); module.inspect=lambda conn:Inspector()
        conn=types.SimpleNamespace(dialect=types.SimpleNamespace(name='postgresql'))
        override={'public.employees':{'label':'Employees','json_columns':[self.virtual('profile_city')]}}
        with patch.dict(sys.modules,{'sqlalchemy':module}):data=export_schema(conn,['public'],override)
        schema=Schema(data)
        self.assertEqual(schema.resolve('Employee city'),('public.employees','profile_city'))
        self.assertEqual(data['tables'][0]['columns'][1]['json']['path'],['address','city'])
