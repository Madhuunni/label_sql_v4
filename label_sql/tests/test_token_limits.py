import json
import subprocess
import sys
import unittest
from pathlib import Path
from label_sql.tokenizer import LocalTokenizer
from label_sql.schema import QueryError, Schema
from label_sql.train import prepare_tokens, target_text
from label_sql.model import prompt

ROOT = Path(__file__).resolve().parents[1]

class TokenLimitsTests(unittest.TestCase):
    def test_measures_full_source_and_target(self):
        schema = Schema.load(ROOT/'data/schema.json')
        row = {'question':'Show employee name', 'plan':{'select':[{'column':'Employee name'}]}}
        tok = LocalTokenizer.fit([prompt(schema,row['question']),target_text(row)]*2)
        encoded, report = prepare_tokens(schema,[row],[row],tok)
        self.assertEqual(report['required_input_tokens'],len(tok.encode(prompt(schema,row['question']))))
        self.assertEqual(tok.decode(encoded['train'][0][1]),target_text(row))
        self.assertEqual(encoded['train'][0][0][-1],tok.eos_id)
        with self.assertRaisesRegex(QueryError, r'Text contains .* exceeding the limit'):
            tok.encode(prompt(schema,row['question']),1)

    def test_cli_preflight_and_actionable_failure(self):
        args = [sys.executable,'-m','label_sql.train','--schema','data/schema.json',
                '--train','data/train.jsonl','--validation','data/validation.jsonl','--max-input-tokens','1']
        result = subprocess.run(args+['--check-only'],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertGreater(json.loads(result.stdout.splitlines()[0])['required_input_tokens'],1)
        result = subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('--auto-token-limits',result.stderr)
        self.assertNotIn('ModuleNotFoundError',result.stderr)
