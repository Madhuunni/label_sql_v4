import tempfile
import unittest
from pathlib import Path
from label_sql.tokenizer import LocalTokenizer
from label_sql.schema import QueryError, Schema
from label_sql.model import prompt
from label_sql.train import read_examples, target_text

class TokenizerTests(unittest.TestCase):
    def test_roundtrip_json_and_unseen_unicode(self):
        tokenizer = LocalTokenizer.fit(['Employee name', 'Employee name'])
        for text in ['{"id":"0012345","name":"தமிழ் / 日本語 😀"}', ' a\n\t b\r\n', 'employee_id ٩ 123']:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_unseen_digits(self):
        tokenizer = LocalTokenizer.fit(['123', '123'])
        self.assertEqual(tokenizer.decode(tokenizer.encode('000998877')), '000998877')

    def test_save_load(self):
        tokenizer = LocalTokenizer.fit(['salary salary Employee Employee'])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'tokenizer.json'
            tokenizer.save(path)
            self.assertEqual(LocalTokenizer.load(path).encode('Employee 100'), tokenizer.encode('Employee 100'))

    def test_limits(self):
        with self.assertRaises(QueryError):
            LocalTokenizer.fit(['x'], max_vocab=100)
        with self.assertRaises(QueryError):
            LocalTokenizer().encode('abcdef', max_tokens=2)

    def test_all_examples_fit_default_limits(self):
        root = Path(__file__).resolve().parents[1]
        schema = Schema.load(root / 'data/schema.json')
        training = read_examples(root / 'data/train.jsonl', schema)
        tokenizer = LocalTokenizer.fit([text for row in training for text in (prompt(schema, row['question']), target_text(row))])
        for name in ['train', 'validation', 'test']:
            for row in read_examples(root / f'data/{name}.jsonl', schema):
                tokenizer.encode(prompt(schema, row['question']), 2048)
                tokenizer.encode(target_text(row), 1024)
