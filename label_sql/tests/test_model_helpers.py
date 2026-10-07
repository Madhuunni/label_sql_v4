import unittest
from label_sql.model import decode_plan
from label_sql.schema import QueryError


class DecoderTests(unittest.TestCase):
    def test_duplicate_key_rejected(self):
        with self.assertRaises(QueryError):
            decode_plan('{"select": [], "select": []}')

    def test_nonfinite_rejected(self):
        with self.assertRaises(QueryError):
            decode_plan('{"limit": NaN}')

    def test_prose_rejected(self):
        with self.assertRaises(QueryError):
            decode_plan('Here is your query: {}')
