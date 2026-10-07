"""Optional checks of our actual PyTorch network and checkpoint code."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

@unittest.skipUnless(importlib.util.find_spec('torch'), 'Install torch to run network tests')
class ModelSmoke(unittest.TestCase):
    def test_forward_backward_generation(self):
        import torch
        from label_sql.network import ScratchTransformer
        model = ScratchTransformer(300, d_model=16, nhead=2, layers=1, d_ff=32, max_positions=20, dropout=0)
        src = torch.tensor([[5, 6, 2, 0], [7, 8, 9, 2]])
        logits = model(src, torch.tensor([[1, 10, 11], [1, 12, 13]]))
        self.assertEqual(tuple(logits.shape), (2, 3, 300))
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 300), torch.tensor([10, 11, 2, 12, 13, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(model.embedding.weight.grad)
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.zero_()
            model.output.bias[2] = 10
        self.assertEqual(model.generate(src[:1], 5), [1, 2])

    def test_decoder_cannot_see_future(self):
        import torch
        from label_sql.network import ScratchTransformer
        torch.manual_seed(1)
        model = ScratchTransformer(300, d_model=16, nhead=2, layers=1, d_ff=32, max_positions=20, dropout=0).eval()
        src = torch.tensor([[5, 6, 2]])
        with torch.no_grad():
            a = model(src, torch.tensor([[1, 10, 11]]))
            b = model(src, torch.tensor([[1, 90, 91]]))
        torch.testing.assert_close(a[:, 0], b[:, 0])

    def test_one_epoch_checkpoint_reload(self):
        from label_sql.train import main
        from label_sql.model import Predictor
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for filename, question in [('train', 'Show salary'), ('valid', 'List salary')]:
                (folder / (filename + '.jsonl')).write_text(json.dumps({'question': question, 'plan': {'select': [{'column': 'Salary'}]}}) + '\n')
            args = ['train', '--schema', str(root / 'data/schema.json'), '--train', str(folder / 'train.jsonl'),
                    '--validation', str(folder / 'valid.jsonl'), '--output', str(folder / 'model'), '--epochs', '1',
                    '--d-model', '16', '--heads', '2', '--layers', '1', '--ff-size', '32', '--device', 'cpu']
            with patch('sys.argv', args):
                main()
            loaded = Predictor(folder / 'model', device='cpu')
            self.assertEqual(loaded.config['architecture'], 'scratch-transformer-v1')
