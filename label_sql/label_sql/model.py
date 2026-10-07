"""Local inference. No pretrained models, model hubs, or network calls."""
import json
from pathlib import Path
from .schema import QueryError
from .tokenizer import LocalTokenizer

PREFIX = "Convert the question to a label-based JSON query plan. Return JSON only. Schema: "


def prompt(schema, question):
    if not isinstance(question, str) or not question.strip():
        raise QueryError("Question cannot be empty")
    return PREFIX + schema.prompt_schema() + "\nQuestion: " + question.strip() + "\nPlan:"


def decode_plan(text):
    def unique_pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise QueryError(f"Duplicate JSON key: {key}")
            obj[key] = value
        return obj
    try:
        return json.loads(text, object_pairs_hook=unique_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(QueryError("Nonfinite JSON number")))
    except json.JSONDecodeError as exc:
        raise QueryError("Model returned invalid JSON; improve training examples") from exc


class Predictor:
    def __init__(self, checkpoint, device=None):
        import torch
        from .network import ScratchTransformer
        path = Path(checkpoint)
        cfg_path = path / "label_sql_config.json"
        if not cfg_path.exists():
            raise QueryError("Run train first to create a local checkpoint")
        self.config = json.loads(cfg_path.read_text())
        if self.config.get("architecture") != "scratch-transformer-v1":
            raise QueryError("Checkpoint is not a scratch Transformer; retrain with this version")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = LocalTokenizer.load(path / "tokenizer.json")
        self.model = ScratchTransformer(**self.config["network"])
        if self.config["network"]["vocab_size"] != len(self.tokenizer):
            raise QueryError("Tokenizer/checkpoint vocabulary mismatch")
        weights = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(weights)
        self.model.to(self.device).eval()

    def predict(self, schema, question):
        import torch
        from .compiler import compile_plan
        ids = self.tokenizer.encode(prompt(schema, question), self.config["max_input_tokens"])
        source = torch.tensor([ids], dtype=torch.long, device=self.device)
        output = self.model.generate(source, self.config["max_output_tokens"])
        return compile_plan(schema, decode_plan(self.tokenizer.decode(output)))
