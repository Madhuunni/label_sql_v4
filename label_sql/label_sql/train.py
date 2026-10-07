"""Train a randomly initialized Transformer using only local schema/example files."""
import argparse
import json
import random
from pathlib import Path
from .compiler import compile_plan
from .model import prompt
from .schema import Schema, QueryError
from .tokenizer import LocalTokenizer


def read_examples(path, schema):
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise QueryError(f"Missing question at line {number}")
        compile_plan(schema, item["plan"])
        records.append(item)
    if not records:
        raise QueryError("Training/evaluation file contains no examples")
    return records


def target_text(row):
    return json.dumps(row["plan"], ensure_ascii=False, separators=(",", ":"))


def prepare_tokens(schema, training, validation, tokenizer):
    """Encode full sequences once and report lengths including EOS, excluding BOS."""
    encoded, report = {}, {}
    for split, rows in (("train", training), ("validation", validation)):
        pairs = [(tokenizer.encode(prompt(schema, r["question"])), tokenizer.encode(target_text(r))) for r in rows]
        encoded[split] = pairs
        source_lengths = [len(s) for s, t in pairs]
        target_lengths = [len(t) for s, t in pairs]
        report[split] = {"examples": len(rows), "max_input_tokens": max(source_lengths),
                         "max_output_tokens": max(target_lengths),
                         "longest_input_example": source_lengths.index(max(source_lengths)) + 1,
                         "longest_output_example": target_lengths.index(max(target_lengths)) + 1}
    report["required_input_tokens"] = max(report[s]["max_input_tokens"] for s in encoded)
    report["required_output_tokens"] = max(report[s]["max_output_tokens"] for s in encoded)
    report["schema_tokens"] = len(tokenizer.encode(schema.prompt_schema()))
    return encoded, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", default="artifacts/label-sql-scratch")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--check-only", action="store_true", help="Report token lengths without PyTorch or training")
    parser.add_argument("--auto-token-limits", action="store_true", help="Increase limits to fit complete examples")
    parser.add_argument("--max-vocab", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ff-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.max_input_tokens, args.max_output_tokens) < 1 or args.lr <= 0:
        parser.error("Epochs, batch size, patience, token limits and learning rate must be positive")
    schema = Schema.load(args.schema)
    training = read_examples(args.train, schema)
    validation = read_examples(args.validation, schema)
    if {r["question"].strip().casefold() for r in training} & {r["question"].strip().casefold() for r in validation}:
        raise QueryError("Training and validation questions overlap")
    # Vocabulary is learned only from training inputs/targets; never from held-out questions.
    texts = (text for row in training for text in (prompt(schema, row["question"]), target_text(row)))
    tokenizer = LocalTokenizer.fit(texts, args.max_vocab)

    encoded, report = prepare_tokens(schema, training, validation, tokenizer)
    print(json.dumps(report), flush=True)
    if args.check_only:
        print("Check complete; no training. Set limits from these lengths or use --auto-token-limits.")
        return
    if args.auto_token_limits:
        args.max_input_tokens = max(args.max_input_tokens, report["required_input_tokens"])
        args.max_output_tokens = max(args.max_output_tokens, report["required_output_tokens"])
    if report["required_input_tokens"] > args.max_input_tokens or report["required_output_tokens"] > args.max_output_tokens:
        raise QueryError(
            f"Input needs {report['required_input_tokens']} tokens (limit {args.max_input_tokens}); "
            f"output needs {report['required_output_tokens']} (limit {args.max_output_tokens}). "
            "Set --max-input-tokens / --max-output-tokens or --auto-token-limits. "
            "The full schema is included in every input; generator --tables does not reduce it. No text was truncated.")
    print(f"Token limits: input={args.max_input_tokens}, output={args.max_output_tokens}. Longer inputs increase attention memory use.", flush=True)
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence
    from .network import ScratchTransformer
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    def tensors(split):
        return [(torch.tensor(source), torch.tensor([tokenizer.bos_id] + target))
                for source, target in encoded[split]]

    def collate(rows):
        source = pad_sequence([r[0] for r in rows], batch_first=True, padding_value=0)
        target = pad_sequence([r[1] for r in rows], batch_first=True, padding_value=0)
        return source, target[:, :-1], target[:, 1:]

    train_loader = DataLoader(tensors("train"), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(tensors("validation"), batch_size=args.batch_size, collate_fn=collate)
    network = {"vocab_size": len(tokenizer), "d_model": args.d_model, "nhead": args.heads,
               "layers": args.layers, "d_ff": args.ff_size, "dropout": args.dropout,
               "max_positions": max(args.max_input_tokens, args.max_output_tokens) + 1}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = ScratchTransformer(**network).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise QueryError("Output directory must be empty; choose a new directory to preserve earlier checkpoints")
    config = {"architecture": "scratch-transformer-v1", "network": network,
              "max_input_tokens": args.max_input_tokens, "max_output_tokens": args.max_output_tokens,
              "seed": args.seed, "plan_version": 1, "training_args": vars(args)}
    print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()), "vocab_size": len(tokenizer), "device": device}), flush=True)
    best, stale = float("inf"), 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_sum = train_tokens = 0
        for source, decoder_input, labels in train_loader:
            source, decoder_input, labels = [t.to(device) for t in (source, decoder_input, labels)]
            optimizer.zero_grad(set_to_none=True)
            logits = model(source, decoder_input)
            loss = F.cross_entropy(logits.reshape(-1, len(tokenizer)), labels.reshape(-1), ignore_index=0)
            if not torch.isfinite(loss):
                raise QueryError("Training loss became nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = labels.ne(0).sum().item()
            train_sum += loss.item() * count
            train_tokens += count
        model.eval()
        val_sum = val_tokens = 0
        with torch.inference_mode():
            for source, decoder_input, labels in val_loader:
                source, decoder_input, labels = [t.to(device) for t in (source, decoder_input, labels)]
                logits = model(source, decoder_input)
                val_sum += F.cross_entropy(logits.reshape(-1, len(tokenizer)), labels.reshape(-1), ignore_index=0, reduction="sum").item()
                val_tokens += labels.ne(0).sum().item()
        validation_loss = val_sum / val_tokens
        metrics = {"epoch": epoch + 1, "train_loss": train_sum / train_tokens, "validation_loss": validation_loss}
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if validation_loss < best:
            best, stale = validation_loss, 0
            torch.save(model.state_dict(), output / "model.pt")
            tokenizer.save(output / "tokenizer.json")
            (output / "label_sql_config.json").write_text(json.dumps(config, indent=2))
            (output / "training_schema.json").write_text(json.dumps(schema.data, indent=2))
        else:
            stale += 1
        (output / "training_history.json").write_text(json.dumps(history, indent=2))
        if stale >= args.patience:
            print("Early stopping: validation loss stopped improving.")
            break
    print(f"Best checkpoint saved to {output}; evaluate generated plans before using it.")


if __name__ == "__main__":
    main()
