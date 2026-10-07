import argparse
import json
import sys
from pathlib import Path
from .schema import Schema, QueryError
from .compiler import compile_plan


def main():
    parser = argparse.ArgumentParser(description="Generate PostgreSQL SQL without executing it")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["compile", "predict"]:
        p = sub.add_parser(command)
        p.add_argument("--schema", required=True)
        if command == "compile":
            p.add_argument("--plan", required=True)
        else:
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--question", required=True)
    args = parser.parse_args()
    try:
        schema = Schema.load(args.schema)
        if args.command == "compile":
            result = compile_plan(schema, json.loads(Path(args.plan).read_text()))
        else:
            from .model import Predictor
            result = Predictor(args.checkpoint).predict(schema, args.question)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (QueryError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
