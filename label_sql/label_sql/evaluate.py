import argparse
import json
from .schema import Schema, QueryError
from .compiler import compile_plan
from .model import Predictor
from .train import read_examples


def main():
    parser = argparse.ArgumentParser(description="Evaluate held-out questions, without executing SQL")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    schema = Schema.load(args.schema)
    predictor = Predictor(args.checkpoint)
    examples = read_examples(args.data, schema)
    valid = exact = sql_match = 0
    errors = []
    for row in examples:
        try:
            predicted = predictor.predict(schema, row["question"])
            expected = compile_plan(schema, row["plan"])
            valid += 1
            exact += predicted["plan"] == row["plan"]
            sql_match += (predicted["sql"], predicted["params"]) == (expected["sql"], expected["params"])
        except (QueryError, KeyError, TypeError) as exc:
            errors.append({"question": row["question"], "error": str(exc)})
    n = len(examples)
    print(json.dumps({"examples": n, "valid_plan_rate": valid / n, "exact_plan_accuracy": exact / n,
                      "compiled_sql_and_params_match": sql_match / n, "errors": errors}, indent=2))


if __name__ == "__main__":
    main()
