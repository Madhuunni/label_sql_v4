# Label SQL: train-from-scratch PyTorch English-to-PostgreSQL (v0.3)

An executable starter project that maps business labels in an English question to a structured query plan, validates the plan, resolves declared table relationships, and returns a parameterized PostgreSQL SELECT. It also exports PostgreSQL catalog metadata to editable schema JSON.

**This archive contains source and demonstration data, not trained weights.** A schema describes what columns and relationships exist. It does not teach a neural model how English questions map to query operations. Training therefore uses BOTH the schema JSON and supervised question/plan JSONL examples. The included synthetic examples are a starting point, not an accuracy benchmark or a production training corpus.

JSON/JSONB properties can now be represented as labeled virtual columns. See [JSON_COLUMNS.md](JSON_COLUMNS.md) for schema mappings, exporter options, training examples, commands and limitations. The model still trains from scratch with no external pretrained models.

## 1. Architecture

1. Read the schema JSON, including physical identifiers, business labels, aliases, and composite foreign keys.
2. Include the label-based schema context with the user's question.
3. A local PyTorch encoder-decoder model predicts a JSON plan using labels.
4. Validate labels, operations, grouping, limits, and the relationship graph.
5. Resolve labels to physical identifiers and assemble the required joins.
6. Return `sql`, `params`, `plan`, `tables`, and `relations`. **No generated query is executed.**

The model is a custom `ScratchTransformer` built using PyTorch `nn.Transformer`. All learned weights start randomly. There is **no pretrained model, no Hugging Face dependency, no pretrained tokenizer, and no model download or external inference call**. Only PyTorch is needed for the model; SQLAlchemy/Psycopg are needed for metadata export.

Default architecture: 3 encoder layers and 3 decoder layers, embedding size 256, 8 attention heads, feed-forward size 1024, sinusoidal positions, dropout 0.1. These dimensions are configurable. The network initially has no English or SQL knowledge. It must learn the domain language and JSON plan structure from supervised examples. Training from scratch requires much more varied supervision than fine-tuning a pretrained language model.

A successful compile means the plan fits this grammar and schema. It does NOT prove that the model understood the question correctly, selected the correct grain, or used the right business definition. Review predictions and measure accuracy on your own held-out questions.

## 2. Files

| File | Purpose |
| --- | --- |
| `label_sql/export_schema.py` | PostgreSQL metadata export; CLI or existing SQLAlchemy Connection |
| `label_sql/schema.py` | Schema validation, label resolution, unique shortest join paths |
| `label_sql/compiler.py` | Closed plan grammar → parameterized PostgreSQL SELECT |
| `label_sql/network.py` | Randomly initialized PyTorch encoder-decoder Transformer |
| `label_sql/tokenizer.py` | Locally learned pieces plus UTF-8 byte fallback |
| `label_sql/model.py` | Local checkpoint loading, schema prompts, prediction and strict JSON parsing |
| `label_sql/train.py` | Explicit PyTorch supervised training and best validation-loss checkpoint |
| `label_sql/evaluate.py` | Held-out plan validity, exact plan, and compiled SQL/parameter matches |
| `label_sql/bootstrap.py` | Synthetic starter training pairs from a schema |
| `label_sql/cli.py` | Compile a plan or predict from English |
| `data/schema.json` | Three-table schema example |
| `data/labels.example.json` | Business label overrides for the exporter |
| `data/train.jsonl` | 161 demonstration training pairs: synthetic plus hand-authored |
| `data/validation.jsonl`, `data/test.jsonl` | Small separate demonstration sets (6 and 5 questions) |
| `data/example_plan.json` | Three-table SELECT with filters and ordering |
| `data/demo_database.sql` | Optional disposable PostgreSQL fixture; never run automatically |
| `tests/` | Compiler/schema/export contract tests and optional model wiring smoke test |

## 3. Installation

Python 3.10+ is required. Run from the extracted `label_sql` directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
```

For GPU training, install a PyTorch build appropriate to your CUDA/driver setup from the official PyTorch installation selector. CPU also works but is slower. Start with batch size 2; reduce to 1 if memory is insufficient. Sequence length and schema size strongly affect memory use.

If you only need the exporter:

```bash
python -m pip install -e '.[db]'
```

The schema reader, compiler, synthetic generator, and most tests use only the Python standard library. You can run those directly from the project directory without installing model dependencies.

Dependency ranges in `pyproject.toml` are compatibility targets, not a fully tested lockfile. After validating your installation, record its resolved versions for repeatable deployments.

## 4. Try SQL generation immediately, without a model

```bash
python -m label_sql.cli compile \
  --schema data/schema.json \
  --plan data/example_plan.json
```

This validates the compiler only. The English question corresponding to the example plan is:

> Show employee name, department name and overpayment amount for open cases over 5000, highest overpayment first.

Output SQL:

```sql
SELECT t0."full_name", t2."department_name", t1."overpayment_amount"
FROM "public"."employees" AS t0
INNER JOIN "public"."cases" AS t1 ON t1."employee_id" = t0."employee_id"
INNER JOIN "public"."departments" AS t2 ON t0."department_id" = t2."department_id"
WHERE (t1."status" = %(p0)s AND t1."overpayment_amount" > %(p1)s)
ORDER BY t1."overpayment_amount" DESC
LIMIT %(p2)s
```

Parameters:

```json
{"p0": "Open", "p1": 5000, "p2": 100}
```

The `%(p0)s` syntax is Psycopg binding syntax. The SQL template and parameter mapping are a pair; do not interpolate values with Python string formatting. It is not a literal SQL script to paste into psql. Physical identifiers are quoted, and values never enter the SQL text. Literal percent signs inside unusual identifiers are doubled for Psycopg's parameter parser.

## 5. Export your PostgreSQL schema

Supply a SQLAlchemy PostgreSQL URL through an environment variable. Use a database account with only the metadata/data access you intend to expose. The CLI starts a read-only transaction and reads catalog metadata, not table rows.

```bash
export DATABASE_URL='postgresql+psycopg://db_user:YOUR_PASSWORD@localhost:5432/your_database'
label-sql-export --schemas public --out data/my_schema.json
```

URL-encode reserved characters in URL credentials, or construct a SQLAlchemy `URL.create(...)` object in Python. Do not commit credentials.

Multiple PostgreSQL namespaces:

```bash
label-sql-export --schemas hr finance --out data/my_schema.json
```

Set business labels through an override file:

```bash
label-sql-export --schemas public \
  --labels data/labels.example.json \
  --out data/my_schema.json
```

The bundled override file refers to the demo tables; replace it with your real table/column names before using it against another database.

An existing SQLAlchemy connection is supported directly:

```python
import json
from sqlalchemy import create_engine
from label_sql.export_schema import export_schema

engine = create_engine(database_url)
try:
    with engine.connect() as connection:
        connection.exec_driver_sql("SET TRANSACTION READ ONLY")
        schema_dict = export_schema(connection, schemas=["public"], overrides={})
    with open("schema.json", "w", encoding="utf-8") as file:
        json.dump(schema_dict, file, indent=2, ensure_ascii=False)
finally:
    engine.dispose()
```

The utility exports ordinary tables by default. `--include-views` includes ordinary views, whose logical relationships generally need manual declaration. Materialized views are not included in this starter. Only tables visible to the database user are exported. Foreign keys pointing outside the selected schemas are listed under `omitted_relations`; include those namespaces if the joins are needed. Constraints unavailable to the user cannot be discovered.

Label selection order:

1. Your override file.
2. A database comment explicitly starting with `label:`, e.g. `label: Employee name`.
3. A title-cased physical name. Table defaults include the namespace to reduce duplicates.

PostgreSQL cannot infer your business vocabulary. Review/edit generated labels and aliases. No row values, database URL, or password are stored in the exported JSON. Existing column/table comments are included; review them before sharing metadata.

## 6. Schema JSON format

Minimal two-table example:

```json
{
  "version": 1,
  "dialect": "postgresql",
  "tables": [
    {
      "id": "public.employees",
      "schema": "public",
      "name": "employees",
      "label": "Employees",
      "primary_key": ["employee_id"],
      "columns": [
        {"name": "employee_id", "label": "Employee ID", "type": "INTEGER", "nullable": false},
        {"name": "full_name", "label": "Employee name", "aliases": ["Staff name"], "type": "TEXT"},
        {"name": "department_id", "label": "Employee department ID", "type": "INTEGER"}
      ]
    },
    {
      "id": "public.departments",
      "schema": "public",
      "name": "departments",
      "label": "Departments",
      "primary_key": ["department_id"],
      "columns": [
        {"name": "department_id", "label": "Department ID", "type": "INTEGER"},
        {"name": "department_name", "label": "Department name", "type": "TEXT"}
      ]
    }
  ],
  "relations": [
    {
      "id": "employees_department_fk",
      "from_table": "public.employees",
      "from_columns": ["department_id"],
      "to_table": "public.departments",
      "to_columns": ["department_id"]
    }
  ]
}
```

Composite foreign keys use arrays of matching column pairs in the same order. Relations may be manually added when your application knows relationships that have no database foreign-key constraint. Do not infer relationships merely because column names happen to match.

Table labels must be unique. Column labels may repeat across tables, but then refer to them as `Employees.ID` or `Departments.ID`. Labels and aliases match case-insensitively with whitespace normalized. A dot is reserved for qualification and must not occur in table or column labels. Physical schema/table/column identifiers remain separate from labels.

## 7. Training data: what the model actually learns

Every JSONL line contains an English question and its correct label-based plan:

```json
{"question":"Show employee name and department name for salary above 60000","plan":{"select":[{"column":"Employee name"},{"column":"Department name"}],"where":{"column":"Salary","op":"gt","value":60000}}}
```

During training:

- Encoder input: task prefix + label/schema context + English question.
- Decoder target: compact JSON query plan.
- Loss: token-level cross entropy; padding token 0 is ignored.
- Optimizer: PyTorch AdamW; gradient clipping is enabled.
- Checkpoint: lowest validation loss across requested epochs.
- Tokenizer: learns common pieces from training inputs/targets only. Unseen pieces fall back to UTF-8 bytes; digits remain separate to represent unseen IDs and leading zeros.
- Decoder inputs start with BOS and receive previous correct output tokens during training (teacher forcing); targets finish with EOS. A causal mask prevents seeing future target tokens.
- Validation loss is averaged over non-padding tokens. Default early stopping: 15 epochs without improvement.
- Tokenizer coverage does not guarantee correct copying; the model must still learn to reproduce labels and literal values.

The model learns label selection, predicates, aggregates, grouping and ordering from the examples. It does not learn physical join SQL from scratch. The compiler resolves the actual join relationships from your authoritative JSON at prediction time.

For your schema, generate a first draft of examples:

```bash
python -m label_sql.bootstrap --schema data/my_schema.json --out data/generated_train.jsonl
```

Review the output. Synthetic placeholders such as Active, Pending and Chennai may not be meaningful for your columns. Add business-relevant examples, realistic values, aliases, multi-table questions, combined predicates, aggregates, and LEFT JOIN questions. The synthetic generator does not cover every supported plan feature; manually add those examples. Cross-column combinations are capped to keep generation bounded.

Create separate `my_validation.jsonl` and `my_test.jsonl`. Hold out entire paraphrase families and combinations where possible, not just different numbers in otherwise identical templates. The bundled sets demonstrate file formats; their tiny size is not sufficient to estimate real-world accuracy. There is no universal minimum example count or promised accuracy.

## 8. Train from scratch

The 161 demonstration training pairs illustrate the workflow. They do not establish broad English understanding. Collect varied domain questions and correct plans, then measure performance on unseen questions. More epochs cannot compensate for missing examples.

```bash
python -m label_sql.train \
  --schema data/schema.json \
  --train data/train.jsonl \
  --validation data/validation.jsonl \
  --epochs 100 \
  --batch-size 4 \
  --d-model 256 \
  --heads 8 \
  --layers 3 \
  --ff-size 1024 \
  --output artifacts/label-sql-scratch
```

Substitute your schema and training/validation files. Training validates every plan, rejects exact overlap between training/validation questions, and prints the actual parameter count. The output directory must be empty so existing checkpoints are preserved.

`--layers` sets both encoder and decoder depth. Defaults: `--lr 0.0003`, `--dropout 0.1`, `--patience 15`. Use `--device cpu` or `--device cuda` explicitly, or allow automatic CUDA selection. Dependencies must first be installed; training and prediction then work offline.

Checkpoint files:

- `model.pt`: your learned PyTorch weights.
- `tokenizer.json`: your locally learned vocabulary.
- `label_sql_config.json`: architecture and training configuration.
- `training_schema.json`: training metadata snapshot.
- `training_history.json`: losses for each epoch.

Input and output limits default to 2048 and 1024 tokens. Oversized examples raise an error; nothing is silently truncated. For large databases, curate a business-area schema including required bridge tables. Automatic schema retrieval is not included. Raising limits increases attention memory use.

Training is full-precision and single-process. Optimizer resume, distributed training and quantization are not implemented. Earlier pretrained-model checkpoints are incompatible; train fresh in a new output directory.

## 9. Predict SQL from English

After training:

```bash
label-sql predict \
  --schema data/schema.json \
  --checkpoint artifacts/label-sql-scratch \
  --question 'Show employee name and department name for salary above 60000'
```

Python interface (load the model once per process):

```python
from label_sql import Schema
from label_sql.model import Predictor

schema = Schema.load("data/my_schema.json")
predictor = Predictor("artifacts/label-sql-scratch")
result = predictor.predict(schema, "Show employee name and department name")
print(result["sql"])
print(result["params"])
```

Prediction loads local files only. Greedy decoding reuses encoder output and predicts one token at a time until EOS. Generated output must pass JSON and compiler validation. Invalid output raises `QueryError`; there is no fallback that invents a query. Unsupported questions can still produce an incorrect but valid plan: this version has no calibrated confidence score or learned abstention mechanism.

If integrating execution separately, pass `result['sql']` and `result['params']` as separate arguments to a Psycopg cursor. Apply application authorization, a read-only database role/transaction, query timeout, and your tenant policy before executing. A row limit does not cap scan cost. This project deliberately returns SQL only.

Changing physical names while preserving labels can work through compiler remapping. New labels or changed business meanings require fresh evaluation and additional training from scratch in this version. The training schema is saved with the checkpoint for comparison, but prediction uses the schema supplied to it.

## 10. Supported plan grammar

| Field | Accepted form |
| --- | --- |
| `from` | Optional table LABEL; required for unqualified count(*) and LEFT JOIN |
| `select` | Nonempty array of `{column: label, aggregate?: count/sum/avg/min/max}` |
| `where` | Predicate or nested `{and: [...]}` / `{or: [...]}` |
| Predicate | `{column: label, op: operator, value: scalar}`; null checks omit `value` |
| Operators | `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `like`, `ilike`, `in`, `is_null`, `is_not_null` |
| `in` value | Nonempty array of at most 500 scalar values |
| `group_by` | Array of column labels |
| `having` | Same predicate syntax, optionally with `aggregate` |
| `order_by` | Array of `{column: label, aggregate?: ..., direction?: asc/desc}` |
| `distinct` | Boolean, default false |
| `join_type` | `inner` (default) or `left`; applies to all inferred joins |
| `limit` | Integer 1–1000; default 100 |

Only `count` accepts the `*` column. There is no `SELECT *` projection. Dates are ISO-format strings bound as parameters. Explicit inclusive ranges use two predicates. Relative dates such as “last month” are not resolved against a clock; add a date-normalization component before promising that behavior.

Examples of additional plans:

```json
{
  "select": [
    {"column": "Department name"},
    {"column": "Overpayment amount", "aggregate": "sum"}
  ],
  "group_by": ["Department name"],
  "having": {"column": "Overpayment amount", "aggregate": "sum", "op": "gt", "value": 10000},
  "order_by": [{"column": "Overpayment amount", "aggregate": "sum", "direction": "desc"}]
}
```

```json
{
  "from": "Employees",
  "select": [{"column": "Employee name"}, {"column": "Department name"}],
  "join_type": "left"
}
```

LEFT JOIN preserves the specified root while adding tables along its relationship tree. Predicates on the nullable side in WHERE can remove unmatched rows, as in normal SQL. The grammar does not express ON-clause filters, mixed join types, or role-specific joins.

## 11. Join rules and limitations

- Joins use declared relationships only, in either traversal direction, with all composite-key columns.
- Intermediate bridge tables are included even when none of their columns are selected.
- A unique shortest path is required from the root to each requested table. Multiple equally short paths, including two foreign keys between the same table pair, are rejected.
- Unique shortest does not imply the correct business relationship. Curate the schema to the intended business path and review generated joins.
- Without `from`, the first resolved column determines the root. Specify `from` when root semantics matter.
- Self-joins and repeated roles such as billing/shipping addresses are not supported. Exported self-FKs remain in metadata, but the planner does not traverse them.
- CTEs, subqueries, UNION, window functions, calculated expressions, arbitrary functions, COUNT DISTINCT, user-defined output aliases, and DML/DDL are outside the plan grammar.
- The validator checks structural SQL rules, not all PostgreSQL type/operator compatibility. PostgreSQL may still reject unsuitable aggregates, comparisons or values. Add domain type validation if needed.
- A one-to-many join can multiply rows and inflate aggregates. Training examples must specify the intended grain; complex pre-aggregation needs a richer grammar.
- The code does not implement row-level authorization or tenant filtering. Limit exposed metadata and enforce access at the database/application boundary.

## 12. Evaluation and tests

Evaluate the generated plans, separately from training loss:

```bash
python -m label_sql.evaluate \
  --schema data/schema.json \
  --checkpoint artifacts/label-sql-scratch \
  --data data/test.jsonl
```

The report includes valid-plan rate, exact JSON-plan accuracy, and exact compiled SQL + parameter match. Semantically equivalent plans may differ on these strict metrics. Execution-result accuracy needs a controlled PostgreSQL test database and is not measured by this command.

Run local tests:

```bash
python -m unittest discover -s tests -v
```

Validation performed for v0.3: **36 standard-library tests passed; 3 optional PyTorch tests skipped** because PyTorch was unavailable. Checks cover the original 172 target plans and 228 additional JSON-field target plans, tokenizer round trips for unseen numeric and Unicode text, default sequence lengths, and compiler/exporter behavior. Python syntax was checked. The optional network tests cover forward/backward, causal masking, generation and a one-epoch checkpoint reload; they were NOT run here. Exporter tests use a fake inspector, not live PostgreSQL. Actual training, model accuracy and live database reflection remain unverified. No trained weights are included.

## 13. Changes in v0.2

- Removed pretrained-model loading and Transformers/SentencePiece/Safetensors dependencies.
- Added `network.py` and `tokenizer.py` for a randomly initialized model and a local tokenizer.
- Replaced training, prediction and checkpoint handling.
- Updated dependencies, documentation and tests.
- Kept schema export, label resolution, SQL compilation and the question/plan data format.

## 14. Primary documentation

- PyTorch: https://pytorch.org/get-started/locally/
- Transformer building block: https://docs.pytorch.org/docs/stable/generated/torch.nn.Transformer.html
- SQLAlchemy reflection: https://docs.sqlalchemy.org/en/20/core/reflection.html
- Psycopg parameters: https://www.psycopg.org/psycopg3/docs/basic/params.html

## 15. Changes in v0.3

Added typed JSON/JSONB virtual columns in `schema.py`, expression generation in `compiler.py`, and `json_columns` mapping support in `export_schema.py`. Added JSON example schema, override mapping, query plan, train/validation/test datasets, optional database fixture, documentation and tests. The network and training-plan format are unchanged.

## Large schema-based training datasets

See [TRAINING_GENERATOR.md](TRAINING_GENERATOR.md) for 1,200 or more examples per column, table filtering, JSON labels, and sample-value customization.

## Token-limit errors

The complete schema is included with **each** question. Generating examples with
`--tables` does not reduce the schema passed to training. The limit is per sequence,
not a limit on the number of training examples. Input defaults to 2048 tokens;
output defaults to 1024. Local tokens are not equivalent to words.

Measure actual lengths with the training-fitted tokenizer (no PyTorch needed):

```bash
python -m label_sql.train --schema data/my_schema.json --train data/train.jsonl --validation data/validation.jsonl --check-only
```

Train using explicit limits at least as large as `required_input_tokens` and
`required_output_tokens`, or opt in to increasing the limits automatically:

```bash
python -m label_sql.train --schema data/my_schema.json --train data/train.jsonl --validation data/validation.jsonl --auto-token-limits --batch-size 1 --output artifacts/label-sql-long
```

No schema text or target plan is truncated. Effective limits are saved in the
checkpoint, along with sufficient positional capacity. This measures the provided
train/validation examples; future longer questions or expanded schemas can still
exceed the saved limit. Set a larger explicit input limit for desired headroom.
Long sequences use substantially more attention memory; automatic sizing does not
guarantee they fit on your GPU. For very large schemas, curate a smaller domain
schema retaining required columns, JSON source columns, and join relationships,
and use the same scope for training and inference. Do not randomly discard tables.
Do not edit an existing checkpoint config to raise its positional capacity: train
a new checkpoint with the desired limits. Use a fresh/empty output directory.
