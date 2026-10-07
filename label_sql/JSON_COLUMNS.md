# JSON/JSONB properties as labeled virtual columns (v0.3)

You can expose a scalar JSON property as a column in the schema used for training and prediction. No physical database column is created. The model refers to its business label; the compiler extracts and casts the JSON value in PostgreSQL.

## Define the physical container and virtual columns

Suppose `employees.profile` is JSONB and contains:

```json
{"address":{"city":"Chennai"},"bonus":12500,"joined_on":"2025-01-15","active":true}
```

Add the following entries to the employees table's `columns` list:

```json
[
  {"name":"profile","label":"Employee profile","type":"JSONB"},
  {
    "name":"profile_city",
    "label":"Employee city",
    "type":"text",
    "json":{"column":"profile","path":["address","city"]}
  },
  {
    "name":"profile_bonus",
    "label":"Annual bonus",
    "type":"numeric",
    "json":{"column":"profile","path":["bonus"]}
  }
]
```

`name` is a unique internal identifier within the table. For a virtual column it does not name a physical database column. `label` is what the model and user refer to. `json.column` names a physical JSON/JSONB column in the same table. `json.path` is an array of literal keys or integer array indexes, for example `["addresses", 0, "city"]`. Keys containing dots remain a single key: `["home.city"]` differs from `["home", "city"]`.

The source column must also be declared with type `JSON` or `JSONB`. Columns storing JSON inside TEXT, chained virtual sources, JSONPath wildcards, and array expansion into multiple rows are not supported. Empty string keys are supported. The path must contain 1–32 elements; booleans are not accepted as array indexes.

Virtual column type is required. Supported scalar types:

`text`, `integer`, `bigint`, `numeric`, `decimal`, `real`, `double precision`, `boolean`, `date`, `timestamp`, `timestamptz`, `uuid`.

Use `text` for IDs whose leading zeros must remain significant. Use `numeric` for amounts. These are allowlisted type names; arbitrary SQL type/expression text is rejected. Column aliases work the same way as for physical columns.

## Use the labels in training

The question/plan format is unchanged:

```json
{
  "question":"Show employee name and annual bonus where employee city is Chennai",
  "plan":{
    "select":[{"column":"Employee name"},{"column":"Annual bonus"}],
    "where":{"column":"Employee city","op":"eq","value":"Chennai"}
  }
}
```

The model sees labels, types, and table relationships in its schema context. It does not need to generate JSON paths or PostgreSQL extraction syntax. The compiler obtains the path from the current schema.

Add question/plan examples using your new labels and train a fresh checkpoint. The current training command does not resume an existing model. Simply modifying the metadata does not establish that an older checkpoint will interpret new labels correctly.

## SQL generation

The expressions corresponding to the examples above are equivalent to:

```sql
profile #>> '{address,city}'
(profile #>> '{bonus}')::numeric
```

The compiler uses bound `text[]` paths rather than embedding path strings in SQL:

```sql
SELECT t0."full_name", ((t0."profile" #>> %(p0)s::text[])::numeric)
FROM "public"."employees" AS t0
WHERE (t0."profile" #>> %(p1)s::text[]) = %(p2)s
LIMIT %(p3)s
```

```json
{"p0":["bonus"],"p1":["address","city"],"p2":"Chennai","p3":100}
```

Pass SQL and parameters separately to Psycopg if you integrate execution. This project itself only produces SQL. The same JSON expression/path binding is reused when a column appears in SELECT, GROUP BY, WHERE, HAVING, or ORDER BY.

## Include mappings during database export

In the exporter's labels override file, add `json_columns` under the table:

```json
{
  "public.employees": {
    "label":"Employees",
    "columns":{"full_name":"Employee name"},
    "json_columns":[
      {
        "name":"profile_city",
        "label":"Employee city",
        "type":"text",
        "json":{"column":"profile","path":["address","city"]}
      }
    ]
  }
}
```

```bash
label-sql-export --schemas public \
  --labels data/labels_json_fields.example.json \
  --out data/my_schema.json
```

The bundled mapping assumes the demo tables and the added `profile` column. Adapt the mapping before using another database. Export reflects the physical JSON/JSONB source and merges your declared virtual columns. It reads no row data and does not guess keys or types from values. Invalid source columns, duplicate column names, and unsupported virtual types fail schema validation.

## Run the JSON demo

For compiler-only validation:

```bash
python -m label_sql.cli compile \
  --schema data/schema_json_fields.json \
  --plan data/example_json_plan.json
```

For training from scratch:

```bash
python -m label_sql.train \
  --schema data/schema_json_fields.json \
  --train data/json_fields_train.jsonl \
  --validation data/json_fields_validation.jsonl \
  --output artifacts/label-sql-json
```

Then:

```bash
label-sql predict \
  --schema data/schema_json_fields.json \
  --checkpoint artifacts/label-sql-json \
  --question 'Show employee name and annual bonus where employee city is Chennai'

python -m label_sql.evaluate \
  --schema data/schema_json_fields.json \
  --checkpoint artifacts/label-sql-json \
  --data data/json_fields_test.jsonl
```

The JSON example dataset contains 222 training pairs, 3 validation pairs, and 3 test pairs. These demonstrate usage, not measured model quality. Review synthetic pairs before training. To generate your own starting examples, use `bootstrap.py` with your JSON-aware schema; its existing logic already includes these virtual labels.

`data/demo_json_fields.sql` is an optional database fixture extension, intended only after `demo_database.sql` in a disposable database. Neither script runs automatically.

## Behavior and boundaries

- Missing paths, a NULL source, and extracted JSON null produce SQL NULL. `is_null` matches these cases collectively; this grammar does not distinguish them.
- Values that cannot be cast to the declared type cause PostgreSQL errors, including an empty string cast to numeric. This version does not silently replace malformed values with NULL or zero.
- `numeric` gives numeric comparisons and sorting; leaving numeric-looking values as `text` gives lexical ordering.
- JSON columns may be selected/filtered/grouped/aggregated while joining other tables through ordinary declared physical foreign keys. A virtual JSON property cannot itself be used as a relation endpoint in this version.
- Fixed array indexes return a single value. Expanding every array item into rows requires a different query grammar.
- Query validity does not guarantee that the trained model chose the right labels or operations.

Validation: 36 tests passed; 3 PyTorch tests skipped because PyTorch is unavailable. JSON tests cover extraction, numeric/date/boolean types, grouping, joins, binding unusual keys/indexes, invalid metadata, exporter mapping and training token limits. SQL has not been executed against a live PostgreSQL instance here; model training remains unverified.

PostgreSQL reference: https://www.postgresql.org/docs/current/functions-json.html
