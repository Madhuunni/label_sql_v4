"""Create deterministic synthetic question/plan pairs from schema labels."""
import argparse
import json
import random
import re
from pathlib import Path
from .schema import Schema, QueryError, norm
from .compiler import compile_plan

REQUESTS = ['Show {}', 'List {}', 'Display {}', 'Retrieve {}', 'Fetch {}', 'Return {}',
            'Please show {}', 'Please list {}', 'Can you display {}?',
            'Could you retrieve {}?', 'I need {}', 'Get me {}']
LIMITS = [5,10,15,20,25,30,40,50,60,75,80,100,125,150,200,250,300,400,500,1000]


def kind(c):
    t = c.get('type', '').strip().lower()
    if '[]' in t or t.endswith('array'): return 'other'
    if re.match(r'^(smallint|integer|bigint|int\d*|numeric|decimal|real|double precision|float\d*|smallserial|serial|bigserial)\b', t): return 'number'
    if t in ('bool', 'boolean'): return 'boolean'
    if t == 'date' or t.startswith('timestamp'): return 'date'
    if t == 'uuid': return 'uuid'
    if t == 'text' or t.startswith(('varchar', 'character', 'char(')): return 'text'
    return 'other'


def select_tables(schema, selectors):
    if not selectors: return list(schema.tables)
    selected = []
    for value in selectors:
        hits = [tid for tid,t in schema.tables.items() if norm(value) in
                {norm(tid), norm(t['name']), norm(t['label']), norm(t['schema']+'.'+t['name'])}]
        if len(hits) != 1: raise QueryError(f'Unknown or ambiguous table: {value}')
        if hits[0] not in selected: selected.append(hits[0])
    return selected


def select_columns(schema, tables, selectors=None):
    """Resolve label or physical-name selectors within the selected tables."""
    available = [(tid, c) for tid in tables for c in schema.tables[tid]['columns']]
    if selectors is None:
        return {tid: list(schema.tables[tid]['columns']) for tid in tables}
    chosen = set()
    for value in selectors:
        hits = set()
        for tid, c in available:
            t = schema.tables[tid]
            names = [c['name'], c['label']] + c.get('aliases', [])
            qualifiers = {tid, t['schema']+'.'+t['name'], t['name'], t['label']}
            forms = names + [q+'.'+name for q in qualifiers for name in names]
            if norm(value) in {norm(form) for form in forms}:
                hits.add((tid, c['name']))
        if len(hits) != 1:
            raise QueryError(f"Unknown or ambiguous column within selected tables: {value!r}. "
                             "Use 'Table label.Column label' or 'schema.table.column'.")
        chosen.update(hits)
    if not chosen:
        raise QueryError('Select at least one column')
    return {tid: [c for c in schema.tables[tid]['columns'] if (tid,c['name']) in chosen]
            for tid in tables}


def variants(ref, c, samples):
    base = {'select': [{'column': ref}]}
    yield f'values of {ref}', base, 'select'
    for op,word in [('is_null','missing'),('is_not_null','not missing')]:
        yield f'values of {ref} where {ref} is {word}', dict(base, where={'column':ref,'op':op}), op
    yield f'the count of non-null values of {ref}', {'select':[{'column':ref,'aggregate':'count'}]}, 'count'
    k = kind(c)
    # Opaque types and raw JSON get projection/null/count operations only.
    if k == 'other': return
    yield f'distinct values of {ref}', dict(base, distinct=True), 'distinct'
    for direction,word in [('asc','ascending'),('desc','descending')]:
        yield f'values of {ref} sorted in {word} order', dict(base, order_by=[{'column':ref,'direction':direction}]), 'sort'
    defaults = {'number':[0,10,100,500,1000], 'text':['Sample A','Sample B','Sample C'],
                'boolean':[True,False], 'date':['2024-01-01','2025-06-15','2026-12-31'],
                'uuid':['00000000-0000-4000-8000-000000000001']}
    values = samples.get(ref, defaults[k])
    if not isinstance(values,list): raise QueryError(f'Sample values for {ref} must be a list')
    for v in values:
        if not ((k=='number' and type(v) in (int,float)) or (k=='boolean' and type(v) is bool) or
                (k in ('text','date','uuid') and isinstance(v,str))):
            raise QueryError(f'Wrong sample value type for {ref}: {v!r}')
        ops = [('eq','equal to'),('ne','not equal to')]
        if k in ('number','date'):
            ops += [('gt','greater than'),('gte','greater than or equal to'),('lt','less than'),('lte','less than or equal to')]
        for op,word in ops:
            yield f'values of {ref} where {ref} is {word} {json.dumps(v,ensure_ascii=False)}', dict(base,where={'column':ref,'op':op,'value':v}), 'filter'
    if k == 'number':
        for agg,word in [('sum','total'),('avg','average'),('min','minimum'),('max','maximum')]:
            yield f'the {word} of {ref}', {'select':[{'column':ref,'aggregate':agg}]}, 'aggregate'


def generate(schema, tables=None, per_column=1200, seed=42, samples=None, include_joins=False, columns=None):
    if per_column < 1200: raise QueryError('--per-column must be at least 1200')
    selected = select_tables(schema,tables)
    selected_columns = select_columns(schema, selected, columns)
    samples = {} if samples is None else samples
    known = {t['label']+'.'+c['label'] for t in schema.tables.values() for c in t['columns']}
    if not isinstance(samples,dict) or set(samples)-known:
        raise QueryError('Sample keys must be exact Table label.Column label references')
    rng = random.Random(seed)
    rows,coverage,seen = [],[],set()
    for tid in selected:
        t = schema.tables[tid]
        for c in selected_columns[tid]:
            ref = t['label']+'.'+c['label']
            bases = list(variants(ref,c,samples))
            if include_joins:
                for other in selected:
                    if other == tid: continue
                    try: schema.join_tree(tid,{other})
                    except QueryError: continue
                    for oc in selected_columns[other][:2]:
                        partner = schema.tables[other]['label']+'.'+oc['label']
                        bases.append((f'values of {ref} and {partner} for related records',
                                      {'select':[{'column':ref},{'column':partner}]},'join'))
            candidates = []
            for phrase,plan,family in bases:
                for limit in LIMITS:
                    for ending in [f', with a maximum of {limit} result rows', f', limited to {limit} output rows']:
                        for request in REQUESTS:
                            candidates.append((request.format(phrase+ending),dict(plan,limit=limit),family))
            rng.shuffle(candidates)
            count,families = 0,{}
            for question,plan,family in candidates:
                if question.casefold() in seen: continue
                compile_plan(schema,plan)
                seen.add(question.casefold())
                rows.append({'question':question,'plan':plan,'source':'synthetic; human review required',
                             'metadata':{'table_id':tid,'column_label':ref,'family':family}})
                families[family] = families.get(family,0)+1
                count += 1
                if count == per_column: break
            if count != per_column:
                raise QueryError(f'Only {count} unique questions available for {ref}; requested {per_column}')
            coverage.append({'table_id':tid,'column_label':ref,'count':count,'families':families})
    if not rows: raise QueryError('Selected tables have no columns')
    rng.shuffle(rows)
    return rows,{'seed':seed,'total':len(rows),'per_column':per_column,'coverage':coverage}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--schema',required=True)
    p.add_argument('--tables','--table',nargs='+',help='Table IDs, physical names, or table labels; default: all')
    p.add_argument('--columns','--column',nargs='+',help='Column labels or names; qualify with table for multiple tables')
    p.add_argument('--per-column',type=int,default=1200)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--sample-values',help='JSON mapping qualified labels to sample values')
    p.add_argument('--include-joins',action='store_true',help='Join selected tables along unambiguous relationship paths')
    p.add_argument('--out',required=True)
    a = p.parse_args()
    try:
        samples = json.loads(Path(a.sample_values).read_text(encoding='utf-8')) if a.sample_values else None
        rows,report = generate(Schema.load(a.schema),a.tables,a.per_column,a.seed,samples,a.include_joins,columns=a.columns)
        payload = ''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows)
        out = Path(a.out)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(payload,encoding='utf-8')
        report_path = out.with_suffix(out.suffix+'.report.json')
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    except (QueryError,OSError,ValueError) as error: p.error(str(error))
    print(f'Wrote {len(rows)} examples to {out}; coverage: {report_path}')
    print('Review synthetic examples. Use independently authored validation/test questions.')

if __name__ == '__main__': main()
