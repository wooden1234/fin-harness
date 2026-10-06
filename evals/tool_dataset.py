"""Validate/export the tool benchmark without credentials; explicitly upload if requested.

This is dataset plumbing, not a fixture runner or a success-rate evaluator.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
NAMES = {'full': 'fin-harness-tools-v1', 'smoke': 'fin-harness-tools-smoke-v1'}
BUCKETS = {'calculation', 'iwencai', 'financial_report', 'web_search', 'runtime_mcp'}
MODES = {'agent', 'tool_contract', 'runtime', 'answer_evaluation'}


def load_tool_cases(source: str = 'full') -> list[dict[str, Any]]:
    path = ROOT / 'datasets' / f'{NAMES[source]}.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    expected_count = 100 if source == 'full' else 3
    if len(rows) != expected_count:
        raise ValueError(f'{source}: expected {expected_count} cases, got {len(rows)}')
    seen: set[str] = set()
    for row in rows:
        cid = row['id']
        if not cid or cid in seen:
            raise ValueError(f'duplicate/empty case id: {cid}')
        seen.add(cid)
        inputs, fixture, outputs, meta = (row[k] for k in ('inputs', 'fixture', 'outputs', 'metadata'))
        if bool(inputs.get('query')) == bool(inputs.get('turns')):
            raise ValueError(f'{cid}: exactly one of query/turns required')
        if set(inputs) - {'query', 'turns'}:
            raise ValueError(f'{cid}: fixture/expected data must not enter model inputs')
        if 'turns' in inputs and not all(t.get('role') == 'user' and t.get('content') for t in inputs['turns']):
            raise ValueError(f'{cid}: invalid user turns')
        if meta['bucket'] not in BUCKETS or meta['execution_mode'] not in MODES:
            raise ValueError(f'{cid}: unknown bucket/mode')
        if fixture.get('synthetic') is not True or fixture.get('network') != 'deny' or not fixture.get('setup'):
            raise ValueError(f'{cid}: explicit synthetic offline fixture required')
        checks = outputs['assertions']
        if len(checks) < 2 or len({c['id'] for c in checks}) != len(checks):
            raise ValueError(f'{cid}: at least two uniquely named assertions required')
        if not isinstance(outputs['required_facts'], list):
            raise ValueError(f'{cid}: required_facts must be a list')
        if not all(c.get('requirement') and c.get('severity') == 'hard' and c.get('evaluation') in {'trace_or_code', 'trace_and_answer'} for c in checks):
            raise ValueError(f'{cid}: invalid assertion')
        if not outputs.get('global_gates'):
            raise ValueError(f'{cid}: missing global gates')
    if source == 'full' and Counter(r['metadata']['bucket'] for r in rows) != Counter({b: 20 for b in BUCKETS}):
        raise ValueError('full: expected 20 cases per bucket')
    if source == 'smoke':
        full_ids = {r['id'] for r in load_tool_cases('full')}
        if any(r['metadata']['parent_case_id'] not in full_ids for r in rows):
            raise ValueError('smoke: invalid parent case')
    return rows


def langsmith_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for row in rows:
        canonical = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        examples.append({
            'inputs': row['inputs'],
            'outputs': row['outputs'],
            'metadata': {
                **row['metadata'], 'case_id': row['id'], 'fixture': row['fixture'],
                'case_sha256': hashlib.sha256(canonical.encode()).hexdigest(),
                'schema_version': 'tool-benchmark-v1',
            },
        })
    return examples


def upload(source: str, examples: list[dict[str, Any]], dataset_name: str) -> dict[str, Any]:
    # Credentials are only loaded on explicit --upload; validation/export is offline.
    from dotenv import load_dotenv
    from langsmith import Client
    load_dotenv()
    client = Client()
    name = dataset_name or NAMES[source]
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(dataset_name=name, description='合成工具契约与稳定性评测；fixture必须由测试驱动注入，不能直接发送给模型。')
    existing = {str((e.metadata or {}).get('case_id')): e for e in client.list_examples(dataset_id=dataset.id)}
    for item in examples:
        prior = existing.get(item['metadata']['case_id'])
        if prior and (prior.metadata or {}).get('case_sha256') != item['metadata']['case_sha256']:
            raise ValueError(f"remote case drift: {item['metadata']['case_id']}; use a new --dataset-name version")
    pending = [e for e in examples if e['metadata']['case_id'] not in existing]
    if pending:
        client.create_examples(dataset_id=dataset.id, examples=pending)
    return {'dataset': name, 'created': len(pending), 'skipped': len(examples)-len(pending)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=NAMES, default='full')
    parser.add_argument('--export', type=Path, help='Write LangSmith example payloads as JSONL locally')
    parser.add_argument('--upload', action='store_true', help='Explicitly upload to LangSmith')
    parser.add_argument('--dataset-name', default='')
    args = parser.parse_args()
    rows = load_tool_cases(args.source)
    examples = langsmith_examples(rows)
    print(json.dumps({'validated': len(rows), 'buckets': dict(Counter(r['metadata']['bucket'] for r in rows)), 'modes': dict(Counter(r['metadata']['execution_mode'] for r in rows))}, ensure_ascii=False))
    if args.export:
        args.export.write_text(''.join(json.dumps(e, ensure_ascii=False)+'\n' for e in examples), encoding='utf-8')
    if args.upload:
        print(json.dumps(upload(args.source, examples, args.dataset_name), ensure_ascii=False))


if __name__ == '__main__':
    main()
