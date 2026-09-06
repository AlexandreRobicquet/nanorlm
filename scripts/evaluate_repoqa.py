"""Run the frozen repository QA comparison; gold facts never enter model prompts."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from artifacts import artifact_path, write_text_atomic
from repoqa import digest, git_value, load_evidence, run_question


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    write_text_atomic(path, json.dumps(value, indent=2) + '\n')


def validate_dataset(dataset: dict, repositories: Path) -> dict[str, Path]:
    if dataset['schema'] != 'nanorlm-repoqa-eval-v1':
        raise ValueError('unknown evaluation schema')
    roots = {}
    for name, record in dataset['repositories'].items():
        if not re.fullmatch(r'[a-z0-9_-]+', name):
            raise ValueError('unsafe repository name')
        root = repositories / name
        if git_value(root, 'rev-parse', 'HEAD') != record['commit'] or git_value(root, 'status', '--porcelain') != '':
            raise ValueError(f'{name}: expected clean pinned repository')
        roots[name] = root
    seen = set()
    for task in dataset['tasks']:
        if not re.fullmatch(r'[a-z0-9_-]+', task['id']) or task['id'] in seen:
            raise ValueError('unsafe or duplicate task identity')
        seen.add(task['id'])
        root = roots[task['repository']]
        for span in task['reference_spans']:
            path = artifact_path(root, span['path'])
            if file_hash(path) != span['file_sha256']:
                raise ValueError(f"{task['id']}: reference file hash mismatch")
            with path.open(newline='') as handle:
                lines = handle.read().splitlines(keepends=True)
            if ''.join(lines[span['start']-1:span['end']]) != span['text']:
                raise ValueError(f"{task['id']}: reference excerpt mismatch")
    return roots


def verified_receipt(directory: Path, task: dict, strategy: str, binding: str) -> dict:
    bound = json.loads((directory / 'binding.json').read_text())
    if bound != {'experiment_sha256': binding, 'task_id': task['id'], 'strategy': strategy}:
        raise ValueError('case binding mismatch')
    checksums = json.loads((directory / 'checksums.json').read_text())
    actual = {path.name: file_hash(path) for path in directory.iterdir()
              if path.is_file() and path.name != 'checksums.json'}
    if actual != checksums:
        raise ValueError('case checksum inventory mismatch; partial runs cannot be resumed')
    run = json.loads((directory / 'run.json').read_text())
    if run['question'] != task['question'] or run['strategy'] != strategy or run['status'] == 'started':
        raise ValueError('case identity or completion mismatch')
    evidence = load_evidence(directory / 'evidence.json')
    if evidence['bundle_sha256'] != run['evidence_sha256']:
        raise ValueError('receipt evidence binding mismatch')
    return run


def evaluate(dataset_path: Path, repositories: Path, output: Path, *, resume: bool = False) -> dict:
    data = json.loads(dataset_path.read_text())
    roots = validate_dataset(data, repositories)
    protocol = data['protocol']
    if protocol['strategies'] != ['lexical', 'full', 'retention']:
        raise ValueError('expected three frozen strategies')
    if git_value(ROOT, 'status', '--porcelain') != '':
        raise ValueError('evaluation implementation checkout must be clean')
    for name in ('repoqa.py', 'nanorlm.py', 'policies.py', 'learned_retention.py'):
        frozen = subprocess.run(['git', '-C', str(ROOT), 'show', f"{data['implementation_commit']}:{name}"],
                                check=True, capture_output=True).stdout
        if hashlib.sha256(frozen).hexdigest() != file_hash(ROOT / name):
            raise ValueError(f'implementation changed since freeze: {name}')
    output = artifact_path(output)
    manifest = {'schema': 'nanorlm-repoqa-experiment-v1', 'dataset_sha256': file_hash(dataset_path),
                'implementation_commit': data['implementation_commit'],
                'runner_commit': git_value(ROOT, 'rev-parse', 'HEAD'),
                'runner_sha256': file_hash(Path(__file__)), 'protocol': protocol,
                'repositories': data['repositories']}
    binding = digest(manifest)
    manifest['experiment_sha256'] = binding
    if output.exists() and any(output.iterdir()):
        if not resume or json.loads((output / 'experiment.json').read_text()) != manifest:
            raise ValueError('output must be empty or an exact matching --resume experiment')
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'experiment.json', manifest)
    rows, spent = [], 0.0
    for index, task in enumerate(data['tasks']):
        strategies = protocol['strategies']
        for strategy in strategies[index % 3:] + strategies[:index % 3]:
            directory = artifact_path(output, f"{task['id']}--{strategy}")
            if directory.exists():
                run = verified_receipt(directory, task, strategy, binding)
            else:
                remaining = protocol['max_total_estimated_usd'] - spent
                if remaining <= 0:
                    raise ValueError('experiment cost cap exhausted')
                # Only the question and source checkout are passed. No expected facts,
                # reference excerpts, grading instructions or previous answers are sent.
                try:
                    run_question(repository=roots[task['repository']], question=task['question'], output=directory,
                        strategy=strategy, model=protocol['model'],
                        context_budget=protocol['full_context_budget'] if strategy == 'full' else protocol['context_budget'],
                        candidate_budget=protocol['candidate_budget'], retention_budget=protocol['retention_budget'],
                        retention_policy=protocol['retention_policy'], max_output_tokens=protocol['max_output_tokens'],
                        max_cost=min(protocol['max_cost_per_question_usd'], remaining))
                except Exception:
                    if not (directory / 'run.json').exists():
                        raise
                write_json(directory / 'binding.json', {'experiment_sha256': binding, 'task_id': task['id'], 'strategy': strategy})
                checksums = json.loads((directory / 'checksums.json').read_text())
                checksums['binding.json'] = file_hash(directory / 'binding.json')
                write_json(directory / 'checksums.json', checksums)
                run = verified_receipt(directory, task, strategy, binding)
            if run['failed_request_billing_unknown']:
                raise ValueError('remote request billing unknown; stopped instead of claiming a bounded total')
            spent += run['estimated_usd']
            row = {'task_id': task['id'], 'repository': task['repository'], 'strategy': strategy,
                   'directory': directory.name, 'status': run['status'], 'estimated_usd': run['estimated_usd'],
                   'latency_ms': run['latency_ms'], 'calls': len(run['usage_ledger']),
                   'evidence_sha256': run['evidence_sha256'], 'run_sha256': file_hash(directory / 'run.json')}
            rows.append(row)
            write_json(output / 'results.json', {'experiment_sha256': binding, 'rows': rows, 'estimated_usd': spent})
            print(f"{len(rows):02d}/{len(data['tasks'])*3} {task['id']} {strategy}: {run['status']} ${run['estimated_usd']:.5f}; total ${spent:.5f}", flush=True)
    snapshots = {}
    for row in rows:
        evidence = load_evidence(output / row['directory'] / 'evidence.json')
        repository = row['repository']
        snapshot = evidence['repository']['snapshot_sha256']
        if repository in snapshots and snapshots[repository] != snapshot:
            raise ValueError('source snapshot changed between strategies')
        snapshots[repository] = snapshot
    return {'rows': rows, 'estimated_usd': spent, 'source_snapshots': snapshots}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'evaluations/repoqa_v1.json')
    parser.add_argument('--repositories', type=Path, required=True, help='Parent of the three pinned repository checkouts')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    evaluate(args.dataset, args.repositories, args.output, resume=args.resume)


if __name__ == '__main__':
    main()
