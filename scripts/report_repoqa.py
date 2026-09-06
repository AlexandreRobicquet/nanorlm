"""Aggregate fact/citation judgments without mixing quality, latency and cost."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repoqa import digest, load_evidence, validate_answer
from scripts.evaluate_repoqa import file_hash, verified_receipt, write_json
from scripts.grade_repoqa import validate_grade


def adjudicate_grade(original: dict, audit: dict, checksum: str, facts: int, claims: int) -> dict:
    grade = json.loads(json.dumps(original))
    if audit:
        if audit.get('receipt_sha256') != checksum:
            raise ValueError('audit is not bound to the original grading receipt')
        if 'replacement_grade' in audit:
            if not isinstance(audit.get('reason'), str) or not audit['reason'].strip():
                raise ValueError('replacement adjudication requires an explanation')
            grade = audit['replacement_grade']
    if grade.get('requires_adjudication') or 'error' in grade:
        raise ValueError('malformed grader response requires a complete replacement_grade in the audit')
    validate_grade(grade, facts, claims)
    for change in audit.get('changes', []):
        kind = change['kind']
        if kind not in {'facts','claims'} or not change.get('reason') or not 1 <= change['index'] <= len(grade[kind]):
            raise ValueError('audit must name a valid fact/claim and explain the change')
        grade[kind][change['index']-1].update(change['values'])
    validate_grade(grade, facts, claims)
    return grade


def select_strategy(summaries: dict) -> tuple[list[str], str | None]:
    best = max(summaries, key=lambda key: (summaries[key]['fully_correct'], -summaries[key]['estimated_usd']))
    best_precision = summaries[best]['citation_precision'] or 0
    eligible = [strategy for strategy, summary in summaries.items()
                if summary['fully_correct'] >= summaries[best]['fully_correct']-1
                and (summary['citation_precision'] or 0) >= best_precision-.05
                and (strategy != 'retention' or summary['fully_correct'] >= summaries['lexical']['fully_correct']+3)]
    selected = min(eligible, key=lambda key: summaries[key]['estimated_usd']) if eligible else None
    return eligible, selected


def report(dataset: Path, experiment: Path, grades: Path, output: Path, audit: Path | None) -> None:
    data = json.loads(dataset.read_text())
    manifest = json.loads((experiment / 'experiment.json').read_text())
    results = json.loads((experiment / 'results.json').read_text())
    if file_hash(dataset) != manifest['dataset_sha256'] or results['experiment_sha256'] != manifest['experiment_sha256']:
        raise ValueError('dataset/experiment binding mismatch')
    expected = {(task['id'], strategy) for task in data['tasks'] for strategy in data['protocol']['strategies']}
    if len(results['rows']) != len(expected) or {(row['task_id'],row['strategy']) for row in results['rows']} != expected:
        raise ValueError('expected exactly one result for every task and strategy')
    tasks = {task['id']: task for task in data['tasks']}
    audits = json.loads(audit.read_text()) if audit else {'cases': {}, 'method': 'No secondary audit recorded.'}
    scored, snapshots = [], {}
    for row in results['rows']:
        task = tasks[row['task_id']]
        directory = experiment / row['directory']
        run = verified_receipt(directory, task, row['strategy'], manifest['experiment_sha256'])
        if file_hash(directory / 'run.json') != row['run_sha256']:
            raise ValueError('result/receipt binding mismatch')
        evidence = load_evidence(directory / 'evidence.json')
        snapshot = evidence['repository']['snapshot_sha256']
        if row['repository'] in snapshots and snapshots[row['repository']] != snapshot:
            raise ValueError('repository snapshot differs between cases')
        snapshots[row['repository']] = snapshot
        if evidence['repository']['commit'] != data['repositories'][row['repository']]['commit']:
            raise ValueError('wrong source commit')
        answer = json.loads((directory / 'answer.json').read_text())
        validate_answer(answer, evidence['spans'])
        path = grades / (row['directory'] + '.json')
        receipt = json.loads(path.read_text())
        checksum = receipt.pop('sha256')
        if digest(receipt) != checksum or receipt['answer_sha256'] != file_hash(directory / 'answer.json'):
            raise ValueError('grader receipt/answer binding mismatch')
        grade = adjudicate_grade(receipt['grade'], audits.get('cases', {}).get(row['directory'], {}),
                                 checksum, len(task['expected_facts']), len(answer['claims']))
        facts = sum(item['correct'] for item in grade['facts'])
        incorrect = sum(item['materially_incorrect'] for item in grade['claims'])
        supported = sum(item['supported'] for item in grade['claims'])
        scored.append({**row, 'facts_correct': facts, 'facts_total': len(task['expected_facts']),
            'fully_correct': run['status'] == 'answered' and facts == len(task['expected_facts']) and incorrect == 0,
            'claims_supported': supported, 'claims_total': len(grade['claims']), 'incorrect_claims': incorrect,
            'source_citation_integrity': True, 'selected_spans': evidence['coverage']['selected_spans'],
            'scanned_spans': evidence['coverage']['scanned_spans'], 'grading_estimated_usd': receipt['estimated_usd'],
            'reference_concern': grade['reference_concern']})
    summaries = {}
    for strategy in data['protocol']['strategies']:
        rows = [row for row in scored if row['strategy'] == strategy]
        times = sorted(row['latency_ms'] for row in rows if row['latency_ms'] is not None)
        claims = sum(row['claims_total'] for row in rows)
        support = sum(row['claims_supported'] for row in rows)
        summaries[strategy] = {'questions': len(rows), 'answered': sum(row['status']=='answered' for row in rows),
            'fully_correct': sum(row['fully_correct'] for row in rows),
            'facts_correct': sum(row['facts_correct'] for row in rows), 'facts_total': sum(row['facts_total'] for row in rows),
            'claims_supported': support, 'claims_total': claims, 'citation_precision': support/claims if claims else None,
            'estimated_usd': sum(row['estimated_usd'] for row in rows),
            'batch_estimated_usd': sum(row.get('batch_estimated_usd',0) for row in rows) if not times else None,
            'median_latency_ms': statistics.median(times) if times else None,
            'p95_latency_ms': times[min(len(times)-1, int(len(times)*.95))] if times else None,
            'calls': sum(row['calls'] for row in rows),
            'by_repository': {repo: {'fully_correct': sum(row['fully_correct'] for row in rows if row['repository']==repo),
                                   'questions': sum(row['repository']==repo for row in rows)} for repo in data['repositories']}}
    eligible, selected = select_strategy(summaries)
    output.mkdir(parents=True, exist_ok=True)
    report_data = {'dataset_sha256': file_hash(dataset), 'experiment_sha256': manifest['experiment_sha256'],
        'report_script_sha256': file_hash(Path(__file__)),
        'grading_protocol_sha256': file_hash(grades / 'protocol.json'), 'audit_sha256': file_hash(audit) if audit else None,
        'audit_method': audits['method'], 'source_snapshots': snapshots, 'summaries': summaries,
        'eligible_under_frozen_rule': eligible, 'selected_under_frozen_rule': selected, 'cases': scored,
        'grading_estimated_usd': sum(row['grading_estimated_usd'] for row in scored)}
    write_json(output / 'summary.json', report_data)
    lines = ['# Repository QA evaluation results', '',
        '| Strategy | Fully correct | Facts | Supported claims | Normal-price estimate | Batch estimate |',
        '|---|---:|---:|---:|---:|---:|']
    for strategy, summary in summaries.items():
        lines.append(f"| {strategy} | {summary['fully_correct']}/30 | {summary['facts_correct']}/90 | "
            f"{summary['claims_supported']}/{summary['claims_total']} | ${summary['estimated_usd']:.4f} | "
            + (f"${summary['batch_estimated_usd']:.4f} |" if summary['batch_estimated_usd'] is not None else 'N/A |'))
    lines += ['', f'Frozen selection rule: **{selected or "no eligible strategy"}**.', '',
              'Citation precision counts produced claims; failures still score zero factual completeness. '
              'All source citation identities and span hashes passed integrity checks.', '',
              'Interactive latency is unavailable for batch runs. Local replay times are not API latency.', '',
              audits['method'], '',
              'This small assistant-authored evaluation covers three Python repositories and uses model-assisted grading. '
              'Public sources may have appeared in pretraining; results do not establish general superiority.', '']
    (output / 'summary.md').write_text('\n'.join(lines))
    print(json.dumps({'selection': selected, 'summaries': summaries}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'evaluations/repoqa_v1.json')
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--grades', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--audit', type=Path)
    args = parser.parse_args()
    report(args.dataset, args.experiment, args.grades, args.output, args.audit)
