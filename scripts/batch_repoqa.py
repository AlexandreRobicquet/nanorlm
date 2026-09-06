"""Batch transport for the frozen QA comparison when synchronous quota is exhausted.

Only transport/scheduling changes. The real engine constructs every prompt and
validates every answer; placeholder planning outputs are never published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import uuid
from collections import Counter
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanorlm import OpenAICompatibleBackend, Usage
from artifacts import artifact_path
from repoqa import digest, git_value, run_question
from scripts.evaluate_repoqa import file_hash, validate_dataset, verified_receipt, write_json

PLACEHOLDER = json.dumps({'summary': 'Planning placeholder', 'evidence': [],
                         'answer_candidate': '', 'confidence': 0.0})


def api(path: str, body: dict | None = None, *, raw: bytes | None = None,
        content_type: str = 'application/json') -> dict | bytes:
    data = json.dumps(body).encode() if body is not None else raw
    request = urllib.request.Request('https://api.openai.com/v1' + path, data=data,
        headers={'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'], 'Content-Type': content_type})
    with urllib.request.urlopen(request, timeout=120) as response:
        result = response.read()
        return result if path.endswith('/content') else json.loads(result)


class Transport:
    def __init__(self, case: str, strategy: str, responses: dict, pending: dict):
        self.case, self.strategy = case, strategy
        self.responses, self.pending = responses, pending
        self.occurrences: Counter = Counter()
        self.missing = False

    def chat(self, backend, system: str, user: str) -> dict:
        body = {'model': backend.config.model,
                'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                'temperature': 0.0, 'max_completion_tokens': backend.config.max_output_tokens}
        identity = digest(body)
        self.occurrences[identity] += 1
        custom_id = f'{self.case}.{identity[:24]}.{self.occurrences[identity]}'
        if custom_id in self.responses:
            stored = self.responses[custom_id]
            if stored['request_sha256'] != identity:
                raise ValueError('batch response/request mismatch')
            response = stored['response']
            if response.get('error') or not response.get('response') or response['response']['status_code'] != 200:
                raise RuntimeError('batch request failed; original response is retained')
            payload = response['response']['body']
            usage = payload['usage']
            backend._record_response_model_identifier(payload['model'])
            return {'content': payload['choices'][0]['message']['content'],
                    'usage': Usage(usage['prompt_tokens'], usage['completion_tokens'], 1)}
        # Inspection branches depend on source sizes, never on worker summaries.
        # Continue planning other independent leaves with a valid placeholder.
        # A retention final prompt is deferred until ALL inspections/repairs exist.
        if not (self.strategy == 'retention' and backend.stage == 'answer' and self.missing):
            self.pending[custom_id] = {'custom_id': custom_id, 'method': 'POST',
                                      'url': '/v1/chat/completions', 'body': body}
        self.missing = True
        return {'content': PLACEHOLDER if backend.stage == 'inspect' else '{"claims":[],"uncertainties":[]}',
                'usage': Usage()}


def collect_responses(output: Path) -> tuple[dict, bool]:
    responses = {}
    waiting = False
    for directory in sorted(output.glob('round-*')):
        directory = artifact_path(output, directory.name)
        if not (directory / 'batch.json').exists():
            raise ValueError(f'{directory.name}: interrupted submission requires investigation before retry')
        batch = json.loads((directory / 'batch.json').read_text())
        submission = json.loads((directory / 'submission.json').read_text())
        if file_hash(directory / 'input.jsonl') != submission['input_sha256']:
            raise ValueError('submitted batch input changed')
        if batch['status'] not in {'completed', 'failed', 'expired', 'cancelled'}:
            batch = api('/batches/' + batch['id'])
            write_json(directory / 'batch.json', batch)
        print(directory.name, batch['status'], batch.get('request_counts'), flush=True)
        if batch['status'] not in {'completed', 'failed', 'expired', 'cancelled'}:
            waiting = True
            continue
        if batch['status'] != 'completed':
            raise ValueError(f"batch {batch['id']} {batch['status']}: {batch.get('errors')}; no silent rerun")
        if not (directory / 'output.jsonl').exists():
            (directory / 'output.jsonl').write_bytes(api('/files/' + batch['output_file_id'] + '/content'))
        if batch.get('error_file_id') and not (directory / 'errors.jsonl').exists():
            (directory / 'errors.jsonl').write_bytes(api('/files/' + batch['error_file_id'] + '/content'))
        requests = [json.loads(line) for line in (directory / 'input.jsonl').read_text().splitlines()]
        expected = {row['custom_id']: digest(row['body']) for row in requests}
        returned = set()
        for name in ('output.jsonl', 'errors.jsonl'):
            path = directory / name
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                row = json.loads(line)
                cid = row['custom_id']
                if cid not in expected or cid in returned or cid in responses:
                    raise ValueError('batch response identity mismatch or duplicate')
                returned.add(cid)
                responses[cid] = {'request_sha256': expected[cid], 'response': row, 'batch_id': batch['id']}
        if returned != set(expected):
            raise ValueError('batch response inventory is incomplete')
    return responses, waiting


def submit(output: Path, requests: dict, responses: dict, model: str, cap: float) -> None:
    # Reservation uses a byte upper bound and the full output limit at normal
    # list prices. Actual Batch API prices are lower; discounts are separate.
    from nanorlm import REMOTE_MODEL_PRICES
    prices = REMOTE_MODEL_PRICES['openai_compatible', model.removesuffix('-2025-04-14')]
    spent = 0.0
    for stored in responses.values():
        usage = (stored['response'].get('response') or {}).get('body', {}).get('usage', {})
        spent += usage.get('prompt_tokens', 0) * prices[0] + usage.get('completion_tokens', 0) * prices[1]
    reservation = sum((sum(len(message['content'].encode()) for message in row['body']['messages']) + 256) * prices[0]
                      + row['body']['max_completion_tokens'] * prices[1] for row in requests.values())
    if spent + reservation > cap:
        raise ValueError(f'batch reservation exceeds experiment cap: {spent + reservation:.4f} > {cap}')
    directory = output / f'round-{len(list(output.glob("round-*"))) + 1:02d}'
    directory.mkdir()
    raw = ''.join(json.dumps(row) + '\n' for row in requests.values()).encode()
    (directory / 'input.jsonl').write_bytes(raw)
    write_json(directory / 'submission.json', {'input_sha256': hashlib.sha256(raw).hexdigest(),
        'requests': len(requests), 'prior_list_price_usd': spent, 'reserved_list_price_usd': reservation,
        'status': 'submitting', 'created_at': time.time()})
    boundary = 'nanorlm' + uuid.uuid4().hex
    multipart = (f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="input.jsonl"\r\n'
        'Content-Type: application/jsonl\r\n\r\n').encode() + raw + f'\r\n--{boundary}--\r\n'.encode()
    uploaded = api('/files', raw=multipart, content_type=f'multipart/form-data; boundary={boundary}')
    write_json(directory / 'upload.json', uploaded)
    batch = api('/batches', {'input_file_id': uploaded['id'], 'endpoint': '/v1/chat/completions',
        'completion_window': '24h', 'metadata': {'experiment': 'nanorlm-repoqa-v1-batch', 'round': directory.name}})
    write_json(directory / 'batch.json', batch)
    print(f"Submitted {len(requests)} requests as {batch['id']}; list-price reservation ${reservation:.4f}", flush=True)


def advance(dataset: Path, repositories: Path, output: Path, send: bool) -> None:
    output = artifact_path(output)
    data = json.loads(dataset.read_text())
    roots = validate_dataset(data, repositories)
    if git_value(ROOT, 'status', '--porcelain') != '':
        raise ValueError('batch runner checkout must be clean')
    protocol = {**data['protocol'], 'execution': 'batch',
        'ordering': 'independent inspection leaves and baseline answers batched; dependent repairs and retention answers in later rounds',
        'latency': 'batch turnaround and local replay assembly only; interactive latency unavailable',
        'amendment_reason': 'Synchronous account limit 50 requests/day per model exhausted after four completed cases; no prompt or question changes.'}
    manifest = {'schema': 'nanorlm-repoqa-experiment-v1', 'dataset_sha256': file_hash(dataset),
        'implementation_commit': git_value(ROOT, 'rev-parse', 'HEAD'),
        'implementation_sha256': {name: file_hash(ROOT / name) for name in ('repoqa.py','nanorlm.py','policies.py','learned_retention.py')},
        'runner_sha256': file_hash(Path(__file__)), 'protocol': protocol, 'repositories': data['repositories']}
    manifest['experiment_sha256'] = digest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'experiment.json').exists():
        if json.loads((output / 'experiment.json').read_text()) != manifest:
            raise ValueError('batch experiment code or protocol changed')
    else:
        write_json(output / 'experiment.json', manifest)
    responses, waiting = collect_responses(output)
    if waiting:
        return
    pending, rows = {}, []
    for task in data['tasks']:
        for strategy in protocol['strategies']:
            name = f"{task['id']}--{strategy}"
            destination = output / name
            if destination.exists():
                run = verified_receipt(destination, task, strategy, manifest['experiment_sha256'])
            else:
                transport = Transport(name, strategy, responses, pending)
                with tempfile.TemporaryDirectory(prefix='nanorlm-batch-plan-') as temporary:
                    case = Path(temporary) / 'case'
                    def chat(backend, system, user):
                        return transport.chat(backend, system, user)
                    with patch.object(OpenAICompatibleBackend, '_chat_text', chat):
                        try:
                            run_question(repository=roots[task['repository']], question=task['question'], output=case,
                                strategy=strategy, model=protocol['model'],
                                context_budget=protocol['full_context_budget'] if strategy == 'full' else protocol['context_budget'],
                                candidate_budget=protocol['candidate_budget'], retention_budget=protocol['retention_budget'],
                                retention_policy=protocol['retention_policy'], max_output_tokens=protocol['max_output_tokens'],
                                max_cost=protocol['max_cost_per_question_usd'])
                        except Exception:
                            if not (case / 'run.json').exists():
                                raise
                    if transport.missing:
                        continue
                    run = json.loads((case / 'run.json').read_text())
                    run.update({'execution_mode': 'batch-replay', 'interactive_latency_ms': None,
                        'local_assembly_latency_ms': run.pop('latency_ms'), 'latency_ms': None,
                        'batch_estimated_usd': run['estimated_usd'] * .5})
                    for call in run['usage_ledger']:
                        call['local_replay_latency_ms'] = call.pop('latency_ms')
                        call['latency_ms'] = None
                    write_json(case / 'run.json', run)
                    write_json(case / 'binding.json', {'experiment_sha256': manifest['experiment_sha256'],
                        'task_id': task['id'], 'strategy': strategy})
                    # The original Markdown's replay timing must not look like API latency.
                    answer_md = case / 'answer.md'
                    if answer_md.exists():
                        answer_md.write_text('Batch execution: interactive latency is unavailable. Costs below use normal list prices; Batch API estimate is half.\n\n' + answer_md.read_text())
                    write_json(case / 'checksums.json', {p.name: file_hash(p) for p in case.iterdir() if p.name != 'checksums.json'})
                    shutil.copytree(case, destination)
            rows.append({'task_id': task['id'], 'repository': task['repository'], 'strategy': strategy,
                'directory': name, 'status': run['status'], 'estimated_usd': run['estimated_usd'],
                'batch_estimated_usd': run['batch_estimated_usd'], 'latency_ms': None,
                'calls': len(run['usage_ledger']), 'evidence_sha256': run['evidence_sha256'],
                'run_sha256': file_hash(destination / 'run.json')})
    write_json(output / 'results.json', {'experiment_sha256': manifest['experiment_sha256'], 'rows': rows,
        'estimated_usd': sum(row['estimated_usd'] for row in rows),
        'batch_estimated_usd': sum(row['batch_estimated_usd'] for row in rows)})
    write_json(output / 'pending.json', list(pending.values()))
    print(f"{len(rows)}/90 cases assembled; {len(pending)} requests pending", flush=True)
    if pending and send:
        submit(output, pending, responses, protocol['model'], protocol['max_total_estimated_usd'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'evaluations/repoqa_v1.json')
    parser.add_argument('--repositories', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--submit', action='store_true', help='Submit the next pending round after assembling available results')
    args = parser.parse_args()
    advance(args.dataset, args.repositories, args.output, args.submit)
