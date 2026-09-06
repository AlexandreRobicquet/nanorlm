"""Run the same blind grader through Batch API without consuming daily chat quota."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from artifacts import artifact_path
from nanorlm import OpenAICompatibleBackend
from repoqa import digest
from scripts.batch_repoqa import Transport, collect_responses, submit
from scripts.evaluate_repoqa import file_hash, write_json
from scripts.grade_repoqa import GRADER_MODEL, GRADER_PROMPT, GRADING_CAP, grade_experiment


def advance(dataset: Path, experiment: Path, output: Path, send: bool) -> None:
    output = artifact_path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {'experiment_manifest_sha256': file_hash(experiment / 'experiment.json'),
                'experiment_results_sha256': file_hash(experiment / 'results.json'),
                'dataset_sha256': file_hash(dataset), 'model': GRADER_MODEL,
                'prompt_sha256': digest(GRADER_PROMPT), 'script_sha256': file_hash(Path(__file__)),
                'grader_sha256': file_hash(ROOT / 'scripts/grade_repoqa.py'),
                'transport_sha256': file_hash(ROOT / 'scripts/batch_repoqa.py'),
                'execution': 'batch', 'max_list_price_usd': GRADING_CAP}
    if (output / 'experiment.json').exists():
        if json.loads((output / 'experiment.json').read_text()) != protocol:
            raise ValueError('grading inputs or protocol changed')
    else:
        write_json(output / 'experiment.json', protocol)
    responses, waiting = collect_responses(output)
    if waiting:
        return
    if (output / 'grades').exists():
        print('Grading already complete; preserved existing receipts.')
        return
    pending = {}
    transport = Transport('grade', 'grade', responses, pending)
    def chat(backend, system, user):
        return transport.chat(backend, system, user)
    with tempfile.TemporaryDirectory(prefix='nanorlm-batch-grade-') as temporary:
        scratch = Path(temporary) / 'grades'
        with patch.object(OpenAICompatibleBackend, '_chat_text', chat):
            grade_experiment(dataset, experiment, scratch)
        if not transport.missing:
            for path in scratch.glob('*.json'):
                if path.name == 'protocol.json':
                    continue
                receipt = json.loads(path.read_text())
                receipt.pop('sha256')
                receipt['execution_mode'] = 'batch-replay'
                receipt['local_assembly_latency_ms'] = receipt.pop('latency_ms')
                receipt['latency_ms'] = None
                receipt['batch_estimated_usd'] = receipt['estimated_usd'] * .5
                for call in receipt['usage_ledger']:
                    call['local_replay_latency_ms'] = call.pop('latency_ms')
                    call['latency_ms'] = None
                write_json(path, {**receipt, 'sha256': digest(receipt)})
            shutil.copytree(scratch, output / 'grades')
            print('All grading receipts assembled from actual batch responses.')
    write_json(output / 'pending.json', list(pending.values()))
    if pending and send:
        submit(output, pending, responses, GRADER_MODEL, GRADING_CAP)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'evaluations/repoqa_v1.json')
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    advance(args.dataset, args.experiment, args.output, args.submit)
