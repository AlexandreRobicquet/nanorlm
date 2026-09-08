"""Grade independent answer pairs with constrained synchronous output and pacing."""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nanorlm.artifacts import artifact_path
from nanorlm import REMOTE_MODEL_PRICES, extract_json_object
from nanorlm.repoqa import digest, load_evidence
from scripts.evaluate_repoqa import file_hash, verified_receipt, write_json
from scripts.grade_repoqa import GRADER_PROMPT, grading_packet, normalize_grade

MODEL = 'gpt-5.4-mini-2026-03-17'
MAX_OUTPUT = 12000
CAP = 6.0
PROMPT = GRADER_PROMPT.replace('Return JSON only with three keys: facts, claims, reference_concern.',
    'For each supplied case, construct a grade with three keys: facts, claims, reference_concern.') + '''
Input contains independent cases keyed by opaque IDs. Evaluate each independently; do not compare answers or transfer evidence between cases. Return one JSON object with exactly those IDs as keys, each mapping to its complete grade. The response schema enforces all counts. Every claim contains its own cited_excerpts. Distinguish wrong behavior from missing details. Read all conditions, negations and branch boundaries before deciding.'''


def grade_schema(packet: dict) -> dict:
    def obj(properties):
        return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}
    def array(properties,count):
        return {'type':'array','items':obj(properties),'minItems':count,'maxItems':count}
    return obj({
        'facts':array({'index':{'type':'integer'},'covered_by_answer':{'type':'boolean'},
            'contradicted_by_answer':{'type':'boolean'},'claim_indices':{'type':'array','items':{'type':'integer'}},
            'reason':{'type':'string'}},packet['fact_count']),
        'claims':array({'index':{'type':'integer'},'citation_verdict':{'type':'string','enum':['supports','contradicts','insufficient']},
            'materially_incorrect':{'type':'boolean'},'reason':{'type':'string'}},packet['claim_count']),
        'reference_concern':{'type':'string'}})


def request_body(packets: dict) -> dict:
    schema={'type':'object','properties':{key:grade_schema(packet) for key,packet in packets.items()},
            'required':list(packets),'additionalProperties':False}
    return {'model':MODEL,'messages':[{'role':'system','content':PROMPT},
            {'role':'user','content':json.dumps(packets,ensure_ascii=True)}],
            'reasoning_effort':'medium','max_completion_tokens':MAX_OUTPUT,
            'response_format':{'type':'json_schema','json_schema':{'name':'repository_grades','strict':True,'schema':schema}}}


def send(body: dict) -> dict:
    started=time.perf_counter()
    request=urllib.request.Request('https://api.openai.com/v1/chat/completions',data=json.dumps(body).encode(),
        headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY'],'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=120) as response:
            return {'response':json.loads(response.read()),'rate_headers':{k.lower():v for k,v in response.headers.items()
                if k.lower().startswith('x-ratelimit')},'latency_ms':(time.perf_counter()-started)*1000}
    except urllib.error.HTTPError as exc:
        # Preserve status and request binding without publishing account identifiers.
        return {'error':{'http_status':exc.code,'billing':'unknown'},'latency_ms':(time.perf_counter()-started)*1000}


def group_cases(cases: list[dict]) -> list[list[dict]]:
    remaining=list(cases)
    random.Random(0).shuffle(remaining)
    groups=[]
    while remaining:
        group=[remaining.pop(0)]
        for index,case in enumerate(remaining):
            if case['task_id'] != group[0]['task_id']:
                group.append(remaining.pop(index));break
        groups.append(group)
    return groups


def grade(dataset: Path, experiment: Path, output: Path, execute: bool) -> None:
    import tiktoken
    data=json.loads(dataset.read_text()); manifest=json.loads((experiment/'experiment.json').read_text())
    results=json.loads((experiment/'results.json').read_text())
    expected={(task['id'],s) for task in data['tasks'] for s in data['protocol']['strategies']}
    if file_hash(dataset)!=manifest['dataset_sha256'] or results['experiment_sha256']!=manifest['experiment_sha256']:
        raise ValueError('dataset/experiment binding mismatch')
    if len(results['rows'])!=len(expected) or {(r['task_id'],r['strategy']) for r in results['rows']}!=expected:
        raise ValueError('grade only a complete experiment')
    tasks={task['id']:task for task in data['tasks']};cases=[]
    for index,row in enumerate(results['rows']):
        task=tasks[row['task_id']];directory=artifact_path(experiment,row['directory'])
        run=verified_receipt(directory,task,row['strategy'],manifest['experiment_sha256'])
        if file_hash(directory/'run.json')!=row['run_sha256']:raise ValueError('run receipt changed')
        answer=json.loads((directory/'answer.json').read_text())
        packet=grading_packet(task,answer,load_evidence(directory/'evidence.json'))
        cases.append({'id':f'c{index:03d}','case':row['directory'],'task_id':row['task_id'],'strategy':row['strategy'],
            'packet':packet,'answer_sha256':file_hash(directory/'answer.json'),'status':run['status']})
    groups=group_cases([case for case in cases if case['status']=='answered'])
    spec={'version':4,'experiment_sha256':manifest['experiment_sha256'],'dataset_sha256':file_hash(dataset),
        'model':MODEL,'reasoning_effort':'medium','max_output_tokens':MAX_OUTPUT,'max_estimated_usd':CAP,
        'script_sha256':file_hash(Path(__file__)),'packet_script_sha256':file_hash(ROOT/'scripts/grade_repoqa.py'),
        'prompt':PROMPT,'groups':[[c['id'] for c in group] for group in groups],
        'grouping':'Seed 0 shuffle; at most two cases with different questions; no strategy/cost metadata shown.',
        'cost_allocation':'Each case receives an equal share of its group grading cost. Group receipts hold actual unsplit usage.',
        'calibration_limitation':'Paired calibration passed coverage and citation-support checks but missed one conditional factual-error label. Raw calibration is preserved; model scores alone cannot pass release.',
        'mandatory_audit':'Review all fact judgments, every unsupported or materially incorrect claim, every claim in a provisionally fully-correct answer, and predefined source-audit seed cases.',
        'method':'Constrained model grading plus separate assistant source audit; not human adjudication.'}
    output=artifact_path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_text())!=spec:
        raise ValueError('grading protocol changed; use a fresh directory')
    write_json(output/'protocol.json',spec)
    (output/'groups').mkdir(exist_ok=True);(output/'grades').mkdir(exist_ok=True)
    write_json(output/'grades/protocol.json',spec)
    prices=REMOTE_MODEL_PRICES['openai_compatible',MODEL]
    encoder=tiktoken.encoding_for_model(MODEL)
    planned=[]
    for group in groups:
        body=request_body({case['id']:case['packet'] for case in group})
        # Include the structured-output schema in both reservation and token pacing.
        input_text=json.dumps(body['messages'],ensure_ascii=True)+json.dumps(body['response_format'])
        reserve=(len(input_text.encode())+512)*prices[0]+MAX_OUTPUT*prices[1]
        tokens=len(encoder.encode(input_text,disallowed_special=()))+MAX_OUTPUT+512
        if tokens>100_000:raise ValueError('group exceeds synchronous token allowance')
        planned.append((group,body,reserve,tokens))
    write_json(output/'plan.json',{'groups':len(groups),'cases':len(cases),'model_cases':sum(len(g) for g in groups),
        'maximum_request_token_reservation':max(p[3] for p in planned),
        'maximum_total_cost_reservation':sum(p[2] for p in planned)})
    if sum(p[2] for p in planned)>CAP:raise ValueError('complete grading reservation exceeds cap')
    if not execute:
        print(json.dumps(json.loads((output/'plan.json').read_text()),indent=2));return
    spent=0.0;available=100_000.0;limit=100_000.0
    def save_case(case,judgment,raw,cost,group_hash=None,model=None):
        record={'case':case['case'],'task_id':case['task_id'],'strategy':case['strategy'],
            'packet_sha256':digest(case['packet']),'answer_sha256':case['answer_sha256'],'grade':judgment,
            'raw_response':raw,'estimated_usd':cost,'usage_ledger':[],
            'shared_grading_group_sha256':group_hash,'response_models':[model] if model else [],
            'latency_ms':None,'execution_mode':'shared-synchronous-grading'}
        write_json(output/'grades'/(case['case']+'.json'),{**record,'sha256':digest(record)})
    for case in cases:
        if case['status']!='answered':
            save_case(case,{'facts':[{'index':i+1,'correct':False,'reason':'No usable answer.'}
                for i in range(case['packet']['fact_count'])],'claims':[],'reference_concern':''},'',0)
    for number,(group,body,reserve,tokens) in enumerate(planned,1):
        path=output/'groups'/f'{number:02d}.json';identity=digest(body)
        if path.exists():
            record=json.loads(path.read_text());seal=record.pop('sha256')
            if digest(record)!=seal or record['request_sha256']!=identity:raise ValueError('group receipt changed')
        else:
            if spent+reserve>CAP:raise ValueError('grading cost cap exceeded')
            delay=max(0,(tokens-available)/limit*60)+1
            if delay>0:time.sleep(delay)
            # A persisted pending request prevents unknown outcomes from being silently resubmitted.
            pending=path.with_suffix('.pending.json')
            if pending.exists():raise ValueError('interrupted grading request requires reconciliation')
            write_json(pending,{'request_sha256':identity,'request':body,'reservation_usd':reserve})
            response=send(body)
            record={'request_sha256':identity,'request':body,**response}
            write_json(path,{**record,'sha256':digest(record)})
            pending.unlink()
        if 'error' in record:raise ValueError('grading request failed; preserved receipt requires reconciliation')
        response=record['response'];usage=response['usage']
        if response['model']!=MODEL:raise ValueError('grader returned a different model')
        cost=usage['prompt_tokens']*prices[0]+usage['completion_tokens']*prices[1];spent+=cost
        headers=record['rate_headers'];limit=float(headers.get('x-ratelimit-limit-tokens',100000))
        available=float(headers.get('x-ratelimit-remaining-tokens',0))
        raw=response['choices'][0]['message']['content'] or ''
        try:
            parsed=extract_json_object(raw)
            if set(parsed)!={case['id'] for case in group}:raise ValueError('group case identity mismatch')
        except ValueError as exc:
            parsed={case['id']:{'error':str(exc),'requires_adjudication':True} for case in group}
        for case in group:
            try:judgment=normalize_grade(parsed[case['id']],case['packet']['fact_count'],case['packet']['claim_count'])
            except ValueError as exc:judgment={'error':str(exc),'requires_adjudication':True}
            save_case(case,judgment,raw,cost/len(group),digest(record),response['model'])
        print(f'Group {number}/{len(groups)} complete; grading ${spent:.6f}; requests remaining {headers.get("x-ratelimit-remaining-requests","unknown")}',flush=True)
    write_json(output/'completion.json',{'cases':len(cases),'groups':len(groups),'estimated_usd':spent,
        'protocol_sha256':file_hash(output/'protocol.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'evaluations/repoqa_v1.json')
    p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');args=p.parse_args()
    grade(args.dataset,args.experiment,args.output,args.execute)
