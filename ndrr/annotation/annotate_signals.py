"""Annotate interaction signals in completed fraud dialogues."""
import argparse
import copy
import json
import os
from pathlib import Path

from .signal_prompts import SYSTEM_PROMPT, ALLOWED_SIGNALS


def apply_signals(sample, result):
    phases = sample['labels']['level_2_phases']
    rows = result['phases']
    expected = [str(p['phase_id']) for p in phases]
    if len(rows) != len(expected) or [str(p['phase_id']) for p in rows] != expected:
        raise ValueError('Signal annotation must cover each phase exactly once in order')
    signals=[]
    for row in rows:
        if not isinstance(row.get('tactics'),list) or any(x not in ALLOWED_SIGNALS for x in row['tactics']):
            raise ValueError('Invalid interaction signal identifier')
        signals.extend(row['tactics'])
    sample['labels']['level_3_signals']=list(dict.fromkeys(signals))
    sample['labels']['signals_with_phase']=[{'signal_id':sig,'phase':r['phase_id']} for r in rows for sig in r['tactics']]
    return sample


def annotation_input(sample):
    sample = copy.deepcopy(sample)
    for key in ('level_3_signals', 'signals_with_phase'):
        sample['labels'].pop(key, None)
    return sample


def annotate_samples(rows, output, client, model, resume=False):
    output = Path(output)
    ids = [sample['case_id'] for sample in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate input case_id')
    saved = {}
    if resume and output.exists():
        for line in output.read_text(encoding='utf-8').splitlines():
            if line.strip():
                sample = json.loads(line)
                if sample['case_id'] in saved:
                    raise ValueError('Duplicate saved signal case_id')
                saved[sample['case_id']] = sample
        if set(saved) - set(ids):
            raise ValueError('Saved signals contain cases absent from this input; use a new output')
        for sample in rows:
            cached = saved.get(sample['case_id'])
            if cached is not None and annotation_input(cached) != annotation_input(sample):
                raise ValueError('Signal input changed: ' + sample['case_id'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('a' if resume else 'w', encoding='utf-8') as handle:
        for sample in rows:
            if sample['case_id'] in saved:
                continue
            payload = {'phases': sample['labels']['level_2_phases'],
                       'D': sample['reconstructed_dialogue']['flat_turns']}
            response = client.chat.completions.create(model=model, temperature=0.2,
                messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                          {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
                response_format={'type': 'json_object'})
            annotated = apply_signals(copy.deepcopy(sample), json.loads(response.choices[0].message.content))
            handle.write(json.dumps(annotated, ensure_ascii=False) + '\n')
            handle.flush()
            saved[sample['case_id']] = annotated
    # Keep evidence-chain input indices stable even after a partially completed run.
    from ..export import write_text_atomic
    write_text_atomic(output, ''.join(json.dumps(saved[cid], ensure_ascii=False) + '\n' for cid in ids))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',default=os.getenv('OPENAI_MODEL',''))
    p.add_argument('--base-url',default=os.getenv('OPENAI_BASE_URL',''))
    p.add_argument('--resume', action='store_true', help='Reuse completed signal annotations by case_id')
    a=p.parse_args()
    if not a.model or not a.base_url or not os.getenv("OPENAI_API_KEY"):
        p.error("Explicit model, base URL and OPENAI_API_KEY are required")
    from openai import OpenAI
    client=OpenAI(api_key=os.environ['OPENAI_API_KEY'],base_url=a.base_url,timeout=120,max_retries=0)
    rows=[json.loads(x) for x in Path(a.input).read_text().splitlines() if x.strip()]
    try:
        annotate_samples(rows, a.output, client, a.model, resume=a.resume)
    finally:client.close()

if __name__=='__main__':main()
