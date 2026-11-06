"""Package reconstructed dialogues and annotations in the dataset format."""
import argparse
import copy
import glob
import json
import tempfile
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_text_atomic(path, text):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix='.' + path.name + '-', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(text)
            handle.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def build_sample(record, procedure, annotation=None, split='train'):
    if record.get('status') != 'ok':
        raise ValueError('Cannot export failed reconstruction')
    cid = str(record['案件编号'])
    dialogue = copy.deepcopy(record['reconstructed_dialogue'])
    phases = []
    for p in dialogue.get('phase_dialogues', []):
        turns = [t['turn_id'] for t in dialogue['flat_turns'] if str(t['phase_id']) == str(p['phase_id'])]
        if turns:
            phases.append({'phase_id': p['phase_id'], 'type': p['phase_type'], 'turn_range': [min(turns), max(turns)]})
    labels = {'is_fraud': True, 'level_1_category': record['案件类别'], 'level_2_phases': phases}
    if annotation is not None:
        if annotation.get('status') != 'ok' or not isinstance(annotation.get('evidence_chain'), dict):
            raise ValueError('Cannot export failed evidence-chain annotation: ' + cid)
        chain = copy.deepcopy(annotation['evidence_chain'])
        signals = chain['step_3_signal_identification']['output']['signals']
        labels['level_3_signals'] = list(dict.fromkeys(s['signal_id'] for s in signals))
        labels['evidence_chain'] = chain
    return {'case_id': cid, 'split': split, 'data_type': 'fraud', 'is_fraud': True,
            'is_augmented': False, 'audio_path': '', 'reconstructed_dialogue': dialogue,
            'fraud_meta': {'structured_procedure': copy.deepcopy(procedure)}, 'labels': labels}


def export(input_path, procedure_pattern, output_dir, annotations=None, split='train', signal_labels=None):
    procedures = {}
    for filename in sorted(glob.glob(str(procedure_pattern))):
        data = json.loads(Path(filename).read_text())
        for row in data['results']:
            if row.get('status') == 'ok':
                procedures[(Path(filename).name, row['index'])] = row['result']
    annotated = {str(r['case_id']): r for r in read_jsonl(annotations)} if annotations else {}
    signals = {str(r['case_id']): r['labels'] for r in read_jsonl(signal_labels)} if signal_labels else {}
    # Reconstruction logs append retries; the last record is authoritative.
    records = {}
    for row in read_jsonl(input_path):
        records[(row['source_chunk'], row['index'])] = row
    samples = []
    for row in records.values():
        if annotations and str(row['案件编号']) not in annotated:
            raise ValueError('Missing annotation: ' + str(row['案件编号']))
        samples.append(build_sample(row, procedures[(row['source_chunk'], row['index'])],
                                    annotated.get(str(row['案件编号'])), split))
    for s in samples:
        if signal_labels:
            source_labels = signals[s['case_id']]
            if set(source_labels['level_3_signals']) != set(s['labels']['level_3_signals']):
                raise ValueError('Evidence chain disagrees with post-dialogue signals: ' + s['case_id'])
            s['labels']['signals_with_phase'] = source_labels.get('signals_with_phase', [])
    if len({s['case_id'] for s in samples}) != len(samples):
        raise ValueError('Duplicate case_id')
    for s in samples:
        if Path(s['case_id']).name != s['case_id'] or s['case_id'] in ('.', '..'):
            raise ValueError('Unsafe case_id')
    dest = Path(output_dir); (dest/'samples').mkdir(parents=True, exist_ok=True)
    for s in samples:
        write_text_atomic(dest/'samples'/f"{s['case_id']}.json", json.dumps(s, ensure_ascii=False, indent=2)+'\n')
    def write(name, rows):
        write_text_atomic(dest/name, ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    write('samples.jsonl', samples)
    write(f'{split}_index.jsonl', [{'case_id':s['case_id']} for s in samples])
    write(f'{split}_labels.jsonl', [{'case_id':s['case_id'], 'labels':s['labels']} for s in samples])
    return len(samples)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True); p.add_argument('--procedures', required=True)
    p.add_argument('--output-dir', required=True); p.add_argument('--annotations'); p.add_argument('--signal-labels')
    p.add_argument('--split', default='train', choices=['train','test'])
    a=p.parse_args();print('Exported samples:',export(a.input,a.procedures,a.output_dir,a.annotations,a.split,a.signal_labels))

if __name__=='__main__':main()
