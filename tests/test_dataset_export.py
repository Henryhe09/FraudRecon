import copy
import json
import tempfile
import unittest
from pathlib import Path
from ndrr.export import build_sample, export
from bench.data.dataset import load_cases

class ExportTests(unittest.TestCase):
    def fixture(self):
        return {'案件编号':'example_001','案件类别':'冒充电商物流客服类','source_chunk':'part.json','index':0,'status':'ok',
                'reconstructed_dialogue':{'phase_dialogues':[{'phase_id':1,'phase_type':'initial_contact'}],
                 'flat_turns':[{'turn_id':1,'phase_id':1,'phase_type':'initial_contact','speaker':'S','utterance':'您好'}]}}, {'fraud_phases':[]}

    def test_export_reads_back_through_benchmark_loader_and_preserves_turns(self):
        record,procedure=self.fixture();original=copy.deepcopy(record)
        annotation={'case_id':'example_001','status':'ok','evidence_chain':{'step_3_signal_identification':{'output':{'signals':[{'signal_id':'impersonate_ecommerce_cs'}]}}}}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'rec.jsonl').write_text(json.dumps(record)+'\n')
            (p/'part.json').write_text(json.dumps({'results':[{'status':'ok','index':0,'result':procedure}]}))
            (p/'ann.jsonl').write_text(json.dumps(annotation)+'\n')
            self.assertEqual(export(p/'rec.jsonl',str(p/'part.json'),p/'out',p/'ann.jsonl'),1)
            cases=load_cases(p/'out','train');self.assertEqual(len(cases),1)
            self.assertEqual(cases[0].turns,record['reconstructed_dialogue']['flat_turns'])
            self.assertEqual(cases[0].labels['level_3_signals'],['impersonate_ecommerce_cs'])
            self.assertEqual(cases[0].sample['fraud_meta']['structured_procedure'],procedure)
        self.assertEqual(record,original)

    def test_failed_annotation_is_not_a_successful_release(self):
        r,p=self.fixture()
        with self.assertRaises(ValueError):build_sample(r,p,{'status':'error'})

class CombinedPipelineTests(unittest.TestCase):
    def test_annotation_receives_completed_dialogue_then_final_export_includes_signals(self):
        from unittest.mock import patch
        import os
        from types import SimpleNamespace
        from ndrr.cli import main
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); record,procedure=ExportTests().fixture()
            def extraction(args,input_path,output_dir):
                output_dir.mkdir(parents=True)
                (output_dir/'part.json').write_text(json.dumps({'results':[{'status':'ok','index':0,'result':procedure}]}))
                record['source_chunk']='fraud_results_part_0000.json'
                (output_dir/'part.json').rename(output_dir/record['source_chunk'])
            def reconstruction(args,pattern,output,full):
                output.parent.mkdir(parents=True);output.write_text(json.dumps(record)+'\n');return 0
            def annotation(command,env):
                sample=json.loads(Path(command[command.index('--input')+1]).read_text())
                self.assertEqual(sample['reconstructed_dialogue'],record['reconstructed_dialogue'])
                if 'ndrr.annotation.annotate_signals' in command:
                    self.assertNotIn('level_3_signals',sample['labels'])
                    sample['labels']['level_3_signals']=['impersonate_ecommerce_cs']
                    Path(command[command.index('--output')+1]).write_text(json.dumps(sample)+'\n')
                    return SimpleNamespace(returncode=0)
                self.assertEqual(sample['labels']['level_3_signals'],['impersonate_ecommerce_cs'])
                path=Path(command[command.index('--output')+1])
                path.write_text(json.dumps({'case_id':'example_001','status':'ok','evidence_chain':{'step_3_signal_identification':{'output':{'signals':[{'signal_id':'impersonate_ecommerce_cs'}]}}}})+'\n')
                return SimpleNamespace(returncode=0)
            with patch.dict(os.environ, {'OPENAI_API_KEY':'offline-key','OPENAI_BASE_URL':'https://offline.invalid/v1','OPENAI_MODEL':'offline-model','OPENAI_ANNOTATION_MODEL':'offline-annotation'}),patch('ndrr.cli.run_extraction',extraction),patch('ndrr.cli.run_reconstruction',reconstruction),patch('ndrr.cli.subprocess.run',annotation):
                self.assertEqual(main(['run','--input','fictional.json','--output-dir',str(root/'out')]),0)
            final=json.loads((root/'out/dataset/samples/example_001.json').read_text())
            self.assertEqual(final['labels']['level_3_signals'],['impersonate_ecommerce_cs'])


class SignalTests(unittest.TestCase):
    def test_source_taxonomy_and_empty_phase_are_supported(self):
        from ndrr.annotation.annotate_signals import ALLOWED_SIGNALS,apply_signals
        self.assertEqual(len(ALLOWED_SIGNALS),26)
        sample={'labels':{'level_2_phases':[{'phase_id':1}]}}
        self.assertEqual(apply_signals(sample,{'phases':[{'phase_id':1,'tactics':[]} ]})['labels']['level_3_signals'],[])

    def test_unknown_signal_and_missing_phase_rejected(self):
        from ndrr.annotation.annotate_signals import apply_signals
        for result in [{'phases':[]},{'phases':[{'phase_id':1,'tactics':['invented_signal']}]}]:
            with self.assertRaises(ValueError):apply_signals({'labels':{'level_2_phases':[{'phase_id':1}]}},result)
