"""Input layout and task naming checks; no model calls."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from bench.data.dataset import load_cases, build_phase_prefixes
from bench.core.evaluator import EvaluationEngine
from bench.modes import EVAL_MODE_ALIASES

ROOT=Path(__file__).resolve().parents[1]

class BenchmarkInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)/'dataset'
        shutil.copytree(ROOT/'bench/examples',self.root)
    def tearDown(self):self.tmp.cleanup()
    def test_external_labels_are_authoritative_and_sample_metadata_is_preserved(self):
        p=self.root/'samples/demo_fraud_0001.json';s=json.loads(p.read_text());s['labels']={'is_fraud':False};p.write_text(json.dumps(s))
        case=load_cases(self.root)[0]
        self.assertTrue(case.is_fraud)
        self.assertEqual(case.sample['metadata'],s['metadata'])
        self.assertEqual(case.turns,s['reconstructed_dialogue']['flat_turns'])
    def test_split_subdirectory_uses_the_same_contract(self):
        (self.root/'test').mkdir()
        for name in ['test_index.jsonl','test_labels.jsonl','samples']:(self.root/name).rename(self.root/'test'/name)
        self.assertEqual(len(load_cases(self.root)),2)
    def test_requested_test_does_not_fall_back_to_train(self):
        (self.root/'test_index.jsonl').rename(self.root/'train_index.jsonl')
        with self.assertRaises(FileNotFoundError):load_cases(self.root,'test')
    def test_missing_external_labels_are_reported(self):
        (self.root/'test_labels.jsonl').unlink()
        with self.assertRaises(FileNotFoundError):load_cases(self.root)
    def test_duplicate_index_case_is_reported(self):
        p=self.root/'test_index.jsonl';p.write_text(p.read_text()+p.read_text().splitlines()[0]+'\n')
        with self.assertRaisesRegex(ValueError,'Duplicate index'):load_cases(self.root)
    def test_phase_prefixes_keep_original_context_and_endpoint(self):
        case=load_cases(self.root)[0];prefixes=build_phase_prefixes(case)
        self.assertEqual(prefixes[1]['end_turn'],4)
        self.assertIn('[Turn 1]',prefixes[1]['dialogue_prefix'])
        self.assertIn('[Turn 4]',prefixes[1]['dialogue_prefix'])
        self.assertNotIn('[Turn 5]',prefixes[1]['dialogue_prefix'])
    def test_named_modes_select_the_original_execution(self):
        engine=EvaluationEngine.__new__(EvaluationEngine);engine.runtime={}
        for public,legacy in EVAL_MODE_ALIASES.items():
            self.assertEqual(engine._resolve_eval_mode(public),engine._resolve_eval_mode(legacy))
