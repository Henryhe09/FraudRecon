"""Offline checks for run reuse, annotation recovery and the local API."""
import contextlib
import copy
import importlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bench.core.evaluator import EvaluationEngine
from ndrr.annotation.annotate_signals import annotate_samples
from ndrr.cli import main
from ndrr.export import export
import test_dataset_export as export_tests

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkResumeTests(unittest.TestCase):
    def test_changed_task_or_profile_cannot_reuse_completed_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('configs', 'prompts', 'examples'):
                shutil.copytree(ROOT / 'bench' / name, root / 'bench' / name)
            engine = EvaluationEngine(root)
            try:
                args = dict(model_id='mock-main', judge_model_id='mock-judge',
                            eval_mode='detection_only', run_id='resume-check', max_workers=1)
                engine.run_batch(**args)
                with patch.object(engine, '_evaluate_case_payload') as evaluate:
                    engine.run_batch(**dict(args, max_workers=2))
                    evaluate.assert_not_called()
                    with self.assertRaisesRegex(ValueError, 'eval_mode'):
                        engine.run_batch(**dict(args, eval_mode='signals_only'))
                    engine.cfg.models['mock-main']['model_name'] = 'changed-model'
                    with self.assertRaisesRegex(ValueError, 'model_profile'):
                        engine.run_batch(**args)
                saved = json.loads(engine.store.get_run('resume-check')['config_json'])
                self.assertEqual(saved['eval_mode'], 'acc_only')
                self.assertNotIn('api_key', saved['model_profile'])
            finally:
                engine.store.conn.close()

    def test_changed_case_inventory_is_rejected(self):
        from bench.db.store import EvalStore
        with tempfile.TemporaryDirectory() as folder:
            store = EvalStore(Path(folder) / 'runs.db')
            try:
                store.create_or_resume_run('a', {'eval_mode': 'full'}, ['one'])
                with self.assertRaisesRegex(ValueError, 'case_ids'):
                    store.create_or_resume_run('a', {'eval_mode': 'full'}, ['two'])
                self.assertEqual([r['case_id'] for r in store.get_run_cases('a')], ['one'])
            finally:
                store.conn.close()


class AnnotationRecoveryTests(unittest.TestCase):
    def samples(self):
        return [{'case_id': cid, 'labels': {'level_2_phases': [{'phase_id': 1}]},
                 'reconstructed_dialogue': {'flat_turns': [{'utterance': '您好'}]}}
                for cid in ('one', 'two')]

    def response(self):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=json.dumps({'phases': [{'phase_id': 1, 'tactics': []}]})))])

    def test_signal_success_is_saved_before_later_failure_and_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'signals.jsonl'
            rows = self.samples()
            original = copy.deepcopy(rows)
            client = Mock()
            client.chat.completions.create.side_effect = [self.response(), RuntimeError('offline failure')]
            with self.assertRaisesRegex(RuntimeError, 'offline failure'):
                annotate_samples(rows, output, client, 'offline-model')
            self.assertEqual(json.loads(output.read_text())['case_id'], 'one')
            client.chat.completions.create.reset_mock(side_effect=True)
            client.chat.completions.create.return_value = self.response()
            annotate_samples(rows, output, client, 'offline-model', resume=True)
            self.assertEqual(client.chat.completions.create.call_count, 1)
            self.assertEqual([json.loads(x)['case_id'] for x in output.read_text().splitlines()], ['one', 'two'])
            self.assertEqual(rows, original)
            changed = copy.deepcopy(rows)
            changed[0]['reconstructed_dialogue']['flat_turns'][0]['utterance'] = '修改后的对话'
            with self.assertRaisesRegex(ValueError, 'Signal input changed'):
                annotate_samples(changed, output, client, 'offline-model', resume=True)

    def test_failed_annotation_does_not_replace_final_dataset_and_flags_are_forwarded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            chunks = root / 'fraud_results_chunks'
            chunks.mkdir()
            (root / 'results').mkdir()
            record, procedure = export_tests.ExportTests().fixture()
            record['source_chunk'] = 'fraud_results_part_0000.json'
            (chunks / record['source_chunk']).write_text(json.dumps({'results': [
                {'status': 'ok', 'index': 0, 'result': procedure}]}))
            output = root / 'results/reconstructed_dialogues.jsonl'
            output.write_text(json.dumps(record) + '\n')
            final = root / 'dataset/samples'
            final.mkdir(parents=True)
            marker = final / 'existing.json'
            marker.write_text('previous complete dataset')
            commands = []

            def annotation(command, env):
                commands.append(command)
                if 'ndrr.annotation.annotate_signals' in command:
                    self.assertIn('--resume', command)
                    source = Path(command[command.index('--input') + 1])
                    self.assertIn('annotations/input', str(source))
                    self.assertTrue(marker.exists())
                    return SimpleNamespace(returncode=0)
                self.assertIn('--retry-errors', command)
                return SimpleNamespace(returncode=1)

            env = {'OPENAI_API_KEY': 'offline-key', 'OPENAI_BASE_URL': 'https://offline.invalid/v1',
                   'OPENAI_MODEL': 'offline-model', 'OPENAI_ANNOTATION_MODEL': 'offline-annotation'}
            with patch.dict(os.environ, env), patch('ndrr.cli.run_extraction'), \
                    patch('ndrr.cli.run_reconstruction', return_value=0), \
                    patch('ndrr.cli.subprocess.run', side_effect=annotation), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['run', '--input', 'unused.json', '--output-dir', str(root),
                                       '--resume', '--retry-errors']), 1)
            self.assertEqual(len(commands), 2)
            self.assertEqual(marker.read_text(), 'previous complete dataset')
            self.assertEqual(list(final.iterdir()), [marker])

    def test_export_uses_successful_retry_in_append_log(self):
        record, procedure = export_tests.ExportTests().fixture()
        failed = dict(record, status='error')
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'part.json').write_text(json.dumps({'results': [
                {'status': 'ok', 'index': 0, 'result': procedure}]}))
            (root / 'reconstruction.jsonl').write_text(json.dumps(failed) + '\n' + json.dumps(record) + '\n')
            self.assertEqual(export(root / 'reconstruction.jsonl', str(root / 'part.json'), root / 'out'), 1)


class LocalApiTests(unittest.TestCase):
    def test_started_run_status_is_json_serializable(self):
        from fastapi.testclient import TestClient
        server = importlib.import_module('bench.api.server')
        fake_engine = Mock()
        fake_engine.store.get_run.return_value = None
        with patch.object(server, 'engine', fake_engine), patch.object(server, '_run_threads', {}), \
                patch.object(server, '_run_batch_job'):
            client = TestClient(server.app)
            started = client.post('/api/runs/start', json={
                'run_id': 'api-check', 'model_id': 'mock-main', 'judge_model_id': 'mock-judge'})
            self.assertEqual(started.status_code, 200)
            status = client.get('/api/runs/api-check')
            self.assertEqual(status.status_code, 200)
            self.assertEqual(status.json()['runtime']['status'], 'queued')
            self.assertNotIn('thread', status.json()['runtime'])
            self.assertEqual(client.post('/api/runs/start', json={
                'run_id': 'api-check', 'model_id': 'mock-main', 'judge_model_id': 'mock-judge'}).status_code, 409)
