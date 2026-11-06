import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from ndrr.fpe import extract

ROOT=Path(__file__).resolve().parents[1]

class PathConfigurationTests(unittest.TestCase):
    def test_fpe_does_not_select_a_historical_input(self):
        with patch.object(extract,'INPUT_PATH',None),patch.object(extract,'load_cases') as load:
            with self.assertRaisesRegex(ValueError,'--input'):extract.run_batch()
            load.assert_not_called()

    def test_standalone_reconstructors_require_explicit_paths(self):
        env={k:v for k,v in os.environ.items() if not k.startswith('RECON_')}
        for script in ['reconstruct.py','reconstruct_single_llm.py']:
            r=subprocess.run([sys.executable,str(ROOT/'ndrr/fadr'/script)],env=env,text=True,capture_output=True)
            self.assertEqual(r.returncode,2,r.stderr)
            self.assertIn('--input_pattern',r.stderr)
            self.assertIn('--output_jsonl',r.stderr)

    def test_annotation_does_not_search_cwd_for_source_chunks(self):
        from ndrr.annotation import annotate_evidence_chain as module
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'fraud_results_chunks').mkdir()
            args=SimpleNamespace(input='unused.json',output=str(root/'out.jsonl'),output_json_dir='',
                api_key='offline',base_url='https://offline.invalid/v1',model='offline-model',
                labels_jsonl='',start_index=0,end_index=None,limit=None,retry_errors=False,
                source_chunk_dir='',auto_validate=False,timeout=1)
            previous=Path.cwd()
            try:
                os.chdir(root)
                with patch.object(module,'parse_args',return_value=args),patch.object(module,'load_records',return_value=[{}]),patch.object(module,'load_done_status',return_value={}):
                    # Keep the actual source-directory selection and stop before generation.
                    with patch.object(module,'ThreadLocalLLMClient'),patch.object(module,'FraudSourceChunkResolver') as resolver,patch.object(module,'run_generation_pass'),contextlib.redirect_stdout(io.StringIO()):
                        module.main()
                        resolver.assert_called_once_with(source_chunk_dir=None)
            finally:os.chdir(previous)
