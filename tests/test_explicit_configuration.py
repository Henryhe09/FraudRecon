"""Missing connection settings must fail before any paid stage is invoked."""
import contextlib
import io
import os
import unittest
from unittest.mock import patch
from ndrr.cli import main
from ndrr.fpe import extract as fpe
from bench.adapters.factory import build_adapter

class ExplicitConfigurationTests(unittest.TestCase):
    def invoke(self,args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return main(args)

    def test_missing_settings_fail_before_extraction(self):
        with patch.dict(os.environ,{},clear=True),patch('ndrr.cli.run_extraction') as stage:
            self.assertEqual(self.invoke(['run','--input','absent.json','--output-dir','unused']),1)
            stage.assert_not_called()

    def test_full_run_requires_annotation_model_before_generation(self):
        env={'OPENAI_API_KEY':'offline-key','OPENAI_BASE_URL':'https://offline.invalid/v1','OPENAI_MODEL':'offline-model'}
        with patch.dict(os.environ,env,clear=True),patch('ndrr.cli.run_extraction') as stage:
            self.assertEqual(self.invoke(['run','--input','absent.json','--output-dir','unused']),1)
            stage.assert_not_called()

    def test_direct_extraction_rejects_missing_endpoint_before_request(self):
        with patch.object(fpe,'API_KEY','offline-key'),patch.object(fpe,'BASE_URL',''),patch.object(fpe.urllib.request,'urlopen') as request:
            with self.assertRaises(EnvironmentError):fpe.call_api('system','user','offline-model')
            request.assert_not_called()

    def test_empty_real_model_profile_is_rejected_but_mock_remains_available(self):
        with self.assertRaises(ValueError):build_adapter({'provider':'openai_compatible','base_url':'','model_name':''})
        self.assertIsNotNone(build_adapter({'provider':'mock'}))
