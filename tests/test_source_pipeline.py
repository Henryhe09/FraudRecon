"""Offline checks for source fidelity and the thin FPE -> FADR connection."""
import ast
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ndrr.fpe import extract as extraction
from ndrr.cli import build_parser, main, reconstruction_command


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((ROOT / "ndrr/examples/fraud_results_part_0000.json").read_text())["results"][0]["result"]

SDK_STUB = '''
import json
import os
from types import SimpleNamespace

class OpenAI:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        if "你是阶段管理者（Manager）" in system:
            role = "manager"
            content = json.dumps({"should_end_phase": True, "reason": "offline fixture", "coverage": "high"})
        elif "扮演的是诈骗者（S）" in system:
            role, content = "left", "您好，我想和您确认订单情况。"
        else:
            role, content = "right", "我看了官方订单页面，没有异常。"
        with open(os.environ["STUB_LOG"], "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"role": role, "model": kwargs["model"]}) + "\\n")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
'''


class FidelityTests(unittest.TestCase):
    def test_signal_prompt_removal_keeps_original_phase_vocabulary(self):
        self.assertNotIn("## 策略标签定义", extraction.SYSTEM_PROMPT)
        self.assertNotIn("## 策略标注规则", extraction.SYSTEM_PROMPT)
        self.assertNotIn('"tactics"', extraction.USER_PROMPT_TEMPLATE)
        for name in ("initial_contact", "trust_building", "inducement", "operation_guidance",
                     "fund_transfer", "sustained_pressure", "discovery"):
            self.assertIn(name + ":", extraction.SYSTEM_PROMPT)
        self.assertNotIn("phase_subtype", extraction.USER_PROMPT_TEMPLATE)
        self.assertIn("金额严格对齐", extraction.SYSTEM_PROMPT)

    def test_credential_literals_are_replaced_with_environment_lookups(self):
        for path, variable in ((ROOT / "ndrr/fpe/config.py", "API_KEY"),
                               (ROOT / "ndrr/fadr/config.py", "OPENAI_API_KEY")):
            tree = ast.parse(path.read_text())
            assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == variable for t in node.targets))
            self.assertIsInstance(assignment.value, ast.Call)
            self.assertEqual(assignment.value.args[-1].value, "")

    def test_wrapper_does_not_replace_original_defaults(self):
        args = build_parser().parse_args(["reconstruct", "--input", "chunks/*.json", "--output", "out.jsonl"])
        command = reconstruction_command(args, args.input, args.output, Path("full"))
        for flag in ("--workers", "--max_turns_per_phase", "--manager_temperature", "--tail_size", "--max_tokens"):
            self.assertNotIn(flag, command)
        self.assertEqual(extraction.MAX_WORKERS, 5)
        self.assertEqual(extraction.CHUNK_SIZE, 5000)

    def test_explicit_overrides_forward_original_parameter_meanings(self):
        args = build_parser().parse_args([
            "reconstruct", "--input", "chunks/*.json", "--output", "out.jsonl",
            "--max-turns-per-phase", "5", "--director-temperature", "0.3", "--resume", "--retry-errors",
        ])
        command = reconstruction_command(args, args.input, args.output, "full")
        self.assertEqual(command[command.index("--max_turns_per_phase") + 1], "5")
        self.assertEqual(command[command.index("--manager_temperature") + 1], "0.3")
        self.assertIn("--resume", command)
        self.assertIn("--retry_errors", command)


class ConnectionTests(unittest.TestCase):
    def invoke(self, args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with patch.dict(os.environ, {'OPENAI_MODEL':'offline-model', 'OPENAI_BASE_URL':'https://offline.invalid/v1'}):
                return main(args + (['--dialogue-only'] if args[0] == 'run' else []))

    def test_original_extraction_chunks_feed_original_reconstruction(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            sdk = folder / "openai.py"
            sdk.write_text(SDK_STUB)
            output = folder / "run"
            log = folder / "sdk_calls.jsonl"
            body = json.dumps({"choices": [{"message": {"content": json.dumps(EXAMPLE)}}]}).encode()
            calls = []

            def fake_urlopen(request, timeout):
                calls.append(json.loads(request.data))
                return io.BytesIO(body)

            env = {"OPENAI_API_KEY": "offline-test-key", "FRAUDRECON_API_KEY": "offline-test-key",
                   "PYTHONPATH": str(folder), "STUB_LOG": str(log), "PYTHONDONTWRITEBYTECODE": "1"}
            args = ["run", "--input", str(ROOT / "ndrr/examples/narratives.json"),
                    "--output-dir", str(output), "--workers", "1"]
            with patch.dict(os.environ, env), patch.object(extraction.urllib.request, "urlopen", fake_urlopen):
                self.assertEqual(self.invoke(args), 0)
                first_log = log.read_bytes()
                # Preserve the original FPE index-based skip and FADR --resume behavior.
                self.assertEqual(self.invoke(args + ["--resume"]), 0)
                self.assertEqual(log.read_bytes(), first_log)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["temperature"], 0.2)
            chunk = json.loads((output / "fraud_results_chunks/fraud_results_part_0000.json").read_text())
            self.assertIn("results", chunk)
            self.assertEqual(chunk["results"][0]["result"], EXAMPLE)
            rows = [json.loads(line) for line in (output / "results/reconstructed_dialogues.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "ok")
            self.assertIn("reconstructed_dialogue", rows[0])
            self.assertNotIn("dialogue", rows[0])
            self.assertEqual(len(rows[0]["reconstructed_dialogue"]["flat_turns"]), 4)
            roles = [json.loads(line)["role"] for line in log.read_text().splitlines()]
            self.assertEqual(roles, ["left", "right", "manager"] * 2)
            self.assertTrue((output / "results/reconstructed_full_dialogues/reconstructed_case_000001.json").exists())

    def test_original_extraction_continues_to_skip_prior_error_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "chunks"
            args = ["extract", "--input", str(ROOT / "ndrr/examples/narratives.json"),
                    "--output", str(output), "--workers", "1", "--max-retries", "1", "--retry-delay", "0"]
            with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-key", "FRAUDRECON_API_KEY": "offline-test-key"}), \
                    patch.object(extraction, "call_api", side_effect=RuntimeError("offline failure")) as call:
                self.assertEqual(self.invoke(args), 0)
                self.assertEqual(self.invoke(args), 0)
            self.assertEqual(call.call_count, 1)
            chunk = json.loads((output / "fraud_results_part_0000.json").read_text())
            self.assertEqual(chunk["results"][0]["status"], "error")


if __name__ == "__main__":
    unittest.main()

