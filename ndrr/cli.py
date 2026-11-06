"""NDRR procedure extraction, dialogue reconstruction and annotation."""
from __future__ import annotations
import argparse
import importlib
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from . import __version__
RECONSTRUCTION_DIR = Path(__file__).parent / 'fadr'

def build_parser():
    parser = argparse.ArgumentParser(description='NDRR: Narrative-to-Dialogue Reverse Reconstruction (FPE, FADR)')
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    for (name, help_text) in (('extract', 'Fraud Procedure Extraction (FPE)'), ('reconstruct', 'Fact-Anchored Dialogue Reconstruction (FADR)'), ('run', 'Extract, reconstruct, annotate and export dialogues')):
        public_name = {'extract': 'fpe', 'reconstruct': 'fadr'}.get(name, name)
        aliases = [name] if public_name != name else []
        sub = commands.add_parser(public_name, aliases=aliases, help=help_text)
        sub.add_argument('--input', required=True, help='Narrative JSON array (only narrative is required); for reconstruct, a quoted extraction-chunk glob')
        if name == 'run':
            sub.add_argument('--output-dir', type=Path, required=True)
            sub.add_argument('--dialogue-only', action='store_true', help='Stop after FADR')
            sub.add_argument('--annotation-model', default=None, help='Annotation model; or set OPENAI_ANNOTATION_MODEL')
            sub.add_argument('--split', default='train', choices=['train', 'test'])
        else:
            sub.add_argument('--output', type=Path, required=True, help='Extraction chunk directory, or reconstruction JSONL file')
        sub.add_argument('--model', default=None, help='Generation model name; or set OPENAI_MODEL')
        sub.add_argument('--base-url', default=None)
        sub.add_argument('--workers', type=int, default=None, help="Concurrent workers for extraction, reconstruction and evidence-chain annotation")
        sub.add_argument('--timeout', type=float, default=None)
        sub.add_argument('--max-retries', type=int, default=None)
        sub.add_argument('--retry-delay', type=float, default=None)
        if name in ('extract', 'run'):
            sub.add_argument('--chunk-size', type=int, default=None, help='Extraction chunk size (5000 by default)')
        if name in ('reconstruct', 'run'):
            sub.add_argument('--full-output-dir', type=Path, default=None)
            sub.add_argument('--resume', action='store_true', help='Resume completed reconstruction records')
            sub.add_argument('--retry-errors', action='store_true', help='Retry reconstruction and evidence-chain errors when resuming')
            sub.add_argument('--temperature', type=float, default=None, help='FADR role-agent temperature')
            sub.add_argument('--director-temperature', type=float, default=None, help='Director Agent temperature')
            sub.add_argument('--max-tokens', type=int, default=None)
            sub.add_argument('--tail-size', type=int, default=None)
            sub.add_argument('--max-turns-per-phase', type=int, default=None, help='Maximum Scammer/Victim exchange rounds per phase')
            flag = '--max-cases' if name == 'reconstruct' else '--reconstruction-max-cases'
            sub.add_argument(flag, dest='max_cases', type=int, default=None, help='Maximum reconstructed cases; does not limit procedure extraction')
    return parser

def effective_env():
    env = os.environ.copy()
    for suffix in ('API_KEY', 'BASE_URL', 'MODEL'):
        value = env.get('FRAUDRECON_' + suffix)
        if value is not None:
            env['OPENAI_' + suffix] = value
    return env

@contextmanager
def module_settings(module, settings):
    previous = {name: getattr(module, name) for name in settings}
    try:
        for (name, value) in settings.items():
            setattr(module, name, value)
        yield
    finally:
        for (name, value) in previous.items():
            setattr(module, name, value)

def run_extraction(args, input_path, output_dir):
    stage = importlib.import_module('ndrr.fpe.extract')
    env = effective_env()
    settings = {'INPUT_PATH': Path(input_path).expanduser().resolve(), 'OUTPUT_DIR': Path(output_dir).expanduser().resolve(), 'API_KEY': env.get('OPENAI_API_KEY', '')}
    for (target, value) in (('MODEL_NAME', args.model or env.get('OPENAI_MODEL')), ('BASE_URL', args.base_url or env.get('OPENAI_BASE_URL')), ('MAX_WORKERS', args.workers), ('CHUNK_SIZE', args.chunk_size), ('REQUEST_TIMEOUT_SECONDS', args.timeout), ('MAX_RETRIES', args.max_retries), ('RETRY_SLEEP_SECONDS', args.retry_delay)):
        if value is not None:
            settings[target] = value
    with module_settings(stage, settings):
        stage.run_batch()

def reconstruction_command(args, input_pattern, output_file, full_output_dir):
    command = [sys.executable, str(RECONSTRUCTION_DIR / 'reconstruct.py'), '--input_pattern', str(Path(input_pattern).expanduser().resolve()), '--output_jsonl', str(Path(output_file).expanduser().resolve()), '--full_output_dir', str(Path(full_output_dir).expanduser().resolve())]
    for (flag, value) in (('--model', args.model), ('--base_url', args.base_url), ('--workers', args.workers), ('--max_cases', args.max_cases), ('--temperature', args.temperature), ('--manager_temperature', args.director_temperature), ('--max_tokens', args.max_tokens), ('--max_retries', args.max_retries), ('--retry_delay', args.retry_delay), ('--tail_size', args.tail_size), ('--max_turns_per_phase', args.max_turns_per_phase)):
        if value is not None:
            command.extend((flag, str(value)))
    if args.resume:
        command.append('--resume')
    if args.retry_errors:
        command.append('--retry_errors')
    return command

def run_reconstruction(args, input_pattern, output_file, full_output_dir):
    env = effective_env()
    if args.timeout is not None:
        env['OPENAI_TIMEOUT_SECONDS'] = str(args.timeout)
    return subprocess.run(reconstruction_command(args, input_pattern, output_file, full_output_dir), env=env).returncode

def main(argv=None):
    args = build_parser().parse_args(argv)
    args.command = {'fpe': 'extract', 'fadr': 'reconstruct'}.get(args.command, args.command)
    if args.command == 'run':
        chunk_dir = args.output_dir / 'fraud_results_chunks'
        output = args.output_dir / 'results/reconstructed_dialogues.jsonl'
        full = args.full_output_dir or args.output_dir / 'results/reconstructed_full_dialogues'
    elif args.command == 'reconstruct':
        output = args.output
        full = args.full_output_dir or output.parent / 'reconstructed_full_dialogues'
    elif args.command == 'extract':
        chunk_dir = args.output
    try:
        env = effective_env()
        missing = [name for name,value in [('model (--model / OPENAI_MODEL)', args.model or env.get('OPENAI_MODEL')), ('base URL (--base-url / OPENAI_BASE_URL)', args.base_url or env.get('OPENAI_BASE_URL')), ('API key (OPENAI_API_KEY)', env.get('OPENAI_API_KEY'))] if not value or not value.strip()]
        if args.command == 'run' and not args.dialogue_only:
            args.annotation_model = args.annotation_model or env.get('OPENAI_ANNOTATION_MODEL')
            if not args.annotation_model: missing.append('annotation model (--annotation-model / OPENAI_ANNOTATION_MODEL)')
        if missing: raise ValueError('Missing explicit configuration: ' + ', '.join(missing))
        if args.command == 'run' and not args.resume and output.exists() and output.stat().st_size:
            raise ValueError('Reconstruction output already exists. Use --resume or a new --output-dir.')
        if args.command in ('extract', 'run'):
            run_extraction(args, args.input, chunk_dir)
        if args.command in ('reconstruct', 'run'):
            pattern = str(chunk_dir / 'fraud_results_part_*.json') if args.command == 'run' else args.input
            code = run_reconstruction(args, pattern, output, full)
            if code == 0 and args.command == 'run' and not args.dialogue_only:
                from .export import export
                packaged = args.output_dir / 'dataset'
                annotation_input = args.output_dir / 'annotations/input'
                export(output, chunk_dir / 'fraud_results_part_*.json', annotation_input, split=args.split)
                signal_output = args.output_dir / 'annotations/signals.jsonl'
                signal_output.parent.mkdir(parents=True, exist_ok=True)
                signal_command = [sys.executable, '-m', 'ndrr.annotation.annotate_signals', '--input', str(annotation_input/'samples.jsonl'), '--output', str(signal_output)]
                if args.resume: signal_command.append('--resume')
                generation_model = args.model or effective_env().get('OPENAI_MODEL')
                if generation_model: signal_command.extend(['--model', generation_model])
                endpoint = args.base_url or effective_env().get('OPENAI_BASE_URL')
                if endpoint: signal_command.extend(['--base-url', endpoint])
                code = subprocess.run(signal_command, env=effective_env()).returncode
                if code != 0: return code
                annotations = args.output_dir / 'annotations/evidence_chain.jsonl' 
                annotations.parent.mkdir(parents=True, exist_ok=True)
                annotation_command = [sys.executable, '-m', 'ndrr.annotation.annotate_evidence_chain',
                    '--input', str(signal_output), '--output', str(annotations),
                    '--model', args.annotation_model, '--label-mode', 'constrained',
                    '--max-workers', str(args.workers or 2)]
                if args.retry_errors: annotation_command.append('--retry-errors')
                endpoint = args.base_url or effective_env().get('OPENAI_BASE_URL')
                if endpoint: annotation_command.extend(['--base-url', endpoint])
                code = subprocess.run(annotation_command, env=effective_env()).returncode
                if code == 0:
                    export(output, chunk_dir / 'fraud_results_part_*.json', packaged, annotations, args.split, signal_output)
            return code
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f'运行失败: {exc}', file=sys.stderr)
        return 1
