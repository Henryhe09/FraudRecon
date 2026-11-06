import argparse
import glob
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import config
from logic.dialogue_reconstructor import FactAnchoredDialogueReconstructor

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - graceful fallback when tqdm is unavailable
    tqdm = None


THREAD_LOCAL = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 Stage1 结构化标注逐阶段还原对话")
    parser.add_argument(
        "--input_pattern",
        default=config.INPUT_PATTERN,
        required=not bool(config.INPUT_PATTERN),
        help="输入分片文件 glob 模式",
    )
    parser.add_argument(
        "--output_jsonl",
        default=config.OUTPUT_JSONL,
        required=not bool(config.OUTPUT_JSONL),
        help="输出 JSONL 路径",
    )
    parser.add_argument(
        "--full_output_dir",
        default=config.FULL_OUTPUT_DIR,
        help="每条样本完整 JSON 输出目录",
    )
    parser.add_argument("--model", default=config.DEFAULT_MODEL, help="模型名称")
    parser.add_argument(
        "--base_url",
        default=config.OPENAI_BASE_URL,
        help="兼容 OpenAI 的 API Base URL",
    )
    parser.add_argument(
        "--api_key",
        default=config.OPENAI_API_KEY,
        help="API Key（可通过环境变量 OPENAI_API_KEY 提供）",
    )
    parser.add_argument("--workers", type=int, default=config.WORKERS, help="并发 worker 数")
    parser.add_argument(
        "--max_cases", type=int, default=config.MAX_CASES, help="最多处理多少条，0 表示全部"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从已存在 output_jsonl 断点续跑（默认关闭）",
    )
    parser.add_argument(
        "--retry_errors",
        action="store_true",
        help="断点续跑时，是否重试历史 error 样本（默认不重试）",
    )
    parser.add_argument(
        "--temperature", type=float, default=config.TEMPERATURE, help="生成温度"
    )
    parser.add_argument(
        "--manager_temperature",
        type=float,
        default=config.MANAGER_TEMPERATURE,
        help="manager 判定温度",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=config.MAX_TOKENS, help="每阶段最大输出 token"
    )
    parser.add_argument(
        "--max_retries", type=int, default=config.MAX_RETRIES, help="每阶段最大重试次数"
    )
    parser.add_argument(
        "--retry_delay", type=float, default=config.RETRY_DELAY, help="重试间隔秒数"
    )
    parser.add_argument(
        "--tail_size", type=int, default=config.TAIL_SIZE, help="跨阶段滑动窗口大小"
    )
    parser.add_argument(
        "--max_turns_per_phase",
        type=int,
        default=config.MAX_TURNS_PER_PHASE,
        help="每个阶段最多对话回合数（left+right 循环）",
    )
    args = parser.parse_args()
    if not args.full_output_dir:
        args.full_output_dir = str(Path(args.output_jsonl).parent / "reconstructed_full_dialogues")
    return args


def load_processed_keys(
    output_jsonl: Path,
    retry_errors: bool,
) -> Set[Tuple[str, int]]:
    processed: Set[Tuple[str, int]] = set()
    if not output_jsonl.exists():
        return processed

    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_chunk = item.get("source_chunk")
            index = item.get("index")
            status = item.get("status")
            if not isinstance(source_chunk, str) or not isinstance(index, int):
                continue
            if retry_errors and status == "error":
                continue
            processed.add((source_chunk, index))
    return processed


def iter_ok_records(
    input_pattern: str,
    processed_keys: Set[Tuple[str, int]],
    max_cases: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    files = sorted(glob.glob(input_pattern))
    if not files:
        raise FileNotFoundError(f"未匹配到输入文件: {input_pattern}")

    tasks: List[Tuple[str, Dict[str, Any]]] = []
    for fp in files:
        fallback_source_chunk = Path(fp).name
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            results = payload.get("results", [])
        elif isinstance(payload, list):
            results = payload
        else:
            results = []
        if not isinstance(results, list):
            continue
        for rec in results:
            if not isinstance(rec, dict):
                continue
            index = rec.get("index")
            if not isinstance(index, int):
                continue
            if rec.get("status") != "ok":
                continue
            if not isinstance(rec.get("result"), dict):
                continue
            source_chunk = rec.get("source_file")
            if not isinstance(source_chunk, str) or not source_chunk.strip():
                source_chunk = fallback_source_chunk
            key = (source_chunk, index)
            if key in processed_keys:
                continue
            tasks.append((source_chunk, rec))
            if max_cases > 0 and len(tasks) >= max_cases:
                return tasks
    return tasks


def get_reconstructor(args: argparse.Namespace) -> FactAnchoredDialogueReconstructor:
    reconstructor = getattr(THREAD_LOCAL, "reconstructor", None)
    if reconstructor is None:
        reconstructor = FactAnchoredDialogueReconstructor(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            manager_temperature=args.manager_temperature,
            max_tokens=args.max_tokens,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            tail_size=args.tail_size,
            max_turns_per_phase=args.max_turns_per_phase,
        )
        THREAD_LOCAL.reconstructor = reconstructor
    return reconstructor


def process_one_record(
    args: argparse.Namespace,
    source_chunk: str,
    rec: Dict[str, Any],
) -> Dict[str, Any]:
    index = rec.get("index")
    case_id = rec.get("案件编号")
    fraud_type = rec.get("案件类别")
    case_struct = rec.get("result") or {}
    case_meta = {
        "source_chunk": source_chunk,
        "index": index,
        "案件编号": case_id,
        "案件类别": fraud_type,
    }

    reconstructor = get_reconstructor(args)
    try:
        reconstructed = reconstructor.reconstruct_case(
            structured_case=case_struct,
            case_meta=case_meta,
        )
        return {
            "source_chunk": source_chunk,
            "index": index,
            "案件编号": case_id,
            "案件类别": fraud_type,
            "status": "ok",
            "reconstructed_dialogue": reconstructed.get("reconstructed_dialogue", {}),
        }
    except Exception as e:
        return {
            "source_chunk": source_chunk,
            "index": index,
            "案件编号": case_id,
            "案件类别": fraud_type,
            "status": "error",
            "error": str(e),
        }


def build_progress(total: int):
    if tqdm is None:
        return None
    return tqdm(
        total=total,
        desc="Reconstruct",
        unit="case",
        dynamic_ncols=True,
        leave=True,
    )


def main() -> None:
    args = parse_args()
    if not args.api_key or args.api_key == "your-api-key":
        raise EnvironmentError("缺少有效 API Key。请通过 OPENAI_API_KEY 或 --api_key 提供。")

    output_jsonl = Path(args.output_jsonl)
    full_output_dir = Path(args.full_output_dir)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    full_output_dir.mkdir(parents=True, exist_ok=True)

    processed_keys: Set[Tuple[str, int]] = set()
    if args.resume:
        processed_keys = load_processed_keys(output_jsonl, retry_errors=args.retry_errors)

    tasks = iter_ok_records(
        input_pattern=args.input_pattern,
        processed_keys=processed_keys,
        max_cases=args.max_cases,
    )
    total = len(tasks)
    total_phases = sum(
        len((rec.get("result") or {}).get("fraud_phases") or [])
        for _, rec in tasks
        if isinstance(rec, dict)
    )
    worst_api_calls = total_phases * (1 + 4 * args.max_turns_per_phase)
    print(f"待处理样本数: {total}")
    print(f"阶段总数(估计): {total_phases}")
    print(f"最坏情况下 API 调用次数估计: {worst_api_calls}")
    print(f"输入模式: {args.input_pattern}")
    print(f"输出 JSONL: {output_jsonl}")
    print(f"完整输出目录: {full_output_dir}")
    print(f"并发 workers: {args.workers}")
    if total == 0:
        return

    started = time.time()
    ok_count = 0
    err_count = 0

    progress = build_progress(total)

    with output_jsonl.open("a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(process_one_record, args, source_chunk, rec)
                for source_chunk, rec in tasks
            ]
            for i, future in enumerate(as_completed(futures), start=1):
                item = future.result()
                if item.get("status") == "ok":
                    ok_count += 1
                else:
                    err_count += 1

                # 追加到 JSONL
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                f_out.flush()

                # 单条完整文件，便于排查/二次处理
                index = item.get("index", -1)
                case_id = item.get("案件编号")
                if case_id is not None and str(case_id).strip() != "":
                    full_name = f"reconstructed_{case_id}.json"
                elif isinstance(index, int):
                    full_name = f"reconstructed_unknown_idx_{index:06d}.json"
                else:
                    full_name = "reconstructed_unknown_idx_unknown.json"
                with (full_output_dir / full_name).open("w", encoding="utf-8") as fw:
                    json.dump(item, fw, ensure_ascii=False, indent=2)

                elapsed = time.time() - started
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(ok=ok_count, err=err_count, elapsed=f"{elapsed:.1f}s")
                print(
                    f"[{i}/{total}] idx={item.get('index')} status={item.get('status')} "
                    f"ok={ok_count} error={err_count} elapsed={elapsed:.1f}s"
                )

    if progress is not None:
        progress.close()

    elapsed = time.time() - started
    print(f"完成，总计: {total}, 成功: {ok_count}, 失败: {err_count}, 耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
