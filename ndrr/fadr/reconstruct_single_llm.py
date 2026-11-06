import argparse
import glob
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import config
from utils.openai_client import OpenAIClient


THREAD_LOCAL = threading.local()


SYSTEM_PROMPT = """
你是“公安笔录反诈对话还原”专家。你的任务是根据结构化欺诈流程，直接重建完整阶段化对话。

必须遵守:
1. 事实锚定: 只能使用输入里的事实，不得编造新人物/新金额/新平台/新身份。
2. 阶段一致: 对话必须按输入 fraud_phases 的顺序展开，不得跳阶段或反向。
3. 角色格式: speaker 仅允许 "S"(诈骗者) 或 "V"(受害人)。
4. 风格要求: 中文口语化、简洁自然，不要写旁白/分析/解释。
5. 金额一致: 若涉及金额，必须与输入 amount 保持一致。
6. 输出约束: 仅输出 JSON 对象，不要 Markdown 代码块。

输出 JSON 格式:
{
  "phase_dialogues": [
    {
      "phase_id": int,
      "phase_type": "string",
      "dialogue_turns": [
        {"speaker": "S", "utterance": "string"},
        {"speaker": "V", "utterance": "string"}
      ],
      "phase_notes": "string"
    }
  ]
}
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用单一 LLM 从 Stage1 标注重建对话")
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
    parser.add_argument("--base_url", default=config.OPENAI_BASE_URL, help="API Base URL")
    parser.add_argument("--api_key", default=config.OPENAI_API_KEY, help="API Key")
    parser.add_argument("--workers", type=int, default=config.WORKERS, help="并发 worker 数")
    parser.add_argument("--max_cases", type=int, default=config.MAX_CASES, help="最多处理条数，0=全部")
    parser.add_argument("--resume", action="store_true", help="从已有 output_jsonl 断点续跑")
    parser.add_argument("--retry_errors", action="store_true", help="续跑时是否重试历史 error")
    parser.add_argument("--temperature", type=float, default=config.TEMPERATURE, help="生成温度")
    parser.add_argument("--max_tokens", type=int, default=config.MAX_TOKENS, help="单次最大输出 token")
    parser.add_argument("--max_retries", type=int, default=config.MAX_RETRIES, help="重试次数")
    parser.add_argument("--retry_delay", type=float, default=config.RETRY_DELAY, help="重试间隔秒")
    args = parser.parse_args()
    if not args.full_output_dir:
        args.full_output_dir = str(Path(args.output_jsonl).parent / "reconstructed_full_dialogues_single_llm")
    return args


def get_client(args: argparse.Namespace) -> OpenAIClient:
    client = getattr(THREAD_LOCAL, "client", None)
    if client is None:
        client = OpenAIClient(api_key=args.api_key, base_url=args.base_url)
        THREAD_LOCAL.client = client
    return client


def load_processed_keys(output_jsonl: Path, retry_errors: bool) -> Set[Tuple[str, int]]:
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


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def build_user_prompt(case_struct: Dict[str, Any], case_meta: Dict[str, Any]) -> str:
    return f"""
请根据以下结构化案件信息重建完整对话。

【案件元信息】
{_json_dump(case_meta)}

【结构化信息】
{_json_dump(case_struct)}

要求:
1. 以 fraud_phases 为主线，按 phase_id 顺序生成每个阶段对话。
2. 每个阶段建议 2-8 轮对话，确保与阶段类型一致。
3. discovery 阶段体现“受害人察觉异常/被骗”。
4. 若某字段是 unknown/null/未提及，则保持模糊表达，不要补充细节。
5. 仅返回 JSON，不要额外文字。
""".strip()


def _normalize_speaker(raw: str) -> str:
    s = (raw or "").strip().lower()
    if s in {"s", "scammer", "left", "诈骗者"}:
        return "S"
    if s in {"v", "victim", "right", "受害人", "用户"}:
        return "V"
    return ""


def _fallback_turns(phase: Dict[str, Any]) -> List[Dict[str, str]]:
    scammer_text = (
        str(phase.get("scammer_speech_summary") or "").strip()
        or str(phase.get("scammer_action") or "").strip()
        or "我这边按流程给您处理，您先配合一下。"
    )
    victim_text = (
        str(phase.get("victim_action") or "").strip()
        or str(phase.get("victim_psychology") or "").strip()
        or "好的，我先按你说的看看。"
    )
    return [
        {"speaker": "S", "utterance": scammer_text},
        {"speaker": "V", "utterance": victim_text},
    ]


def _parse_json_response(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    candidates: List[str] = []
    stack: List[int] = []
    for idx, ch in enumerate(raw):
        if ch == "{":
            stack.append(idx)
        elif ch == "}":
            if not stack:
                continue
            start = stack.pop()
            if not stack:
                candidates.append(raw[start : idx + 1])
    candidates.sort(key=len, reverse=True)
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("无法解析模型 JSON 输出")


def _normalize_result(case_struct: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, Any]:
    src_phases = case_struct.get("fraud_phases") or []
    src_by_id = {}
    for p in src_phases:
        if isinstance(p, dict) and isinstance(p.get("phase_id"), int):
            src_by_id[p["phase_id"]] = p

    raw_phase_dialogues = parsed.get("phase_dialogues") or []
    if not isinstance(raw_phase_dialogues, list):
        raw_phase_dialogues = []

    normalized_phase_dialogues: List[Dict[str, Any]] = []
    flat_turns: List[Dict[str, Any]] = []
    turn_id = 1

    for i, item in enumerate(raw_phase_dialogues, start=1):
        if not isinstance(item, dict):
            continue
        phase_id = item.get("phase_id")
        if not isinstance(phase_id, int):
            phase_id = i
        src_phase = src_by_id.get(phase_id, {})
        phase_type = item.get("phase_type") or src_phase.get("phase_type") or "unknown"

        turns = item.get("dialogue_turns") or []
        norm_turns: List[Dict[str, str]] = []
        if isinstance(turns, list):
            for t in turns:
                if not isinstance(t, dict):
                    continue
                speaker = _normalize_speaker(str(t.get("speaker", "")))
                utt = str(t.get("utterance", "")).strip()
                if not speaker or not utt:
                    continue
                norm_turns.append({"speaker": speaker, "utterance": utt})

        if not norm_turns:
            norm_turns = _fallback_turns(src_phase or {})

        # enforce alternation
        alternating: List[Dict[str, str]] = []
        prev = ""
        for t in norm_turns:
            cur = t["speaker"]
            if cur == prev:
                cur = "V" if prev == "S" else "S"
            alternating.append({"speaker": cur, "utterance": t["utterance"]})
            prev = cur

        phase_entry = {
            "phase_id": phase_id,
            "phase_type": phase_type,
            "dialogue_turns": alternating,
            "phase_notes": str(item.get("phase_notes", "")),
        }
        normalized_phase_dialogues.append(phase_entry)

        for t in alternating:
            flat_turns.append(
                {
                    "turn_id": turn_id,
                    "phase_id": phase_id,
                    "phase_type": phase_type,
                    "speaker": t["speaker"],
                    "utterance": t["utterance"],
                }
            )
            turn_id += 1

    # If model didn't return phases, fallback from source phases.
    if not normalized_phase_dialogues:
        for p in src_phases:
            if not isinstance(p, dict):
                continue
            phase_id = p.get("phase_id", 0)
            phase_type = p.get("phase_type", "unknown")
            turns = _fallback_turns(p)
            phase_entry = {
                "phase_id": phase_id,
                "phase_type": phase_type,
                "dialogue_turns": turns,
                "phase_notes": "fallback",
            }
            normalized_phase_dialogues.append(phase_entry)
            for t in turns:
                flat_turns.append(
                    {
                        "turn_id": turn_id,
                        "phase_id": phase_id,
                        "phase_type": phase_type,
                        "speaker": t["speaker"],
                        "utterance": t["utterance"],
                    }
                )
                turn_id += 1

    return {
        "phase_dialogues": normalized_phase_dialogues,
        "flat_turns": flat_turns,
        "turning_point": case_struct.get("turning_point", "未提及"),
        "total_loss": case_struct.get("total_loss", "未提及"),
    }


def process_one_record(args: argparse.Namespace, source_chunk: str, rec: Dict[str, Any]) -> Dict[str, Any]:
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

    client = get_client(args)
    user_prompt = build_user_prompt(case_struct=case_struct, case_meta=case_meta)

    last_err = ""
    reply = ""
    for _ in range(args.max_retries):
        try:
            reply = client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            parsed = _parse_json_response(reply)
            reconstructed = _normalize_result(case_struct=case_struct, parsed=parsed)
            return {
                "source_chunk": source_chunk,
                "index": index,
                "案件编号": case_id,
                "案件类别": fraud_type,
                "status": "ok",
                "reconstructed_dialogue": reconstructed,
            }
        except Exception as e:
            last_err = str(e)
            time.sleep(args.retry_delay)

    return {
        "source_chunk": source_chunk,
        "index": index,
        "案件编号": case_id,
        "案件类别": fraud_type,
        "status": "error",
        "error": last_err,
    }


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
    print(f"待处理样本数: {total}")
    print(f"输入模式: {args.input_pattern}")
    print(f"输出 JSONL: {output_jsonl}")
    print(f"完整输出目录: {full_output_dir}")
    print(f"并发 workers: {args.workers}")
    if total == 0:
        return

    started = time.time()
    ok_count = 0
    err_count = 0

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

                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
                f_out.flush()

                index = item.get("index", -1)
                case_id = item.get("案件编号")
                if case_id is not None and str(case_id).strip() != "":
                    name = f"reconstructed_single_{case_id}.json"
                elif isinstance(index, int):
                    name = f"reconstructed_single_unknown_idx_{index:06d}.json"
                else:
                    name = "reconstructed_single_unknown_idx_unknown.json"
                with (full_output_dir / name).open("w", encoding="utf-8") as fw:
                    json.dump(item, fw, ensure_ascii=False, indent=2)

                elapsed = time.time() - started
                print(
                    f"[{i}/{total}] idx={item.get('index')} status={item.get('status')} "
                    f"ok={ok_count} error={err_count} elapsed={elapsed:.1f}s"
                )

    elapsed = time.time() - started
    print(f"完成，总计: {total}, 成功: {ok_count}, 失败: {err_count}, 耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
