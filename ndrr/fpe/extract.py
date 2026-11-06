"""Procedure extraction requests, batching and chunk output."""
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from ..inputs import load_narratives as load_cases
from .prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .config import (
    API_KEY,
    BASE_URL,
    INPUT_PATH,
    MODEL_NAME,
    OUTPUT_DIR,
    OUTPUT_FILE_PREFIX,
    CHUNK_SIZE,
    REQUEST_TIMEOUT_SECONDS,
    MAX_RETRIES,
    RETRY_SLEEP_SECONDS,
    MAX_WORKERS,
)


def call_api(system_prompt: str, user_prompt: str, model: str) -> str:
    if not API_KEY or not BASE_URL or not model:
        raise EnvironmentError("请显式配置 OPENAI_API_KEY、OPENAI_BASE_URL 和 OPENAI_MODEL")

    url = BASE_URL.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "stream": False,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            raw = resp.read().decode("utf-8")
        obj = json.loads(raw)
        return (obj["choices"][0]["message"]["content"] or "").strip()
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} 调用失败: {body}") from e
    except Exception as e:
        raise RuntimeError(
            "API 调用失败。请检查 BASE_URL、MODEL_NAME、API_KEY 是否正确。\n"
            f"当前 BASE_URL: {BASE_URL}\n"
            f"当前 MODEL_NAME: {model}\n"
            f"原始错误: {e}"
        ) from e


def try_parse_json(text: str):
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()
    return json.loads(raw)


def get_chunk_id(index: int) -> int:
    return index // CHUNK_SIZE


def get_chunk_path(chunk_id: int) -> Path:
    return OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}_part_{chunk_id:04d}.json"


def init_chunk_payload(chunk_id: int, total: int) -> dict:
    start_index = chunk_id * CHUNK_SIZE
    end_index = min((chunk_id + 1) * CHUNK_SIZE - 1, total - 1)
    return {
        "input_path": str(INPUT_PATH),
        "model": MODEL_NAME,
        "chunk_id": chunk_id,
        "chunk_size": CHUNK_SIZE,
        "start_index": start_index,
        "end_index": end_index,
        "total": total,
        "processed": 0,
        "results": [],
    }


def build_index_pos_map(results: list) -> dict:
    index_pos = {}
    for pos, item in enumerate(results):
        if isinstance(item, dict) and isinstance(item.get("index"), int):
            index_pos[item["index"]] = pos
    return index_pos


def save_chunk(payload: dict):
    chunk_id = payload["chunk_id"]
    path = get_chunk_path(chunk_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["results"].sort(
        key=lambda x: x.get("index", 10**12) if isinstance(x, dict) else 10**12
    )
    payload["processed"] = len(payload["results"])
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def load_existing_chunks(total: int):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chunk_store = {}
    global_index_map = {}
    pattern = re.compile(rf"^{re.escape(OUTPUT_FILE_PREFIX)}_part_(\d{{4}})\.json$")

    for path in sorted(OUTPUT_DIR.glob(f"{OUTPUT_FILE_PREFIX}_part_*.json")):
        m = pattern.match(path.name)
        if not m:
            continue
        chunk_id = int(m.group(1))
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"分片文件格式错误（应为 JSON 对象）: {path}")
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise ValueError(f"分片文件格式错误（results 应为数组）: {path}")
        payload["input_path"] = str(INPUT_PATH)
        payload["model"] = MODEL_NAME
        payload["total"] = total
        payload["chunk_size"] = CHUNK_SIZE
        payload["chunk_id"] = chunk_id
        payload["processed"] = len(results)
        index_pos = build_index_pos_map(results)
        for idx in index_pos:
            global_index_map[idx] = chunk_id
        chunk_store[chunk_id] = {"payload": payload, "index_pos": index_pos}

    return chunk_store, global_index_map


def get_or_create_chunk(chunk_store: dict, chunk_id: int, total: int):
    if chunk_id in chunk_store:
        return chunk_store[chunk_id]
    payload = init_chunk_payload(chunk_id, total)
    holder = {"payload": payload, "index_pos": {}}
    chunk_store[chunk_id] = holder
    return holder


def upsert_record(chunk_store: dict, global_index_map: dict, record: dict, total: int):
    index = record.get("index")
    if not isinstance(index, int):
        return None
    chunk_id = get_chunk_id(index)
    holder = get_or_create_chunk(chunk_store, chunk_id, total)
    results = holder["payload"]["results"]
    index_pos = holder["index_pos"]

    if index in index_pos:
        results[index_pos[index]] = record
    else:
        results.append(record)
        index_pos[index] = len(results) - 1
    global_index_map[index] = chunk_id
    return chunk_id


def process_one_case(i: int, item: dict, total: int) -> dict:
    case_text = (item.get("案情描述") or "").strip()
    if not case_text:
        return {
            "index": i,
            "案件编号": item.get("案件编号"),
            "案件类别": item.get("案件类别"),
            "status": "error",
            "error": "案情描述为空",
        }

    user_prompt = USER_PROMPT_TEMPLATE.replace("{case_text}", case_text)
    last_err = None
    raw_output = ""
    parsed_output = None

    for _ in range(MAX_RETRIES):
        try:
            raw_output = call_api(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=MODEL_NAME,
            )
            parsed_output = try_parse_json(raw_output)
            last_err = None
            break
        except Exception as e:
            last_err = str(e)
            time.sleep(RETRY_SLEEP_SECONDS)

    if last_err is None:
        print(f"[{i + 1}/{total}] 完成")
        return {
            "index": i,
            "案件编号": item.get("案件编号"),
            "案件类别": item.get("案件类别") or (parsed_output.get("fraud_type") if isinstance(parsed_output, dict) else None),
            "status": "ok",
            "result": parsed_output,
        }

    print(f"[{i + 1}/{total}] 失败: {last_err}")
    return {
        "index": i,
        "案件编号": item.get("案件编号"),
        "案件类别": item.get("案件类别"),
        "status": "error",
        "error": last_err,
        "raw_output": raw_output,
    }


def run_batch():
    if INPUT_PATH is None:
        raise ValueError("请通过 --input 指定 narrative 文件")
    cases = load_cases(INPUT_PATH)
    total = len(cases)
    chunk_store, global_index_map = load_existing_chunks(total)

    print(f"总样本数: {total}")
    print(f"已处理: {len(global_index_map)}，将从未处理样本继续（断点续跑）")
    print(f"并发线程数: {MAX_WORKERS}")
    print(f"分片大小: {CHUNK_SIZE}，输出目录: {OUTPUT_DIR}")

    pending_jobs = []
    for i, item in enumerate(cases):
        if i in global_index_map:
            continue
        if not isinstance(item, dict):
            record = {"index": i, "status": "error", "error": "样本结构不是 dict"}
            chunk_id = upsert_record(chunk_store, global_index_map, record, total)
            if chunk_id is not None:
                save_chunk(chunk_store[chunk_id]["payload"])
            continue
        pending_jobs.append((i, item))

    if not pending_jobs:
        print("没有待处理样本。")
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            in_flight = {}
            next_job_idx = 0

            while next_job_idx < len(pending_jobs) and len(in_flight) < MAX_WORKERS:
                i, item = pending_jobs[next_job_idx]
                future = executor.submit(process_one_case, i, item, total)
                in_flight[future] = i
                next_job_idx += 1

            while in_flight:
                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    _ = in_flight.pop(future)
                    record = future.result()
                    chunk_id = upsert_record(chunk_store, global_index_map, record, total)
                    if chunk_id is not None:
                        save_chunk(chunk_store[chunk_id]["payload"])

                    if next_job_idx < len(pending_jobs):
                        i, item = pending_jobs[next_job_idx]
                        new_future = executor.submit(process_one_case, i, item, total)
                        in_flight[new_future] = i
                        next_job_idx += 1

    # 最终保证所有分片有序且落盘
    for holder in chunk_store.values():
        save_chunk(holder["payload"])
    print(f"全部完成，结果目录: {OUTPUT_DIR}")


def main():
    run_batch()

