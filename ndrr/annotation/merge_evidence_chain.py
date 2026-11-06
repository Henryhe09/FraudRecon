#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 CoT 结果回填到样本 labels.evidence_chain")
    parser.add_argument("--source", required=True, help="源样本文件（.json 或 .jsonl）")
    parser.add_argument("--annotations", "--cot", dest="cot", required=True, help="evidence_chain 输出的 JSONL")
    parser.add_argument("--output", required=True, help="输出文件路径（.json 或 .jsonl）")
    parser.add_argument(
        "--id-key",
        default="index",
        choices=["index", "case_id"],
        help="用哪个字段匹配 CoT 结果",
    )
    parser.add_argument(
        "--drop-unmatched",
        action="store_true",
        help="丢弃没有匹配到 CoT 的样本（默认保留）",
    )
    return parser.parse_args()


def load_source(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"source JSONL 每行必须是对象: line={lineno}")
                rows.append(item)
        return rows, "jsonl"

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return _assert_rows(obj), "json"
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        return _assert_rows(obj["results"]), "json_results"
    raise ValueError("source 仅支持 JSON 数组、JSONL，或包含 results 数组的 JSON 对象")


def _assert_rows(rows: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(rows):
        if not isinstance(item, dict):
            raise ValueError(f"source 样本必须是对象: index={i}")
        out.append(item)
    return out


def load_evidence_chain_map(path: Path, id_key: str) -> Dict[Any, Dict[str, Any]]:
    data: Dict[Any, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                continue
            if item.get("status") != "ok":
                continue
            key = item.get(id_key)
            if key is None:
                continue
            if not isinstance(item.get("evidence_chain"), dict):
                continue
            data[key] = item
    return data


def infer_case_id(sample: Dict[str, Any], index: int) -> str:
    value = sample.get("case_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    value = sample.get("simulated_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return f"idx_{index}"


def write_output(path: Path, rows: List[Dict[str, Any]], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    if mode == "json_results":
        obj = {"results": rows}
    else:
        obj = rows
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    source_rows, source_mode = load_source(Path(args.source))
    cot_map = load_evidence_chain_map(Path(args.cot), args.id_key)

    merged = 0
    unmatched = 0
    out_rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(source_rows):
        key = idx if args.id_key == "index" else infer_case_id(sample, idx)
        cot_item = cot_map.get(key)
        if cot_item is None:
            unmatched += 1
            if not args.drop_unmatched:
                out_rows.append(sample)
            continue

        labels = sample.get("labels")
        if not isinstance(labels, dict):
            labels = {}
            sample["labels"] = labels
        labels["evidence_chain"] = cot_item["evidence_chain"]
        if "is_fraud" not in labels and isinstance(cot_item.get("is_fraud"), bool):
            labels["is_fraud"] = cot_item["is_fraud"]

        out_rows.append(sample)
        merged += 1

    write_output(Path(args.output), out_rows, source_mode)
    print(f"总样本: {len(source_rows)}")
    print(f"成功回填: {merged}")
    print(f"未匹配: {unmatched}")
    print(f"输出文件: {args.output}")


if __name__ == "__main__":
    main()
