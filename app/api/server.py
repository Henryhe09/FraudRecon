from __future__ import annotations

import threading
from urllib.parse import urlencode
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.core.evaluator import EvaluationEngine
from app.data.dataset import load_cases
from app.utils import extract_json_obj


PROJECT_ROOT = Path(__file__).resolve().parents[2]
engine = EvaluationEngine(PROJECT_ROOT)
_run_threads: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="FraudRecon-Bench Eval Framework", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SingleEvalRequest(BaseModel):
    case_id: str
    model_id: str
    judge_model_id: str
    dataset_root: Optional[str] = None
    split: str = "test"
    input_mode: str = "text"
    eval_mode: Optional[str] = None
    strip_tone_hints: Optional[bool] = None


class BatchRunRequest(BaseModel):
    model_id: str
    judge_model_id: str
    dataset_root: Optional[str] = None
    split: str = "test"
    input_mode: str = "text"
    eval_mode: Optional[str] = None
    run_id: Optional[str] = None
    max_workers: Optional[int] = None
    dump_raw: Optional[bool] = None
    strip_tone_hints: Optional[bool] = None

@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/models")
def list_models():
    return {"models": list(engine.cfg.models.keys())}


@app.get("/api/runs")
def list_runs(limit: int = 100):
    rows = engine.store.conn.execute(
        "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?",
        (int(limit),),
    ).fetchall()
    data: List[Dict[str, Any]] = []
    for r in rows:
        row = dict(r)
        data.append(
            {
                "run_id": row.get("run_id", ""),
                "status": row.get("status", ""),
                "created_at": row.get("created_at", ""),
                "updated_at": row.get("updated_at", ""),
                "config": extract_json_obj(row.get("config_json") or "{}"),
                "summary": extract_json_obj(row.get("summary_json") or "{}"),
            }
        )
    return {"runs": data}


@app.get("/api/cases")
def list_cases(split: str = "test", dataset_root: Optional[str] = None):
    ds_root = engine.resolve_dataset_root(dataset_root=dataset_root, split=split)
    cases = load_cases(ds_root, split=split)
    return {"cases": [c.case_id for c in cases]}


@app.get("/api/audio")
def get_audio(case_id: str, split: str = "test", dataset_root: Optional[str] = None):
    ds_root = engine.resolve_dataset_root(dataset_root=dataset_root, split=split)
    cases = load_cases(ds_root, split=split)
    case = next((c for c in cases if c.case_id == case_id), None)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
    audio_abs = (ds_root / split / case.audio_relpath).resolve()
    if not audio_abs.exists():
        audio_abs = (ds_root / case.audio_relpath).resolve()
    if not audio_abs.exists():
        raise HTTPException(status_code=404, detail=f"Audio not found: {audio_abs}")
    return FileResponse(audio_abs)


@app.get("/api/runs/{run_id}/replay/{case_id}")
def get_replay_case(
    run_id: str,
    case_id: str,
    split: str = "test",
    dataset_root: Optional[str] = None,
):
    ds_root = engine.resolve_dataset_root(dataset_root=dataset_root, split=split)
    cases = load_cases(ds_root, split=split)
    case = next((c for c in cases if c.case_id == case_id), None)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")

    rows = engine.store.get_run_cases(run_id)
    row = next((x for x in rows if x.get("case_id") == case_id), None)
    if not row:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found in run {run_id}")

    query = urlencode({"case_id": case_id, "split": split})
    return {
        "run_id": run_id,
        "case_id": case_id,
        "split": split,
        "status": row.get("status", ""),
        "audio_url": f"/api/audio?{query}",
        "turns": case.turns,
        "gold_labels": case.labels,
        "round1_output": extract_json_obj(row.get("round1_json") or "{}"),
        "round1_metrics": extract_json_obj(row.get("round1_metrics_json") or "{}"),
        "t2_output": extract_json_obj(row.get("t2_json") or "{}"),
        "t2_metrics": extract_json_obj(row.get("t2_metrics_json") or "{}"),
        "cot": extract_json_obj(row.get("cot_json") or "{}"),
        "error": row.get("error", "") or "",
    }


@app.post("/api/evaluate/single")
def evaluate_single(req: SingleEvalRequest):
    ds_root = engine.resolve_dataset_root(dataset_root=req.dataset_root, split=req.split)
    cases = load_cases(ds_root, split=req.split)
    case = next((c for c in cases if c.case_id == req.case_id), None)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case not found: {req.case_id}")
    try:
        result = engine.evaluate_single_case(
            case=case,
            dataset_root=ds_root,
            model_id=req.model_id,
            judge_model_id=req.judge_model_id,
            input_mode=req.input_mode,
            eval_mode=req.eval_mode,
            strip_tone_hints=bool(req.strip_tone_hints),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def _run_batch_job(run_id: str, req: BatchRunRequest):
    try:
        _run_threads[run_id]["status"] = "running"
        result = engine.run_batch(
            run_id=run_id,
            model_id=req.model_id,
            judge_model_id=req.judge_model_id,
            dataset_root=req.dataset_root,
            split=req.split,
            input_mode=req.input_mode,
            eval_mode=req.eval_mode,
            max_workers=req.max_workers,
            dump_raw=req.dump_raw,
            strip_tone_hints=req.strip_tone_hints,
        )
        _run_threads[run_id]["status"] = result.get("status", "completed")
        _run_threads[run_id]["result"] = result
    except Exception as e:
        _run_threads[run_id]["status"] = "failed"
        _run_threads[run_id]["error"] = str(e)


@app.post("/api/runs/start")
def start_batch(req: BatchRunRequest):
    run_id = req.run_id or engine.make_run_id(
        model_id=req.model_id,
        split=req.split,
        input_mode=req.input_mode,
    )
    if run_id in _run_threads and _run_threads[run_id].get("status") == "running":
        raise HTTPException(status_code=409, detail=f"Run already active: {run_id}")
    _run_threads[run_id] = {"status": "queued", "result": None, "error": ""}
    t = threading.Thread(target=_run_batch_job, args=(run_id, req), daemon=True)
    _run_threads[run_id]["thread"] = t
    t.start()
    return {"run_id": run_id, "status": "queued"}


@app.get("/api/runs/{run_id}")
def get_run_status(run_id: str):
    in_mem = _run_threads.get(run_id, {})
    db_run = engine.store.get_run(run_id)
    if not in_mem and not db_run:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"run_id": run_id, "runtime": in_mem, "db": db_run}


@app.get("/api/runs/{run_id}/cases")
def get_run_cases(run_id: str):
    cases = engine.store.get_run_cases(run_id)
    if not cases:
        raise HTTPException(status_code=404, detail="Run not found or empty")
    return {"run_id": run_id, "cases": cases}
