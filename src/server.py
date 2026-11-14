"""FastAPI backend for the Mini-AutoML web UI.

Serves the static front-end (src/static) and a small JSON API:

    GET  /api/health                       liveness probe
    GET  /api/dataset                      profile of the current dataset (404 if none)
    POST /api/dataset                      upload a CSV, profile its columns
    GET  /api/models                       the models train.py can train
    POST /api/jobs        {"target": ..., "models": [...], "exclude": [...]}
                                           start preprocess.py (minus excluded columns), then
                                           train.py once per model, in the background
                                           (several models = comparison)
    GET  /api/jobs/current                 the job that is running right now (or null)
    GET  /api/jobs/{job_id}                status, per-step logs and resulting run ids
    GET  /api/runs                         every row of outputs/experiments_log.csv
    GET  /api/runs/{run_id}                one run: metrics, threshold curve, files
    GET  /api/runs/{run_id}/files/{name}   a plot, the model or run_metrics.json
    GET  /api/experiments_log.csv          the raw DIY experiment log

The ML code is never imported here (except the shared CSV reader and the model registry's
names, which import no ML library): pipeline stages run as subprocesses, exactly like they
would from the command line.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from models import DEFAULT_MODEL, MODELS
from preprocess import read_dataset

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
SRC_DIR = Path(__file__).resolve().parent
# AUTOML_ROOT lets tests point data/ and outputs/ somewhere else.
ROOT = Path(os.getenv("AUTOML_ROOT", SRC_DIR.parent)).resolve()
RAW_DIR = ROOT / "data" / "raw"
DATASET_PATH = RAW_DIR / "dataset.csv"
DATASET_INFO_PATH = RAW_DIR / "dataset_info.json"
OUTPUT_DIR = ROOT / "outputs"
LOG_PATH = OUTPUT_DIR / "experiments_log.csv"
STATIC_DIR = SRC_DIR / "static"

STEP_TIMEOUT_S = 1800
ALLOWED_EXTENSIONS = {".csv", ".tsv", ".txt"}
RUN_ID_RE = re.compile(r"^run_\d{8}_\d{6}$")
MODEL_MEDIA_TYPES = {".json": "application/json", ".txt": "text/plain"}
RUN_FILES = {
    "confusion_matrix.png": "image/png",
    "roc_curve.png": "image/png",
    "pr_curve.png": "image/png",
    "shap_summary.png": "image/png",
    "run_metrics.json": "application/json",
    # one model file per model type, e.g. xgboost_model.json, catboost_model.cbm
    **{spec.filename: MODEL_MEDIA_TYPES.get(Path(spec.filename).suffix, "application/octet-stream")
       for spec in MODELS.values()},
}
LEGACY_MODEL_LABEL = "XGBoost"  # log rows from before the Model column were XGBoost runs
NUMERIC_LOG_FIELDS = {
    "Max_Depth": int, "N_Estimators": int, "Learning_Rate": float, "Scale_Pos_Weight": float,
    "Best_Threshold": float, "F1_Score": float, "Precision": float, "Recall": float, "PR_AUC": float,
}

app = FastAPI(title="Mini-AutoML", docs_url="/api/docs", redoc_url=None)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Dataset profiling (powers the column cards in the UI)
# --------------------------------------------------------------------------- #
def profile_column(series: pd.Series, n_rows: int) -> dict:
    text = series.astype("string").str.strip()
    missing = series.isna() | (text == "")
    values = series[~missing]
    as_number = pd.to_numeric(values, errors="coerce")
    is_numeric = (
        len(values) > 0
        and not pd.api.types.is_bool_dtype(series)
        and as_number.notna().all()
    )
    n_unique = int(values.astype(str).str.strip().nunique())

    info = {
        "name": str(series.name),
        "kind": "numeric" if is_numeric else "text",
        "n_missing": int(missing.sum()),
        "n_unique": n_unique,
        "binary": n_unique == 2,
        "id_like": n_unique == n_rows and n_rows > 1 and not missing.any()
                   and not (is_numeric and pd.api.types.is_float_dtype(as_number)
                            and not np.all(np.mod(as_number, 1) == 0)),
    }

    finite = as_number[np.isfinite(as_number)] if is_numeric else None
    if is_numeric and n_unique > 10 and len(finite):
        counts, edges = np.histogram(finite.astype(float), bins=12)
        info["histogram"] = {
            "counts": counts.astype(int).tolist(),
            "min": float(edges[0]), "max": float(edges[-1]), "mean": float(finite.mean()),
        }
    else:
        counts = values.astype(str).str.strip().value_counts()
        info["top_values"] = [{"value": str(k), "count": int(v)} for k, v in counts.head(3).items()]
        info["other_count"] = int(counts.iloc[3:].sum())
    return info


def profile_dataset(path: Path, filename: str) -> dict:
    df = read_dataset(path)
    if df.empty:
        raise ValueError("the file has a header but no data rows")
    if df.shape[1] < 2:
        raise ValueError("only one column was detected; is this a delimited text file?")
    return {
        "filename": filename,
        "uploaded_at": now_iso(),
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "separator": df.attrs.get("separator", ","),
        "encoding": df.attrs.get("encoding", "utf-8"),
        "columns": [profile_column(df[c], len(df)) for c in df.columns],
    }


def load_dataset_info() -> dict | None:
    if not (DATASET_PATH.exists() and DATASET_INFO_PATH.exists()):
        return None
    try:
        return json.loads(DATASET_INFO_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# --------------------------------------------------------------------------- #
# Background pipeline jobs (one at a time; state lives in memory)
# --------------------------------------------------------------------------- #
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
CURRENT_JOB_ID: str | None = None


def snapshot(job: dict) -> dict:
    with JOBS_LOCK:
        return json.loads(json.dumps(job))


def last_error_line(output: str) -> str:
    lines = [ln for ln in output.strip().splitlines() if ln.strip()]
    errors = [ln for ln in lines if "ERROR" in ln or "Error" in ln]
    line = (errors or lines or ["unknown error"])[-1]
    return line.split("| ")[-1].strip()


def build_steps(models: list[str]) -> list[dict]:
    """One preprocessing step, then one training step per model (all on the same split)."""
    steps = [{"key": "preprocess", "model": None, "label": "Validate and preprocess the data"}]
    steps += [{"key": f"train_{key}", "model": key,
               "label": f"Train {MODELS[key].label}, tune the threshold, explain the model"}
              for key in models]
    for step in steps:
        step.update({"status": "waiting", "log": "", "run_id": None, "started_at": None, "finished_at": None})
    return steps


def step_command(step: dict, target: str, exclude: list[str]) -> list[str]:
    if step["model"] is None:
        # "--exclude=NAME" (one token) so column names starting with '-' are not read as options
        return [sys.executable, str(SRC_DIR / "preprocess.py"), "--target", target,
                *[f"--exclude={name}" for name in exclude]]
    return [sys.executable, str(SRC_DIR / "train.py"), "--target", target, "--model", step["model"]]


def run_job(job_id: str, target: str) -> None:
    """Run the steps in order and stop at the first failure.

    Runs that finished before a failure stay in the experiment log and in job["run_ids"].
    """
    global CURRENT_JOB_ID
    job = JOBS[job_id]
    try:
        for step in job["steps"]:
            with JOBS_LOCK:
                step["status"], step["started_at"] = "running", now_iso()
            try:
                result = subprocess.run(
                    step_command(step, target, job["exclude"]),
                    cwd=ROOT, capture_output=True, text=True, timeout=STEP_TIMEOUT_S,
                )
                output = (result.stdout + result.stderr).strip()
                failed = result.returncode != 0
                error = last_error_line(output) if failed else None
            except subprocess.TimeoutExpired:
                output, failed = "", True
                error = f"step timed out after {STEP_TIMEOUT_S // 60} minutes"

            found = re.findall(r"run_\d{8}_\d{6}", output) if step["model"] and not failed else []
            with JOBS_LOCK:
                step["log"], step["finished_at"] = output, now_iso()
                step["status"] = "failed" if failed else "done"
                if found:
                    step["run_id"] = found[-1]
                    job["run_ids"].append(found[-1])
                    job["run_id"] = found[-1]
                if failed:
                    job["status"], job["error"] = "failed", error
            if failed:
                return

        with JOBS_LOCK:
            job["status"] = "succeeded"
    except Exception as exc:  # noqa: BLE001 - surface anything unexpected to the UI
        with JOBS_LOCK:
            job["status"], job["error"] = "failed", f"internal error: {exc}"
    finally:
        with JOBS_LOCK:
            job["finished_at"] = now_iso()
            CURRENT_JOB_ID = None


# --------------------------------------------------------------------------- #
# Run history (reads the DIY logger's files)
# --------------------------------------------------------------------------- #
def png_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


def read_run_metrics(run_id: str) -> dict:
    path = OUTPUT_DIR / run_id / "run_metrics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def read_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    try:
        with LOG_PATH.open(newline="") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error) as exc:
        raise HTTPException(500, f"could not read experiments_log.csv: {exc}") from exc

    runs = []
    for row in rows:
        run = {"run_id": row.get("Timestamp", ""), "target": row.get("Target_Column", ""),
               "model_label": row.get("Model") or LEGACY_MODEL_LABEL}
        for field, cast in NUMERIC_LOG_FIELDS.items():
            try:
                run[field.lower()] = cast(float(row[field]))
            except (KeyError, TypeError, ValueError):
                run[field.lower()] = None
        metrics = read_run_metrics(run["run_id"])
        run["accuracy"] = metrics.get("metrics", {}).get("accuracy")
        run["positive_class"] = metrics.get("positive_class")
        runs.append(run)
    return runs


def run_or_404(run_id: str) -> dict:
    if not RUN_ID_RE.match(run_id):
        raise HTTPException(404, "run not found")
    for run in read_log():
        if run["run_id"] == run_id:
            return run
    raise HTTPException(404, "run not found")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
class JobRequest(BaseModel):
    target: str
    models: list[str] = [DEFAULT_MODEL]
    exclude: list[str] = []  # columns to leave out, e.g. ones only known after the outcome


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/models")
def list_models() -> list[dict]:
    return [{"key": spec.key, "label": spec.label, "default": spec.key == DEFAULT_MODEL}
            for spec in MODELS.values()]


@app.get("/api/dataset")
def get_dataset() -> dict:
    info = load_dataset_info()
    if info is None:
        raise HTTPException(404, "no dataset uploaded yet")
    return info


@app.post("/api/dataset")
def upload_dataset(file: UploadFile = File(...)) -> dict:
    if CURRENT_JOB_ID is not None:
        raise HTTPException(409, "a training job is running; wait for it to finish first")
    name = Path(file.filename or "dataset.csv").name
    if Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, "upload a .csv, .tsv or .txt file")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = RAW_DIR / "upload.tmp"
    with tmp_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    try:
        info = profile_dataset(tmp_path, name)
    except Exception as exc:  # noqa: BLE001 - any parse problem is a bad upload
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(400, f"could not read {name}: {exc}") from exc

    tmp_path.replace(DATASET_PATH)
    DATASET_INFO_PATH.write_text(json.dumps(info))
    return info


@app.post("/api/jobs", status_code=202)
def start_job(req: JobRequest) -> dict:
    global CURRENT_JOB_ID
    info = load_dataset_info()
    if info is None:
        raise HTTPException(400, "upload a dataset first")
    if req.target not in {c["name"] for c in info["columns"]}:
        raise HTTPException(400, f"column '{req.target}' is not in the dataset")
    unknown = [m for m in req.models if m not in MODELS]
    if unknown or not req.models:
        raise HTTPException(400, f"choose at least one model from: {', '.join(MODELS)}"
                                 + (f" (unknown: {', '.join(unknown)})" if unknown else ""))
    models = list(dict.fromkeys(req.models))  # drop duplicates, keep order
    columns = {c["name"] for c in info["columns"]}
    exclude = list(dict.fromkeys(req.exclude))
    if req.target in exclude:
        raise HTTPException(400, "the target column cannot also be excluded")
    unknown_cols = [c for c in exclude if c not in columns]
    if unknown_cols:
        raise HTTPException(400, f"columns to exclude are not in the dataset: {', '.join(unknown_cols)}")
    if len(columns) - len(exclude) < 2:
        raise HTTPException(400, "keep at least one feature column besides the target")

    with JOBS_LOCK:
        if CURRENT_JOB_ID is not None:
            raise HTTPException(409, "another training job is already running")
        job_id = uuid.uuid4().hex[:12]
        job = {
            "id": job_id, "target": req.target, "dataset": info["filename"], "models": models, "exclude": exclude,
            "status": "running", "error": None, "run_id": None, "run_ids": [],
            "started_at": now_iso(), "finished_at": None,
            "steps": build_steps(models),
        }
        JOBS[job_id] = job
        CURRENT_JOB_ID = job_id
    threading.Thread(target=run_job, args=(job_id, req.target), daemon=True).start()
    return snapshot(job)


@app.get("/api/jobs/current")
def current_job() -> dict:
    job_id = CURRENT_JOB_ID
    return {"job": snapshot(JOBS[job_id]) if job_id else None}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    if job_id not in JOBS:
        raise HTTPException(404, "job not found (the server may have restarted)")
    return snapshot(JOBS[job_id])


@app.get("/api/runs")
def list_runs() -> list[dict]:
    return list(reversed(read_log()))  # newest first


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    run = run_or_404(run_id)
    metrics = read_run_metrics(run_id)
    files = {}
    for name in RUN_FILES:
        path = OUTPUT_DIR / run_id / name
        if path.exists():
            entry = {"size": path.stat().st_size}
            if name.endswith(".png"):
                dims = png_size(path)
                if dims:
                    entry["width"], entry["height"] = dims
            files[name] = entry
    return {
        **run,
        "model": metrics.get("model", "xgboost"),
        "model_file": metrics.get("model_file", MODELS[DEFAULT_MODEL].filename),
        "hyperparameters": metrics.get("hyperparameters"),
        "excluded_columns": metrics.get("excluded_columns", []),
        "threshold_source": metrics.get("threshold_source"),  # None for runs made before CV thresholds
        "negative_class": metrics.get("negative_class"),
        "n_train": metrics.get("n_train"),
        "n_test": metrics.get("n_test"),
        "n_features": metrics.get("n_features"),
        "threshold_curve": metrics.get("threshold_curve", []),
        "files": files,
    }


@app.get("/api/runs/{run_id}/files/{name}")
def get_run_file(run_id: str, name: str, download: bool = False) -> FileResponse:
    if not RUN_ID_RE.match(run_id) or name not in RUN_FILES:
        raise HTTPException(404, "file not found")
    path = OUTPUT_DIR / run_id / name
    if not path.exists():
        raise HTTPException(404, "file not found for this run")
    return FileResponse(path, media_type=RUN_FILES[name],
                        filename=f"{run_id}_{name}" if download else None)


@app.get("/api/experiments_log.csv")
def get_log_file() -> FileResponse:
    if not LOG_PATH.exists():
        raise HTTPException(404, "no runs logged yet")
    return FileResponse(LOG_PATH, media_type="text/csv", filename="experiments_log.csv")


# --------------------------------------------------------------------------- #
# Front-end
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
