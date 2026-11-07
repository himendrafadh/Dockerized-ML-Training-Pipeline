"""Mini-AutoML: Streamlit front-end that drives the preprocess -> train pipeline.

Flow: upload CSV -> pick target column -> Run Pipeline -> view metrics + SHAP.
The UI never imports the ML code; it calls the CLI scripts via subprocess so the
web layer stays decoupled from the pipeline.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from preprocess import read_dataset  # shared reader (Streamlit puts src/ on sys.path)

ROOT = Path(__file__).resolve().parent.parent  # /app inside Docker
RAW_PATH = ROOT / "data" / "raw" / "dataset.csv"
META_PATH = ROOT / "data" / "processed" / "metadata.json"
OUTPUT_DIR = ROOT / "outputs"
LOG_PATH = OUTPUT_DIR / "experiments_log.csv"
PIPELINE_TIMEOUT_S = 1800


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def save_upload(uploaded) -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_bytes(uploaded.getvalue())


def run_step(label: str, script: str, target: str) -> subprocess.CompletedProcess:
    """Run one pipeline stage; argument list (no shell) so odd column names are safe."""
    return subprocess.run(
        [sys.executable, script, "--target", target],
        cwd=ROOT, capture_output=True, text=True, timeout=PIPELINE_TIMEOUT_S,
    )


def latest_run_dir(preferred: str | None = None) -> Path | None:
    if preferred and (OUTPUT_DIR / preferred).is_dir():
        return OUTPUT_DIR / preferred
    runs = sorted(p for p in OUTPUT_DIR.glob("run_*") if p.is_dir())  # names sort chronologically
    return runs[-1] if runs else None


def show_results() -> None:
    """Render the LAST row of experiments_log.csv and the matching SHAP plot."""
    if not LOG_PATH.exists():
        return
    log = pd.read_csv(LOG_PATH)
    if log.empty:
        return
    last = log.iloc[-1]

    st.subheader("Latest run results")
    st.caption(f"Run `{last['Timestamp']}` · target column: **{last['Target_Column']}**")

    cols = st.columns(5)
    cols[0].metric("F1 score", f"{last['F1_Score']:.3f}")
    cols[1].metric("Precision", f"{last['Precision']:.3f}")
    cols[2].metric("Recall", f"{last['Recall']:.3f}")
    cols[3].metric("PR-AUC", f"{last['PR_AUC']:.3f}")
    cols[4].metric("Best threshold", f"{last['Best_Threshold']:.2f}")
    st.caption(
        f"max_depth={last['Max_Depth']} · learning_rate={last['Learning_Rate']} · "
        f"n_estimators={last['N_Estimators']} · scale_pos_weight={last['Scale_Pos_Weight']:.3f}"
    )

    run_dir = latest_run_dir(str(last["Timestamp"]))
    if run_dir:
        shap_png = run_dir / "shap_summary.png"
        if shap_png.exists():
            st.subheader("SHAP feature importance")
            st.image(str(shap_png), width="stretch")
        model_file = run_dir / "xgboost_model.json"
        if model_file.exists():
            st.download_button("Download trained model (xgboost_model.json)",
                               data=model_file.read_bytes(), file_name="xgboost_model.json",
                               mime="application/json")

    with st.expander("Experiment history (outputs/experiments_log.csv)"):
        st.dataframe(log.iloc[::-1], width="stretch", hide_index=True)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Mini-AutoML: XGBoost Pipeline", page_icon="🤖", layout="wide")
st.title("Mini-AutoML: XGBoost Pipeline")
st.write("Upload any binary-classification CSV, choose the column to predict, and train an "
         "imbalance-aware XGBoost model with an F1-optimised threshold and SHAP explanations.")

uploaded = st.file_uploader("Upload a CSV dataset", type=["csv"])

target = None
if uploaded is not None:
    save_upload(uploaded)
    try:
        preview = read_dataset(RAW_PATH, nrows=5)  # auto-detects separator/encoding
        if preview.shape[1] < 2:
            raise ValueError("only 1 column detected - is this a valid delimited file?")
    except Exception as exc:  # malformed / empty file
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
    columns = [str(c).strip() for c in preview.columns]
    sep_name = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}[preview.attrs["separator"]]
    st.success(f"Saved to `data/raw/dataset.csv` · {len(columns)} columns · "
               f"detected separator: {sep_name}")
    st.dataframe(preview, width="stretch", hide_index=True)

    target = st.selectbox("Select the Target Column to Predict", columns, index=len(columns) - 1)

    if st.button("Run Pipeline", type="primary"):
        steps = [("1/2 Preprocessing", "src/preprocess.py"), ("2/2 Training", "src/train.py")]
        failed = False
        for label, script in steps:
            with st.spinner(f"{label}..."):
                try:
                    result = run_step(label, script, target)
                except subprocess.TimeoutExpired:
                    st.error(f"{label} timed out after {PIPELINE_TIMEOUT_S}s")
                    failed = True
                    break
            with st.expander(f"{label} logs", expanded=result.returncode != 0):
                st.code((result.stdout + result.stderr).strip() or "(no output)", language="text")
            if result.returncode != 0:
                st.error(f"{label} failed (exit code {result.returncode}). See logs above.")
                failed = True
                break
        if not failed:
            st.success("Pipeline finished successfully.")
            if META_PATH.exists():
                meta = json.loads(META_PATH.read_text())
                st.info(f"Target encoding: **{meta['positive_class']}** = 1 (positive), "
                        f"**{meta['negative_class']}** = 0 · {meta['n_features']} features used · "
                        f"{len(meta['dropped_columns'])} column(s) auto-dropped")
else:
    st.info("Upload a CSV to get started.")

show_results()
