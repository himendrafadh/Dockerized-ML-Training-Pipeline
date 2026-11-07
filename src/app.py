"""AutoML Platform: Streamlit dashboard that drives the preprocess -> train pipeline.

Layout
    Sidebar : upload CSV -> pick target -> Train model
    Main    : welcome screen | live pipeline progress | results dashboard
              (KPI row, tabbed plots, downloads, experiment history)

The UI never imports the ML training code; it calls the CLI scripts via
subprocess, so the web layer stays decoupled from the pipeline. The only shared
helper is the CSV reader, so the UI and the pipeline see identical columns.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from preprocess import read_dataset  # shared reader (Streamlit puts src/ on sys.path)

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent  # /app inside Docker
RAW_PATH = ROOT / "data" / "raw" / "dataset.csv"
OUTPUT_DIR = ROOT / "outputs"
LOG_PATH = OUTPUT_DIR / "experiments_log.csv"
PIPELINE_TIMEOUT_S = 1800
PREVIEW_ROWS = 200

STEPS = [
    ("Validating & preprocessing data", "src/preprocess.py"),
    ("Training XGBoost, tuning threshold & generating plots", "src/train.py"),
]
SEPARATOR_NAMES = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}

st.set_page_config(page_title="AutoML Platform", layout="wide")

# Light-touch polish on top of .streamlit/config.toml: tighter page padding,
# quieter metric labels, and a visible keyboard focus ring in the accent color.
st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; padding-bottom: 4rem; max-width: 1200px; }
    h1 { font-size: 1.75rem; font-weight: 650; letter-spacing: -0.01em; }
    h3 { font-weight: 600; margin-top: 1.5rem; }
    [data-testid="stMetricLabel"] p { font-size: 0.8rem; color: #4B524F; }
    [data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 600; }
    [data-testid="stSidebar"] { border-right: 1px solid #D9DCD8; }
    button:focus-visible, [role="tab"]:focus-visible, a:focus-visible {
        outline: 2px solid #0F766E !important; outline-offset: 2px;
    }
    button, [role="tab"], summary, label { cursor: pointer; }
    button:disabled { cursor: not-allowed; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def fmt(value, digits: int = 3) -> str:
    """Format a number for display; anything missing/invalid becomes a dash."""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def save_upload(uploaded) -> None:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_bytes(uploaded.getvalue())


def run_step(script: str, target: str) -> subprocess.CompletedProcess:
    """Run one pipeline stage; argument list (no shell) so odd column names are safe."""
    return subprocess.run(
        [sys.executable, script, "--target", target],
        cwd=ROOT, capture_output=True, text=True, timeout=PIPELINE_TIMEOUT_S,
    )


def last_error_line(result: subprocess.CompletedProcess) -> str:
    """Pull the most useful one-line error message out of a failed step's output."""
    lines = (result.stdout + "\n" + result.stderr).strip().splitlines()
    errors = [ln for ln in lines if "ERROR" in ln or "Error" in ln]
    line = (errors or lines or ["unknown error"])[-1]
    return line.split("| ")[-1].strip()


def show_image(path: Path | None, missing_label: str) -> None:
    """Display a plot, or a friendly placeholder if it was not generated."""
    if path is not None and path.exists():
        st.image(str(path), width="stretch")
    else:
        st.warning(f"{missing_label} is not available for this run.")


def load_latest_run() -> tuple[pd.Series, pd.DataFrame, Path | None, dict] | None:
    """Return (last log row, full log, run folder, run_metrics.json) or None."""
    if not LOG_PATH.exists():
        return None
    try:
        log = pd.read_csv(LOG_PATH)
    except Exception as exc:  # corrupted / partially written log
        st.error(f"Could not read `outputs/experiments_log.csv`: {exc}")
        return None
    if log.empty:
        return None

    last = log.iloc[-1]
    run_dir = OUTPUT_DIR / str(last["Timestamp"])
    if not run_dir.is_dir():  # fall back to the newest timestamped folder
        runs = sorted(p for p in OUTPUT_DIR.glob("run_*") if p.is_dir())
        run_dir = runs[-1] if runs else None

    details: dict = {}
    if run_dir is not None and (run_dir / "run_metrics.json").exists():
        try:
            details = json.loads((run_dir / "run_metrics.json").read_text())
        except (json.JSONDecodeError, OSError):
            details = {}
    return last, log, run_dir, details


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> tuple[str | None, bool, pd.DataFrame | None]:
    """Upload + target selection + run button. Returns (target, clicked, preview)."""
    with st.sidebar:
        st.subheader("AutoML Platform")
        st.caption(
            "Upload any binary-classification dataset, choose what to predict, and get an "
            "imbalance-aware, explainable XGBoost model in one click."
        )
        st.divider()

        uploaded = st.file_uploader("1. Upload dataset", type=["csv", "tsv", "txt"],
                                    help="Separator (, ; tab |) and encoding are detected automatically.")
        if uploaded is None:
            st.caption("No dataset uploaded yet.")
            return None, False, None

        save_upload(uploaded)
        try:
            preview = read_dataset(RAW_PATH, nrows=PREVIEW_ROWS)
            if preview.shape[1] < 2:
                raise ValueError("only 1 column detected - is this a valid delimited file?")
        except Exception as exc:  # malformed / empty file
            st.error(f"Could not read the file: {exc}")
            return None, False, None

        sep = SEPARATOR_NAMES.get(preview.attrs.get("separator"), "unknown")
        st.success(f"**{uploaded.name}**  \n{preview.shape[1]} columns · {sep}-separated")

        columns = list(preview.columns)
        target = st.selectbox("2. Target column to predict", columns,
                              index=len(columns) - 1)

        # Early, non-blocking hint: the pipeline itself performs the real validation.
        n_classes = preview[target].replace(r"^\s*$", pd.NA, regex=True).nunique(dropna=True)
        # Only >2 is conclusive: a sorted file can show a single class in the first rows.
        if n_classes > 2:
            st.warning(f"`{target}` already has {n_classes} distinct values in the first "
                       f"{PREVIEW_ROWS} rows. Binary classification needs exactly 2.")

        clicked = st.button("Train model", type="primary", width="stretch")
        st.divider()
        st.caption("Pipeline: `preprocess.py`, then `train.py`, then the experiment logger.")
        return target, clicked, preview


# --------------------------------------------------------------------------- #
# Main-area sections
# --------------------------------------------------------------------------- #
def render_welcome() -> None:
    st.markdown("#### Upload a CSV in the sidebar to begin")
    st.markdown(
        "1. **Upload** a binary-classification CSV. Separator and encoding are detected "
        "automatically.\n"
        "2. **Pick the target column.** It must contain exactly two classes "
        "(e.g. Yes/No, 1/0, M/B).\n"
        "3. **Train.** The pipeline cleans the data, trains XGBoost, tunes the decision "
        "threshold for maximum F1 and explains the model with SHAP."
    )

    with st.expander("What happens under the hood?"):
        st.markdown(
            "- **Automatic cleaning:** ID-like, constant and high-cardinality columns are dropped; "
            "blank `\" \"` cells become missing values; features that leak the target are removed.\n"
            "- **Imputation & encoding:** median for numbers, mode for categories, one-hot encoding.\n"
            "- **Imbalance handling:** `scale_pos_weight` is computed from the class ratio.\n"
            "- **Threshold tuning:** thresholds 0.10–0.90 are scanned; the F1-optimal one is used.\n"
            "- **DIY experiment tracking:** every run gets its own `outputs/run_<timestamp>/` folder "
            "and a row in `outputs/experiments_log.csv`. No MLflow needed."
        )


def run_pipeline(target: str) -> bool:
    """Execute every stage with live progress feedback. Returns True on success."""
    progress = st.progress(0.0, text="Starting pipeline...")
    logs: list[tuple[str, str]] = []
    ok = True

    with st.status("Training pipeline...", expanded=True) as status:
        for i, (label, script) in enumerate(STEPS, start=1):
            st.write(f"**Step {i}/{len(STEPS)}:** {label}…")
            progress.progress((i - 1) / len(STEPS), text=f"Step {i}/{len(STEPS)}: {label}…")
            try:
                result = run_step(script, target)
            except subprocess.TimeoutExpired:
                st.error(f"Step {i} timed out after {PIPELINE_TIMEOUT_S // 60} minutes.")
                ok = False
                break
            logs.append((label, (result.stdout + result.stderr).strip() or "(no output)"))
            if result.returncode != 0:
                st.error(f"**Step {i} failed:** {last_error_line(result)}")
                ok = False
                break

        if ok:
            progress.progress(1.0, text="Pipeline completed")
            status.update(label="Pipeline completed successfully", state="complete", expanded=False)
        else:
            progress.empty()
            status.update(label="Pipeline failed (see details below)", state="error", expanded=True)

    # Expanders cannot be nested inside st.status, so the full logs go below it.
    with st.expander("Pipeline logs", expanded=not ok):
        for label, text in logs:
            st.markdown(f"**{label}**")
            st.code(text, language="text")
    if ok:
        st.toast("Model trained successfully.")
    return ok


def render_dataset_preview(preview: pd.DataFrame, target: str | None) -> None:
    with st.expander(f"Dataset preview (first {min(len(preview), 10)} rows)", expanded=False):
        st.dataframe(preview.head(10), width="stretch", hide_index=True)
        if target:
            st.caption(f"Selected target: **{target}**")


def render_dashboard() -> None:
    latest = load_latest_run()
    if latest is None:
        return
    last, log, run_dir, details = latest
    m = details.get("metrics", {})

    st.markdown("### Results")
    meta_bits = [f"Run `{last['Timestamp']}`", f"target **{last['Target_Column']}**"]
    if "positive_class" in details:
        meta_bits.append(f"positive class (1): **{details['positive_class']}**")
    if "n_test" in details:
        meta_bits.append(f"evaluated on **{details['n_test']:,}** hold-out rows")
    st.caption(" · ".join(meta_bits))

    # ---- KPI row ----
    kpis = [
        ("Accuracy", fmt(m.get("accuracy")), "Share of hold-out rows classified correctly."),
        ("F1-Score", fmt(last["F1_Score"]), "Harmonic mean of precision and recall (optimised)."),
        ("Precision", fmt(last["Precision"]), "Of the rows predicted positive, how many were."),
        ("Recall", fmt(last["Recall"]), "Of the truly positive rows, how many were found."),
        ("PR-AUC", fmt(last["PR_AUC"]), "Area under the precision-recall curve (threshold-free)."),
        ("Best Threshold", fmt(last["Best_Threshold"], 2), "Probability cut-off that maximises F1."),
    ]
    for col, (label, value, help_text) in zip(st.columns(len(kpis)), kpis):
        col.metric(label, value, help=help_text, border=True)

    st.caption(
        f"Hyperparameters: max_depth={last['Max_Depth']} · learning_rate={last['Learning_Rate']} · "
        f"n_estimators={last['N_Estimators']} · scale_pos_weight={fmt(last['Scale_Pos_Weight'])}"
    )

    # ---- Plots ----
    def plot(name: str) -> Path | None:
        return run_dir / name if run_dir is not None else None

    tab_cm, tab_pr, tab_shap, tab_hist = st.tabs([
        "Confusion matrix & ROC", "Precision-recall",
        "Explainability (SHAP)", "Experiment history",
    ])
    with tab_cm:
        left, right = st.columns(2)
        with left:
            st.markdown("**Confusion matrix**")
            show_image(plot("confusion_matrix.png"), "The confusion matrix")
            st.caption("Counts of correct and incorrect predictions at the optimal threshold.")
        with right:
            st.markdown("**ROC curve**")
            show_image(plot("roc_curve.png"), "The ROC curve")
            st.caption("True-positive vs. false-positive rate across all thresholds; "
                       "the red dot is the chosen operating point.")
    with tab_pr:
        left, right = st.columns([3, 2])
        with left:
            show_image(plot("pr_curve.png"), "The precision-recall curve")
        with right:
            st.markdown("**Reading this chart**")
            st.caption(
                "For imbalanced problems the precision-recall curve is more informative than ROC. "
                "Moving the threshold trades precision (fewer false alarms) for recall (fewer "
                "missed positives). The red dot marks the F1-optimal threshold "
                f"(**{fmt(last['Best_Threshold'], 2)}**), and the dashed line is the score of a "
                "random classifier."
            )
            st.metric("PR-AUC", fmt(last["PR_AUC"]), border=True)
    with tab_shap:
        show_image(plot("shap_summary.png"), "The SHAP summary plot")
        st.caption(
            "**SHAP (SHapley Additive exPlanations)** attributes each prediction to the input "
            "features. Features are ranked by overall impact; each dot is one row. Its horizontal "
            "position shows whether that feature pushed the prediction towards the positive class "
            "(right) or away from it (left), and its colour shows the feature value (red = high, "
            "blue = low)."
        )
    with tab_hist:
        st.dataframe(log.iloc[::-1], width="stretch", hide_index=True)
        st.caption(f"{len(log)} run(s) recorded in `outputs/experiments_log.csv`.")

    # ---- Downloads ----
    st.markdown("### Downloads")
    files = [
        ("Trained model", "xgboost_model.json", "application/json", plot("xgboost_model.json")),
        ("Run metrics", "run_metrics.json", "application/json", plot("run_metrics.json")),
        ("Experiment log", "experiments_log.csv", "text/csv", LOG_PATH),
    ]
    for col, (label, filename, mime, path) in zip(st.columns(len(files)), files):
        available = path is not None and path.exists()
        col.download_button(
            f"Download {label.lower()}", data=path.read_bytes() if available else b"",
            file_name=filename, mime=mime, disabled=not available, width="stretch",
            help=f"`{filename}`" if available else "Not available for this run",
        )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
target, run_clicked, preview = render_sidebar()

st.title("Mini-AutoML")
st.caption("Universal binary classification · F1-optimised threshold · explainable AI · "
           "DIY experiment tracking")

if preview is None:
    render_welcome()
else:
    render_dataset_preview(preview, target)
    if run_clicked and target:
        run_pipeline(target)
    elif not LOG_PATH.exists():
        st.info("Select the target column, then click **Train model**.")

if LOG_PATH.exists():
    st.divider()
    render_dashboard()
