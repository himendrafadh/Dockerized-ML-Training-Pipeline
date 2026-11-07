"""Stage 2: XGBoost training with a custom (file-based) experiment logger.

No external tracking server is used. Every run:
    * creates   outputs/run_YYYYMMDD_HHMMSS/
    * saves     shap_summary.png, xgboost_model.json, run_metrics.json inside it
    * appends a row to the global  outputs/experiments_log.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend (Docker / Streamlit server)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger("train")

LOG_HEADER = [
    "Timestamp", "Target_Column", "Max_Depth", "Learning_Rate", "N_Estimators",
    "Scale_Pos_Weight", "Best_Threshold", "F1_Score", "Precision", "Recall", "PR_AUC",
]


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_split(path: Path, target: str) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run src/preprocess.py first")
    df = pd.read_parquet(path)
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in {path}; was preprocess run with the same target?")
    X, y = df.drop(columns=[target]), df[target].astype(int)
    logger.info("Loaded %s: X=%s, positives=%d", path, X.shape, int(y.sum()))
    return X, y


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Dynamic class-imbalance ratio: #negatives / #positives."""
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Training set must contain both classes")
    ratio = n_neg / n_pos
    logger.info("Class balance: neg=%d, pos=%d -> scale_pos_weight=%.4f", n_neg, n_pos, ratio)
    return ratio


# --------------------------------------------------------------------------- #
# Threshold optimisation (maximise F1)
# --------------------------------------------------------------------------- #
def optimize_threshold(
    y_true: pd.Series, proba: np.ndarray, t_min: float, t_max: float, t_step: float
) -> tuple[float, pd.DataFrame]:
    """Sweep thresholds on predict_proba() output and return the F1-maximising one."""
    thresholds = np.round(np.arange(t_min, t_max + t_step / 2, t_step), 4)
    rows = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        rows.append({"threshold": float(t), "f1": f1_score(y_true, preds, zero_division=0)})
    table = pd.DataFrame(rows)
    best = table.loc[table["f1"].idxmax()]  # first max wins on ties
    logger.info("Best threshold=%.2f -> F1=%.4f", best.threshold, best.f1)
    return float(best.threshold), table


def evaluate(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (proba >= threshold).astype(int)
    metrics = {
        "f1": f1_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "pr_auc": average_precision_score(y_true, proba),  # threshold-independent
        "accuracy": accuracy_score(y_true, preds),
    }
    for k, v in metrics.items():
        logger.info("  %-10s = %.4f", k, v)
    return metrics


# --------------------------------------------------------------------------- #
# Evaluation plots
# --------------------------------------------------------------------------- #
def load_class_labels(train_path: Path) -> dict[str, str]:
    """Read the original class names written by preprocess.py (falls back to 0/1)."""
    meta_path = train_path.parent / "metadata.json"
    labels = {"negative_class": "0", "positive_class": "1"}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            labels = {k: str(meta.get(k, v)) for k, v in labels.items()}
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s: %s", meta_path, exc)
    return labels


def save_confusion_matrix(y_true, proba, threshold: float, labels: dict, out_path: Path) -> None:
    preds = (proba >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, preds, labels=[0, 1],
        display_labels=[f"{labels['negative_class']} (0)", f"{labels['positive_class']} (1)"],
        cmap="Blues", values_format="d", colorbar=False, ax=ax,
    )
    ax.set_title(f"Confusion Matrix (threshold = {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_pr_curve(y_true, proba, threshold: float, metrics: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, proba, name="XGBoost", plot_chance_level=True, ax=ax)
    # Mark where the chosen (F1-optimal) threshold sits on the curve.
    ax.scatter(metrics["recall"], metrics["precision"], s=80, color="crimson", zorder=5,
               label=f"Operating point (t = {threshold:.2f})")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_roc_curve(y_true, proba, threshold: float, out_path: Path) -> None:
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, proba, name="XGBoost", plot_chance_level=True, ax=ax)
    ax.scatter(fpr, tpr, s=80, color="crimson", zorder=5, label=f"Operating point (t = {threshold:.2f})")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def safe_plot(name: str, fn, *args) -> bool:
    """Run one plotting function; a failed plot is logged but never fails the run."""
    try:
        fn(*args)
        logger.info("Saved %s", name)
        return True
    except Exception as exc:  # noqa: BLE001 - plots are non-critical artefacts
        logger.warning("Could not create %s: %s", name, exc)
        return False
    finally:
        plt.close("all")  # never leak figures, even after an error


# --------------------------------------------------------------------------- #
# Explainability
# --------------------------------------------------------------------------- #
def save_shap_summary(model: XGBClassifier, X: pd.DataFrame, out_path: Path, max_rows: int, seed: int) -> None:
    sample = X.sample(n=max_rows, random_state=seed) if len(X) > max_rows else X
    logger.info("Computing SHAP values on %d test rows", len(sample))
    shap_values = shap.TreeExplainer(model).shap_values(sample)
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP summary plot -> %s", out_path)


# --------------------------------------------------------------------------- #
# Custom DIY experiment logger
# --------------------------------------------------------------------------- #
def create_run_dir(output_root: Path) -> tuple[str, Path]:
    """Create outputs/run_YYYYMMDD_HHMMSS/ (waits for the next second on collision)."""
    while True:
        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = output_root / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_id, run_dir
        except FileExistsError:
            import time
            time.sleep(1)


def append_to_experiment_log(log_path: Path, row: dict) -> None:
    """Append one row to the global CSV, writing the header on first use."""
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info("Appended run to %s", log_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Universal XGBoost training with a DIY experiment logger")
    p.add_argument("--target", required=True, help="Name of the target column (same as in preprocess)")
    p.add_argument("--train-path", type=Path, default=Path("data/processed/train.parquet"))
    p.add_argument("--test-path", type=Path, default=Path("data/processed/test.parquet"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--n-estimators", type=int, default=150)
    p.add_argument("--threshold-min", type=float, default=0.1)
    p.add_argument("--threshold-max", type=float, default=0.9)
    p.add_argument("--threshold-step", type=float, default=0.01)
    p.add_argument("--shap-samples", type=int, default=2000, help="Max test rows used for SHAP")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger.info("=== Stage 2: Training started (target='%s') ===", args.target)

    X_train, y_train = load_split(args.train_path, args.target)
    X_test, y_test = load_split(args.test_path, args.target)
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train/test feature columns do not match")

    spw = compute_scale_pos_weight(y_train)
    model = XGBClassifier(
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        tree_method="hist",
        scale_pos_weight=spw,
        eval_metric="aucpr",
        random_state=args.random_state,
        n_jobs=-1,
    )
    logger.info("Training XGBClassifier (depth=%d, lr=%g, trees=%d)", args.max_depth, args.learning_rate, args.n_estimators)
    model.fit(X_train, y_train)

    # Probabilities -> F1-optimal threshold -> final metrics.
    proba = model.predict_proba(X_test)[:, 1]
    best_t, curve = optimize_threshold(y_test, proba, args.threshold_min, args.threshold_max, args.threshold_step)
    logger.info("Final metrics at threshold %.2f:", best_t)
    metrics = evaluate(y_test, proba, best_t)

    # ---- DIY logging: per-run folder + global CSV ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id, run_dir = create_run_dir(args.output_dir)
    logger.info("Run folder: %s", run_dir)

    model.save_model(run_dir / "xgboost_model.json")
    logger.info("Saved model -> %s", run_dir / "xgboost_model.json")

    # Evaluation + explainability plots (each one is optional / non-fatal).
    labels = load_class_labels(args.train_path)
    plots = {
        "confusion_matrix.png": (save_confusion_matrix, y_test, proba, best_t, labels),
        "pr_curve.png": (save_pr_curve, y_test, proba, best_t, metrics),
        "roc_curve.png": (save_roc_curve, y_test, proba, best_t),
        "shap_summary.png": (save_shap_summary, model, X_test),
    }
    saved_plots = []
    for name, (fn, *fn_args) in plots.items():
        out = run_dir / name
        extra = (args.shap_samples, args.random_state) if fn is save_shap_summary else ()
        if safe_plot(name, fn, *fn_args, out, *extra):
            saved_plots.append(name)

    (run_dir / "run_metrics.json").write_text(json.dumps({
        "run_id": run_id, "target": args.target, **labels, "best_threshold": best_t,
        "metrics": metrics, "plots": saved_plots, "n_train": len(X_train), "n_test": len(X_test),
        "n_features": X_train.shape[1], "threshold_curve": curve.to_dict(orient="records"),
    }, indent=2))

    append_to_experiment_log(args.output_dir / "experiments_log.csv", {
        "Timestamp": run_id,
        "Target_Column": args.target,
        "Max_Depth": args.max_depth,
        "Learning_Rate": args.learning_rate,
        "N_Estimators": args.n_estimators,
        "Scale_Pos_Weight": round(spw, 6),
        "Best_Threshold": best_t,
        "F1_Score": round(metrics["f1"], 6),
        "Precision": round(metrics["precision"], 6),
        "Recall": round(metrics["recall"], 6),
        "PR_AUC": round(metrics["pr_auc"], 6),
    })
    logger.info("=== Stage 2: Training finished ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        logger.debug("Traceback", exc_info=True)
        sys.exit(1)
