"""Stage 2: model training with a custom (file-based) experiment logger.

The model is chosen with --model (see models.py: XGBoost, Random Forest, Logistic Regression,
HistGradientBoosting, LightGBM, CatBoost). The decision threshold is chosen with 5-fold
cross-validation on the training set, so the test set is only used for the final scores.
No external tracking server is used. Every run:
    * creates   outputs/run_YYYYMMDD_HHMMSS/
    * saves     the model file, evaluation plots, shap_summary.png, run_metrics.json inside it
    * appends a row to the global  outputs/experiments_log.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend (Docker / web server subprocess)

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
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from models import DEFAULT_MODEL, MODELS, ModelSpec, get_model

logger = logging.getLogger("train")

LOG_HEADER = [
    "Timestamp", "Target_Column", "Model", "Max_Depth", "Learning_Rate", "N_Estimators",
    "Scale_Pos_Weight", "Best_Threshold", "F1_Score", "Precision", "Recall", "PR_AUC",
]
LEGACY_MODEL_NAME = "XGBoost"  # rows written before the Model column existed were XGBoost runs


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


def cv_threshold(
    spec: ModelSpec, hyperparams: dict, spw: float, X_train: pd.DataFrame, y_train: pd.Series, args: argparse.Namespace
) -> tuple[float, pd.DataFrame, str] | None:
    """Choose the F1-optimal threshold on out-of-fold predictions of the TRAINING set.

    The test set is never used to pick the threshold, so test scores stay unbiased.
    Returns None when cross-validation is disabled or a class is too small to split.
    """
    folds = min(args.cv_folds, int(y_train.value_counts().min()))
    if folds < 2:
        return None
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.random_state)
    started = time.time()
    oof = cross_val_predict(spec.build(hyperparams, spw, args.random_state), X_train, y_train,
                            cv=cv, method="predict_proba")[:, 1]
    best_t, curve = optimize_threshold(y_train, oof, args.threshold_min, args.threshold_max, args.threshold_step)
    logger.info("Threshold chosen by %d-fold cross-validation on the training set (%.1f s)", folds, time.time() - started)
    return best_t, curve, f"{folds}-fold cross-validation on the training set"


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


def load_excluded_columns(train_path: Path) -> list[str]:
    """Columns the user left out in preprocess.py (--exclude), recorded for the run report."""
    meta_path = train_path.parent / "metadata.json"
    try:
        return list(json.loads(meta_path.read_text()).get("excluded_columns", []))
    except (OSError, json.JSONDecodeError):
        return []


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


def save_pr_curve(y_true, proba, threshold: float, metrics: dict, model_label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, proba, name=model_label, plot_chance_level=True, ax=ax)
    # Mark where the chosen (F1-optimal) threshold sits on the curve.
    ax.scatter(metrics["recall"], metrics["precision"], s=80, color="crimson", zorder=5,
               label=f"Operating point (t = {threshold:.2f})")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_roc_curve(y_true, proba, threshold: float, model_label: str, out_path: Path) -> None:
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, proba, name=model_label, plot_chance_level=True, ax=ax)
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
def _positive_class_values(values) -> np.ndarray:
    """Different SHAP/model versions return a list per class, a 3D array or a 2D array."""
    if isinstance(values, list):
        return np.asarray(values[1] if len(values) > 1 else values[0])
    arr = np.asarray(values)
    return arr[:, :, 1] if arr.ndim == 3 else arr


def compute_shap_values(spec: ModelSpec, model, X_train: pd.DataFrame, sample: pd.DataFrame, seed: int) -> np.ndarray:
    """SHAP values for the positive class, using the explainer that fits the model type."""
    background = X_train.sample(n=min(100, len(X_train)), random_state=seed)

    if spec.shap_kind == "tree":
        return _positive_class_values(shap.TreeExplainer(model).shap_values(sample))

    if spec.shap_kind == "linear":
        # The linear model sees standardised features, so explain it in that space.
        scaler, clf = model.named_steps["scale"], model.named_steps["clf"]
        scaled_bg = pd.DataFrame(scaler.transform(background), columns=sample.columns)
        scaled_sample = pd.DataFrame(scaler.transform(sample), columns=sample.columns)
        return _positive_class_values(shap.LinearExplainer(clf, scaled_bg).shap_values(scaled_sample))

    # Model-agnostic fallback (HistGradientBoosting has no tree explainer): a permutation
    # explainer on the positive-class probability. Slower, so the caller caps the row count.
    columns = list(sample.columns)

    def predict_positive(data):
        return model.predict_proba(pd.DataFrame(data, columns=columns))[:, 1]

    masker = shap.maskers.Independent(background.iloc[:50], max_samples=50)
    explainer = shap.Explainer(predict_positive, masker, algorithm="permutation")
    return np.asarray(explainer(sample, max_evals=2 * len(columns) + 1).values)


def save_shap_summary(
    spec: ModelSpec, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
    max_rows: int, seed: int, out_path: Path,
) -> None:
    rows = min(max_rows, spec.shap_max_rows) if spec.shap_max_rows else max_rows
    sample = X_test.sample(n=rows, random_state=seed) if len(X_test) > rows else X_test
    logger.info("Computing SHAP values (%s explainer) on %d test rows", spec.shap_kind, len(sample))
    started = time.time()
    shap_values = compute_shap_values(spec, model, X_train, sample, seed)
    logger.info("SHAP values computed in %.1f s", time.time() - started)
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)  # colours use the original feature values
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
            time.sleep(1)


def migrate_log_header(log_path: Path) -> list[str]:
    """Make sure the log has the current header; returns the header the file now uses.

    Logs written before the Model column existed are rewritten once, with those rows
    marked as XGBoost runs (the only model that existed then). A file whose header is
    something else entirely is left untouched and appended to using its own header.
    """
    with log_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    legacy = [c for c in LOG_HEADER if c != "Model"]
    if fieldnames == LOG_HEADER or "Model" in fieldnames or not set(legacy) <= set(fieldnames):
        return fieldnames

    for row in rows:
        row["Model"] = LEGACY_MODEL_NAME
    tmp_path = log_path.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_HEADER, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, log_path)
    logger.info("Migrated %s to the new header (added Model column to %d old row(s))", log_path, len(rows))
    return LOG_HEADER


def append_to_experiment_log(log_path: Path, row: dict) -> None:
    """Append one row to the global CSV, writing the header on first use."""
    is_new = not log_path.exists() or log_path.stat().st_size == 0
    fieldnames = LOG_HEADER if is_new else migrate_log_header(log_path)
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    logger.info("Appended run to %s", log_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Universal binary-classification training with a DIY experiment logger")
    p.add_argument("--target", required=True, help="Name of the target column (same as in preprocess)")
    p.add_argument("--model", choices=list(MODELS), default=DEFAULT_MODEL, help="Which model to train")
    p.add_argument("--train-path", type=Path, default=Path("data/processed/train.parquet"))
    p.add_argument("--test-path", type=Path, default=Path("data/processed/test.parquet"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--n-estimators", type=int, default=150)
    p.add_argument("--threshold-min", type=float, default=0.1)
    p.add_argument("--threshold-max", type=float, default=0.9)
    p.add_argument("--threshold-step", type=float, default=0.01)
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Folds used to choose the threshold on the training set (0 = choose on the test set)")
    p.add_argument("--shap-samples", type=int, default=2000, help="Max test rows used for SHAP")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    spec = get_model(args.model)
    logger.info("=== Stage 2: Training started (model=%s, target='%s') ===", spec.label, args.target)

    X_train, y_train = load_split(args.train_path, args.target)
    X_test, y_test = load_split(args.test_path, args.target)
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train/test feature columns do not match")

    spw = compute_scale_pos_weight(y_train)
    hyperparams = {"max_depth": args.max_depth, "learning_rate": args.learning_rate, "n_estimators": args.n_estimators}
    used = {name: value for name, value in {**hyperparams, "scale_pos_weight": spw}.items() if name in spec.uses}
    logger.info("Training %s with %s", spec.label, {k: round(v, 4) if isinstance(v, float) else v for k, v in used.items()} or "default settings")

    # 1) Decision threshold from cross-validation on the training set only.
    chosen = cv_threshold(spec, hyperparams, spw, X_train, y_train, args)

    # 2) Final model on the whole training set, evaluated once on the test set.
    model = spec.build(hyperparams, spw, args.random_state)
    started = time.time()
    model.fit(X_train, y_train)
    logger.info("Training took %.1f s", time.time() - started)
    proba = model.predict_proba(X_test)[:, 1]

    if chosen is None:  # CV disabled (--cv-folds 0) or a class with fewer than 2 training rows
        best_t, curve = optimize_threshold(y_test, proba, args.threshold_min, args.threshold_max, args.threshold_step)
        threshold_source = "test set (cross-validation disabled or not possible; scores are optimistic)"
        logger.warning("Threshold chosen on the test set: %s", threshold_source)
    else:
        best_t, curve, threshold_source = chosen
    logger.info("Final metrics on the test set at threshold %.2f:", best_t)
    metrics = evaluate(y_test, proba, best_t)

    # ---- DIY logging: per-run folder + global CSV ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id, run_dir = create_run_dir(args.output_dir)
    logger.info("Run folder: %s", run_dir)

    spec.save(model, run_dir / spec.filename)
    logger.info("Saved model -> %s", run_dir / spec.filename)

    # Evaluation + explainability plots (each one is optional / non-fatal).
    labels = load_class_labels(args.train_path)
    plots = {
        "confusion_matrix.png": (save_confusion_matrix, y_test, proba, best_t, labels),
        "pr_curve.png": (save_pr_curve, y_test, proba, best_t, metrics, spec.label),
        "roc_curve.png": (save_roc_curve, y_test, proba, best_t, spec.label),
        "shap_summary.png": (save_shap_summary, spec, model, X_train, X_test, args.shap_samples, args.random_state),
    }
    saved_plots = []
    for name, (fn, *fn_args) in plots.items():
        if safe_plot(name, fn, *fn_args, run_dir / name):
            saved_plots.append(name)

    (run_dir / "run_metrics.json").write_text(json.dumps({
        "run_id": run_id, "target": args.target, "model": spec.key, "model_label": spec.label,
        "model_file": spec.filename, "hyperparameters": used, **labels,
        "excluded_columns": load_excluded_columns(args.train_path), "best_threshold": best_t,
        "threshold_source": threshold_source,
        "metrics": metrics, "plots": saved_plots, "n_train": len(X_train), "n_test": len(X_test),
        "n_features": X_train.shape[1], "threshold_curve": curve.to_dict(orient="records"),
    }, indent=2))

    append_to_experiment_log(args.output_dir / "experiments_log.csv", {
        "Timestamp": run_id,
        "Target_Column": args.target,
        "Model": spec.label,
        "Max_Depth": used.get("max_depth"),            # blank when the model has no such setting
        "Learning_Rate": used.get("learning_rate"),
        "N_Estimators": used.get("n_estimators"),
        "Scale_Pos_Weight": round(spw, 6) if "scale_pos_weight" in used else None,
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
