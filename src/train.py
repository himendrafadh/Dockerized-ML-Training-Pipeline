"""Stage 2: Cost-sensitive XGBoost training, threshold optimization & SHAP explainability.

Steps:
    1. Load processed train/test Parquet files.
    2. Compute scale_pos_weight = (#negatives / #positives) from the training set.
    3. Train an XGBClassifier.
    4. Sweep decision thresholds (0.1 -> 0.9) on predict_proba() output and pick the
       one minimizing business cost (FN = $500, FP = $50).
    5. Log threshold, cost, Accuracy / F1 / PR-AUC, SHAP summary plot and the model
       to MLflow.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for containers

import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger("train")

TARGET = "Churn"


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run src/preprocess.py first")
    df = pd.read_parquet(path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' missing from {path}")
    X, y = df.drop(columns=[TARGET]), df[TARGET].astype(int)
    logger.info("Loaded %s: X=%s, positives=%d", path, X.shape, int(y.sum()))
    return X, y


def compute_scale_pos_weight(y: pd.Series) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("Training set has no positive samples")
    ratio = n_neg / n_pos
    logger.info("Class balance: neg=%d, pos=%d -> scale_pos_weight=%.4f", n_neg, n_pos, ratio)
    return ratio


def train_model(X: pd.DataFrame, y: pd.Series, params: dict) -> XGBClassifier:
    logger.info("Training XGBClassifier with params: %s", params)
    model = XGBClassifier(**params)
    model.fit(X, y)
    logger.info("Training complete")
    return model


def business_cost(y_true: np.ndarray, y_pred: np.ndarray, fn_cost: float, fp_cost: float) -> tuple[float, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fn * fn_cost + fp * fp_cost, int(fn), int(fp)


def optimize_threshold(
    y_true: pd.Series,
    proba: np.ndarray,
    fn_cost: float,
    fp_cost: float,
    t_min: float,
    t_max: float,
    t_step: float,
) -> tuple[float, float, pd.DataFrame]:
    """Iterate over thresholds and return the one with minimum total business cost."""
    thresholds = np.round(np.arange(t_min, t_max + t_step / 2, t_step), 4)
    rows = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        cost, fn, fp = business_cost(y_true, preds, fn_cost, fp_cost)
        rows.append({"threshold": float(t), "total_cost": cost, "false_negatives": fn, "false_positives": fp})
        logger.debug("threshold=%.2f  FN=%d  FP=%d  cost=$%.0f", t, fn, fp, cost)

    table = pd.DataFrame(rows)
    best = table.loc[table["total_cost"].idxmin()]
    logger.info(
        "Optimal threshold=%.2f -> minimum cost=$%.0f (FN=%d, FP=%d)",
        best.threshold, best.total_cost, best.false_negatives, best.false_positives,
    )
    default = table.loc[(table["threshold"] - 0.5).abs().idxmin()]
    logger.info(
        "For reference, threshold=0.50 -> cost=$%.0f (savings: $%.0f)",
        default.total_cost, default.total_cost - best.total_cost,
    )
    return float(best.threshold), float(best.total_cost), table


def evaluate(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (proba >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "f1_score": f1_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds),
        "pr_auc": average_precision_score(y_true, proba),  # threshold-independent
    }
    for k, v in metrics.items():
        logger.info("  %-10s = %.4f", k, v)
    return metrics


def plot_cost_curve(table: pd.DataFrame, best_t: float, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table["threshold"], table["total_cost"], marker="o")
    ax.axvline(best_t, color="red", linestyle="--", label=f"optimal = {best_t:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Total business cost ($)")
    ax.set_title("Business cost vs. decision threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_shap_summary(model: XGBClassifier, X_test: pd.DataFrame, out_path: Path) -> Path:
    logger.info("Computing SHAP values on %d test rows", len(X_test))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("SHAP summary plot saved to %s", out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Telco churn: cost-sensitive XGBoost training stage")
    p.add_argument("--train-path", type=Path, default=Path("data/processed/train.parquet"))
    p.add_argument("--test-path", type=Path, default=Path("data/processed/test.parquet"))
    p.add_argument("--output-dir", type=Path, default=Path("."), help="Where local plots are written")
    p.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000"))
    p.add_argument("--experiment-name", default="Telco_Churn_XGBoost")
    p.add_argument("--registered-model-name", default=None,
                   help="If set, also register the model in the MLflow Model Registry")
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--n-estimators", type=int, default=150)
    p.add_argument("--fn-cost", type=float, default=500.0, help="Cost of a missed churner")
    p.add_argument("--fp-cost", type=float, default=50.0, help="Cost of an unnecessary retention offer")
    p.add_argument("--threshold-min", type=float, default=0.1)
    p.add_argument("--threshold-max", type=float, default=0.9)
    p.add_argument("--threshold-step", type=float, default=0.01)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger.info("=== Stage 2: Training started ===")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    logger.info("MLflow tracking URI: %s | experiment: %s", args.tracking_uri, args.experiment_name)

    X_train, y_train = load_split(args.train_path)
    X_test, y_test = load_split(args.test_path)
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Train/test feature columns do not match")

    params = {
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "tree_method": "hist",
        "scale_pos_weight": compute_scale_pos_weight(y_train),
        "eval_metric": "aucpr",
        "random_state": args.random_state,
        "n_jobs": -1,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        logger.info("MLflow run id: %s", run.info.run_id)
        mlflow.log_params(params)
        mlflow.log_params({
            "fn_cost": args.fn_cost,
            "fp_cost": args.fp_cost,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": X_train.shape[1],
        })

        model = train_model(X_train, y_train, params)

        # Cost-sensitive threshold optimization on probabilities (not .predict()).
        proba = model.predict_proba(X_test)[:, 1]
        best_t, min_cost, cost_table = optimize_threshold(
            y_test, proba, args.fn_cost, args.fp_cost,
            args.threshold_min, args.threshold_max, args.threshold_step,
        )
        mlflow.log_metric("optimal_threshold", best_t)
        mlflow.log_metric("minimum_cost", min_cost)

        logger.info("Final metrics at optimal threshold %.2f:", best_t)
        mlflow.log_metrics(evaluate(y_test, proba, best_t))

        cost_csv = args.output_dir / "threshold_cost_table.csv"
        cost_table.to_csv(cost_csv, index=False)
        mlflow.log_artifact(str(cost_csv))
        mlflow.log_artifact(str(plot_cost_curve(cost_table, best_t, args.output_dir / "cost_curve.png")))

        # Explainable AI.
        shap_png = generate_shap_summary(model, X_test, args.output_dir / "shap_summary.png")
        mlflow.log_artifact(str(shap_png))

        # Model logging (+ optional registration).
        signature = infer_signature(X_test, proba)
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.head(5),
            registered_model_name=args.registered_model_name,
        )
        mlflow.set_tag("optimal_threshold", f"{best_t:.2f}")
        logger.info("Model logged to MLflow (run %s)", run.info.run_id)

    logger.info("=== Stage 2: Training finished ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)
