"""Model registry for the training stage.

Every model is described by one ModelSpec: how to build it, how to save it, and how to
explain it with SHAP. train.py only talks to this registry, so adding a model means adding
one entry here (and its label to the UI).

All models receive the same shared hyperparameters (max_depth, learning_rate, n_estimators)
wherever the model has an equivalent, so runs are comparable. Class imbalance is handled
per model: XGBoost, LightGBM and CatBoost take `scale_pos_weight`; the scikit-learn models
use `class_weight="balanced"`, which has the same effect.

Heavy libraries (xgboost, lightgbm, catboost) are imported lazily, only when that model
is actually built or saved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

SHARED_PARAMS = ("max_depth", "learning_rate", "n_estimators", "scale_pos_weight")


@dataclass(frozen=True)
class ModelSpec:
    key: str                              # CLI / API identifier
    label: str                            # shown in the UI and the experiment log
    filename: str                         # model file saved inside outputs/run_<timestamp>/
    shap_kind: str                        # "tree" | "linear" | "permutation"
    uses: tuple[str, ...]                 # which SHARED_PARAMS this model really uses
    build: Callable[[dict, float, int], Any]   # (hyperparams, scale_pos_weight, seed) -> estimator
    save: Callable[[Any, Path], None]
    shap_max_rows: int | None = None      # cap on SHAP rows for slow explainers


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def _build_xgboost(hp: dict, spw: float, seed: int):
    from xgboost import XGBClassifier

    return XGBClassifier(
        max_depth=hp["max_depth"], learning_rate=hp["learning_rate"], n_estimators=hp["n_estimators"],
        tree_method="hist", scale_pos_weight=spw, eval_metric="aucpr",
        random_state=seed, n_jobs=-1,
    )


def _build_random_forest(hp: dict, spw: float, seed: int):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
        class_weight="balanced", random_state=seed, n_jobs=-1,
    )


def _build_logistic_regression(hp: dict, spw: float, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # Scaling lives inside the model so tree models never see scaled features.
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)),
    ])


def _build_hist_gradient_boosting(hp: dict, spw: float, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=hp["n_estimators"], learning_rate=hp["learning_rate"], max_depth=hp["max_depth"],
        class_weight="balanced", early_stopping=False, random_state=seed,
    )


def _build_lightgbm(hp: dict, spw: float, seed: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=hp["n_estimators"], learning_rate=hp["learning_rate"], max_depth=hp["max_depth"],
        scale_pos_weight=spw, random_state=seed, n_jobs=-1, verbosity=-1,
    )


def _build_catboost(hp: dict, spw: float, seed: int):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=hp["n_estimators"], learning_rate=hp["learning_rate"], depth=hp["max_depth"],
        scale_pos_weight=spw, random_seed=seed, thread_count=-1,
        verbose=0, allow_writing_files=False,  # no catboost_info/ folder next to the code
    )


# --------------------------------------------------------------------------- #
# Savers. XGBoost, LightGBM and CatBoost use their own plain-data formats; only the
# scikit-learn models need joblib (pickle-based: load such files only if you trust them).
# --------------------------------------------------------------------------- #
def _save_xgboost(model, path: Path) -> None:
    model.save_model(path)


def _save_joblib(model, path: Path) -> None:
    import joblib

    joblib.dump(model, path)


def _save_lightgbm(model, path: Path) -> None:
    model.booster_.save_model(str(path))


def _save_catboost(model, path: Path) -> None:
    model.save_model(str(path), format="cbm")


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in [
        ModelSpec("xgboost", "XGBoost", "xgboost_model.json", "tree",
                  ("max_depth", "learning_rate", "n_estimators", "scale_pos_weight"),
                  _build_xgboost, _save_xgboost),
        ModelSpec("random_forest", "Random Forest", "random_forest_model.joblib", "tree",
                  ("max_depth", "n_estimators"),
                  _build_random_forest, _save_joblib),
        ModelSpec("logistic_regression", "Logistic Regression", "logistic_regression_model.joblib", "linear",
                  (),
                  _build_logistic_regression, _save_joblib),
        ModelSpec("hist_gradient_boosting", "HistGradientBoosting", "hist_gradient_boosting_model.joblib",
                  "permutation", ("max_depth", "learning_rate", "n_estimators"),
                  _build_hist_gradient_boosting, _save_joblib, shap_max_rows=300),
        ModelSpec("lightgbm", "LightGBM", "lightgbm_model.txt", "tree",
                  ("max_depth", "learning_rate", "n_estimators", "scale_pos_weight"),
                  _build_lightgbm, _save_lightgbm),
        ModelSpec("catboost", "CatBoost", "catboost_model.cbm", "tree",
                  ("max_depth", "learning_rate", "n_estimators", "scale_pos_weight"),
                  _build_catboost, _save_catboost),
    ]
}

DEFAULT_MODEL = "xgboost"
MODEL_FILENAMES = {spec.filename for spec in MODELS.values()}


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise ValueError(f"unknown model '{key}'; choose one of: {', '.join(MODELS)}") from None
