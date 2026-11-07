"""Stage 1: Schema validation & preprocessing for the Telco Customer Churn dataset.

Steps:
    1. Load raw CSV and normalize the schema (supports both the Kaggle and the
       IBM "extended" variants of the dataset).
    2. Fix the TotalCharges "space trap" (" " -> NaN -> numeric -> 0).
    3. Drop identifiers, encode the target (Yes/No -> 1/0).
    4. Impute remaining missing values (numeric: median, categorical: mode).
    5. One-hot encode categoricals (drop_first=True).
    6. Stratified 80/20 split and export to Parquet.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("preprocess")

TARGET = "Churn"
ID_COL = "customerID"

# Canonical (Kaggle) schema the rest of the pipeline relies on.
CANONICAL_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]

# IBM extended dataset -> canonical names. Columns not in the canonical schema
# (geo data, Count, and the leakage columns Churn Value / Churn Score /
# Churn Reason / CLTV) are dropped during normalization.
IBM_RENAME_MAP = {
    "CustomerID": "customerID",
    "Gender": "gender",
    "Senior Citizen": "SeniorCitizen",
    "Tenure Months": "tenure",
    "Phone Service": "PhoneService",
    "Multiple Lines": "MultipleLines",
    "Internet Service": "InternetService",
    "Online Security": "OnlineSecurity",
    "Online Backup": "OnlineBackup",
    "Device Protection": "DeviceProtection",
    "Tech Support": "TechSupport",
    "Streaming TV": "StreamingTV",
    "Streaming Movies": "StreamingMovies",
    "Paperless Billing": "PaperlessBilling",
    "Payment Method": "PaymentMethod",
    "Monthly Charges": "MonthlyCharges",
    "Total Charges": "TotalCharges",
    "Churn Label": "Churn",
}


class SchemaValidationError(ValueError):
    """Raised when the raw data does not satisfy the expected schema."""


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")
    # keep_default_na=False so a literal " " survives the read untouched and is
    # handled explicitly by the space-trap logic below.
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    logger.info("Loaded raw data from %s: %d rows x %d cols", path, *df.shape)
    return df


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Map known dataset variants onto the canonical schema and validate it."""
    if "Churn Label" in df.columns or "Total Charges" in df.columns:
        logger.info("Detected IBM extended schema -> renaming to canonical schema")
        df = df.rename(columns=IBM_RENAME_MAP)
        dropped = [c for c in df.columns if c not in CANONICAL_COLUMNS]
        logger.info("Dropping %d non-canonical / leakage columns: %s", len(dropped), dropped)
        df = df[[c for c in CANONICAL_COLUMNS if c in df.columns]]

    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(f"Missing required columns: {missing}")

    df = df[CANONICAL_COLUMNS].copy()

    # SeniorCitizen is 0/1 in the Kaggle version but Yes/No in the IBM version.
    if df["SeniorCitizen"].dtype == object:
        df["SeniorCitizen"] = df["SeniorCitizen"].map({"Yes": 1, "No": 0})

    labels = set(df[TARGET].dropna().astype(str).str.strip().unique())
    if not labels <= {"Yes", "No"}:
        raise SchemaValidationError(f"Unexpected target values in '{TARGET}': {labels}")

    if df[ID_COL].duplicated().any():
        raise SchemaValidationError(f"Duplicate values found in '{ID_COL}'")

    logger.info("Schema validation passed (%d columns)", df.shape[1])
    return df


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Handle the TotalCharges 'space trap' for brand-new customers."""
    col = df["TotalCharges"].astype(str)
    blank_mask = col.str.strip().eq("") | col.str.lower().eq("nan")
    n_blank = int(blank_mask.sum())
    if n_blank:
        logger.warning(
            "TotalCharges space trap: %d blank/whitespace value(s) detected "
            "(tenure of affected rows: %s)",
            n_blank, sorted(df.loc[blank_mask, "tenure"].unique().tolist()),
        )

    numeric = pd.to_numeric(col.where(~blank_mask, np.nan), errors="coerce")
    unparseable = numeric.isna() & ~blank_mask
    if unparseable.any():
        raise SchemaValidationError(
            f"TotalCharges has {int(unparseable.sum())} non-numeric value(s): "
            f"{col[unparseable].unique()[:5].tolist()}"
        )

    df["TotalCharges"] = numeric.fillna(0.0).astype(float)
    logger.info("TotalCharges coerced to float; %d NaN(s) filled with 0", n_blank)
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    df[TARGET] = df[TARGET].astype(str).str.strip().map({"Yes": 1, "No": 0}).astype(int)
    logger.info("Target encoded. Class distribution: %s", df[TARGET].value_counts().to_dict())
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    features = [c for c in df.columns if c != TARGET]
    num_cols = df[features].select_dtypes(include="number").columns
    cat_cols = df[features].select_dtypes(exclude="number").columns

    for c in num_cols:
        n = int(df[c].isna().sum())
        if n:
            median = df[c].median()
            df[c] = df[c].fillna(median)
            logger.info("Filled %d missing in numeric '%s' with median=%.4f", n, c, median)

    for c in cat_cols:
        n = int(df[c].isna().sum())
        if n:
            mode = df[c].mode(dropna=True).iloc[0]
            df[c] = df[c].fillna(mode)
            logger.info("Filled %d missing in categorical '%s' with mode=%r", n, c, mode)

    logger.info("Imputation complete (%d numeric, %d categorical features)", len(num_cols), len(cat_cols))
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = [c for c in df.select_dtypes(exclude="number").columns if c != TARGET]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=np.uint8)
    # Sanitize names for XGBoost (no [, ] or < allowed in feature names).
    df.columns = (
        df.columns.str.replace(r"[\[\]<]", "_", regex=True)
        .str.replace(r"[\s()]+", "_", regex=True)
        .str.strip("_")
    )
    logger.info("One-hot encoded %d categorical columns -> %d total columns", len(cat_cols), df.shape[1])
    return df


def split_and_save(df: pd.DataFrame, out_dir: Path, test_size: float, seed: int) -> None:
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    train = X_train.assign(**{TARGET: y_train})
    test = X_test.assign(**{TARGET: y_test})
    train.to_parquet(out_dir / "train.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)

    logger.info("Train: %s, churn rate=%.3f -> %s", train.shape, y_train.mean(), out_dir / "train.parquet")
    logger.info("Test:  %s, churn rate=%.3f -> %s", test.shape, y_test.mean(), out_dir / "test.parquet")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Telco churn: validation & preprocessing stage")
    p.add_argument("--input", type=Path, default=Path("data/raw/Telco_customer_churn.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger.info("=== Stage 1: Preprocessing started ===")

    df = load_raw(args.input)
    df = normalize_schema(df)
    df = fix_total_charges(df)
    df = df.drop(columns=[ID_COL])
    logger.info("Dropped identifier column '%s'", ID_COL)
    df = encode_target(df)
    df = impute_missing(df)
    df = one_hot_encode(df)
    split_and_save(df, args.output_dir, args.test_size, args.random_state)

    logger.info("=== Stage 1: Preprocessing finished ===")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Preprocessing failed")
        sys.exit(1)
