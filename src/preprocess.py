"""Stage 1: Universal validation & preprocessing for any binary-classification CSV.

Pipeline (all steps are dataset-agnostic):
    1. Load the CSV and normalise text cells (strip whitespace, blank -> NaN).
       This neutralises the classic " " trap in every string column.
    2. Drop rows with a missing target, then encode the target to 1/0.
    3. Drop useless features: ID-like columns (all values unique), constant
       columns, all-NaN columns, ultra-high-cardinality text columns and
       features that are (near-)perfect copies of the target (leakage).
    4. Cast string columns that are really numeric (e.g. "29.85", " ") to float.
    5. Impute: median for numerics, mode for categoricals.
    6. One-hot encode categoricals (drop_first=True).
    7. Stratified 80/20 split -> Parquet (+ a small metadata.json for the UI).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger("preprocess")

# Tokens recognised as the "positive" / "negative" class of a binary target.
POSITIVE_TOKENS = {"yes", "y", "true", "t", "1", "positive", "pos"}
NEGATIVE_TOKENS = {"no", "n", "false", "f", "0", "negative", "neg"}
STRING_DTYPES = ["object", "category", "string"]


class DataValidationError(ValueError):
    """Raised when the uploaded dataset cannot be used for binary classification."""


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout,
    )


# --------------------------------------------------------------------------- #
# Loading & text cleaning
# --------------------------------------------------------------------------- #
CANDIDATE_SEPARATORS = [",", ";", "\t", "|"]
ENCODINGS = ["utf-8-sig", "latin-1"]  # utf-8 (with/without BOM) first; latin-1 never fails


def detect_separator(sample: str) -> str:
    """Pick the delimiter that splits the first lines into the same (>1) number of fields.

    Uses the csv module per candidate so delimiters inside quoted cells
    (e.g. "Bandung, Jawa") are not miscounted. Falls back to a comma.
    """
    best_sep, best_fields = ",", 1
    for sep in CANDIDATE_SEPARATORS:
        try:
            counts = [len(r) for r in csv.reader(io.StringIO(sample), delimiter=sep) if r]
        except csv.Error:
            continue
        if len(counts) < 2:
            continue
        mode = max(set(counts), key=counts.count)
        consistent = counts.count(mode) / len(counts) >= 0.9
        if consistent and mode > best_fields:
            best_sep, best_fields = sep, mode
    return best_sep


def read_dataset(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a delimited text file, auto-detecting the separator and encoding."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    for encoding in ENCODINGS:
        try:
            with open(path, encoding=encoding, newline="") as f:
                sample = "".join(f.readline() for _ in range(50))
            sep = detect_separator(sample)
            # keep_default_na=False keeps literal " " cells intact so we can detect them
            # explicitly; only truly empty cells become NaN at read time.
            df = pd.read_csv(path, sep=sep, encoding=encoding, nrows=nrows,
                             keep_default_na=False, na_values=[""], low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    df.columns = [str(c).strip() for c in df.columns]
    df.attrs["separator"], df.attrs["encoding"] = sep, encoding
    return df


def load_csv(path: Path) -> pd.DataFrame:
    df = read_dataset(path)
    if df.empty:
        raise DataValidationError("The uploaded CSV has no rows")
    if df.shape[1] < 2:
        raise DataValidationError(
            f"Only 1 column was detected (separator {df.attrs['separator']!r}); "
            "check that the file is a valid delimited text file"
        )
    logger.info("Loaded %s: %d rows x %d cols (separator=%r, encoding=%s)",
                path, *df.shape, df.attrs["separator"], df.attrs["encoding"])
    return df


def clean_string_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace in every string column; blank / whitespace-only -> NaN."""
    for col in df.select_dtypes(include=STRING_DTYPES).columns:
        stripped = df[col].astype("string").str.strip()
        n_blank = int((stripped == "").sum())
        if n_blank:
            logger.warning("Space trap: %d blank/whitespace cell(s) in '%s' -> NaN", n_blank, col)
        df[col] = stripped.replace("", pd.NA).astype(object).where(stripped.notna() & (stripped != ""), np.nan)
    return df


# --------------------------------------------------------------------------- #
# Target handling
# --------------------------------------------------------------------------- #
def _normalise_label(value) -> str:
    """'Yes ' -> 'yes', 1.0 -> '1', True -> 'true'."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value)).lower()
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower()


def encode_target(y: pd.Series) -> tuple[pd.Series, dict]:
    """Map a 2-class column to 1/0. Returns the encoded series and the mapping used."""
    counts = y.value_counts()
    if len(counts) != 2:
        raise DataValidationError(
            f"Target '{y.name}' must have exactly 2 classes for binary classification; "
            f"found {len(counts)}: {counts.index.tolist()[:10]}"
        )
    classes = list(counts.index)
    norm = {c: _normalise_label(c) for c in classes}
    pos = [c for c in classes if norm[c] in POSITIVE_TOKENS]
    neg = [c for c in classes if norm[c] in NEGATIVE_TOKENS]

    if len(pos) == 1 and len(neg) == 1:
        positive = pos[0]
        reason = "recognised yes/true/1-style label"
    else:
        positive = counts.idxmin()  # minority class is normally the one of interest
        reason = "minority class (no yes/no-style labels recognised)"
    negative = next(c for c in classes if c != positive)

    logger.info("Target classes: positive(1)=%r, negative(0)=%r [%s]", positive, negative, reason)
    encoded = (y == positive).astype(int)
    mapping = {"positive_class": str(positive), "negative_class": str(negative), "reason": reason}
    return encoded, mapping


# --------------------------------------------------------------------------- #
# Feature cleaning
# --------------------------------------------------------------------------- #
def _categorical_error_reduction(series: pd.Series, y: pd.Series) -> float:
    """Share of the majority-class baseline's errors removed by predicting the target
    from the column's categories alone (1.0 = the column perfectly determines y)."""
    mask = series.notna()
    y_m = y[mask]
    baseline_errors = min(int(y_m.sum()), int((1 - y_m).sum()))
    if baseline_errors == 0:
        return 0.0
    ct = pd.crosstab(series[mask], y_m)
    category_errors = int((ct.sum(axis=1) - ct.max(axis=1)).sum())
    return 1.0 - category_errors / baseline_errors


def drop_useless_features(
    X: pd.DataFrame, y: pd.Series, max_categories: int, leakage_threshold: float
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Drop ID-like, constant, empty, high-cardinality and target-leaking columns."""
    n_rows = len(X)
    dropped: dict[str, str] = {}

    for col in X.columns:
        series = X[col]
        n_unique = series.nunique(dropna=True)
        is_float = pd.api.types.is_float_dtype(series)

        if series.isna().all():
            dropped[col] = "all values missing"
        elif n_unique <= 1:
            dropped[col] = "constant column"
        elif n_rows > 1 and series.notna().all() and n_unique == n_rows and not is_float:
            dropped[col] = "ID-like (every value is unique)"
        elif not pd.api.types.is_numeric_dtype(series) and n_unique > max_categories:
            # A string column that is mostly numbers (stored as text) is kept; it gets
            # cast to float later. Otherwise one-hot would explode the feature space.
            as_num = pd.to_numeric(series.dropna(), errors="coerce")
            if as_num.notna().all():
                continue
            dropped[col] = f"high-cardinality text ({n_unique} categories > {max_categories})"

    # Leakage guard: features that (almost) reproduce the target.
    for col in X.columns:
        if col in dropped:
            continue
        series = X[col]
        # (a) numeric features: near-perfect linear correlation with the 1/0 target.
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() >= 2 and numeric.nunique() >= 2:
            corr = numeric.fillna(numeric.median()).corr(y)
            if pd.notna(corr) and abs(corr) >= leakage_threshold:
                dropped[col] = f"target leakage (|corr with target| = {abs(corr):.3f})"
                continue
        # (b) text/categorical features: categories that (almost) determine the target,
        #     e.g. a "Yes/No" copy of the target stored under another name.
        if not pd.api.types.is_numeric_dtype(series):
            removed = _categorical_error_reduction(series, y)
            if removed >= leakage_threshold and n_rows / max(series.nunique(), 1) >= 20:
                dropped[col] = f"target leakage (categories explain {removed:.1%} of the target)"

    for col, why in dropped.items():
        logger.warning("Dropping column '%s': %s", col, why)
    logger.info("Dropped %d column(s); %d feature(s) remain", len(dropped), X.shape[1] - len(dropped))
    return X.drop(columns=list(dropped)), dropped


def cast_numeric_strings(X: pd.DataFrame) -> pd.DataFrame:
    """Cast string columns whose non-null values are all numeric to float; fix bools."""
    for col in X.select_dtypes(include=STRING_DTYPES).columns:
        non_null = X[col].dropna()
        converted = pd.to_numeric(non_null, errors="coerce")
        if len(non_null) and converted.notna().all():
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
            logger.info("Cast text column '%s' to float", col)
    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)
    return X


def impute_missing(X: pd.DataFrame) -> pd.DataFrame:
    """Median for numeric columns, mode for categorical columns."""
    for col in X.columns:
        n_missing = int(X[col].isna().sum())
        if not n_missing:
            continue
        if pd.api.types.is_numeric_dtype(X[col]):
            fill = float(X[col].median())
            kind = "median"
        else:
            fill = X[col].mode(dropna=True).iloc[0]
            kind = "mode"
        X[col] = X[col].fillna(fill)
        logger.info("Imputed %d missing in '%s' with %s=%r", n_missing, col, kind, fill)
    return X


def one_hot_encode(X: pd.DataFrame, reserved: set[str]) -> pd.DataFrame:
    cat_cols = X.select_dtypes(include=STRING_DTYPES).columns.tolist()
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=np.uint8)

    # Make feature names safe for every supported model: XGBoost forbids '[', ']' and '<',
    # LightGBM forbids JSON characters (quotes, commas, colons, braces). Keep letters (any
    # language), digits, '_', '.' and '-'; replace the rest, and guarantee uniqueness.
    seen, new_cols = set(reserved), []
    for col in X.columns:
        name = re.sub(r"[^\w.\-]", "_", str(col))
        name = re.sub(r"_+", "_", name).strip("_") or "feature"
        base, i = name, 1
        while name in seen:
            i += 1
            name = f"{base}_{i}"
        seen.add(name)
        new_cols.append(name)
    X.columns = new_cols
    logger.info("One-hot encoded %d categorical column(s) -> %d feature(s)", len(cat_cols), X.shape[1])
    return X


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def split_and_save(X, y, target, out_dir: Path, test_size: float, seed: int) -> None:
    if y.value_counts().min() < 2:
        raise DataValidationError("Each class needs at least 2 samples for a stratified split")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    X_tr.assign(**{target: y_tr}).to_parquet(out_dir / "train.parquet", index=False)
    X_te.assign(**{target: y_te}).to_parquet(out_dir / "test.parquet", index=False)
    logger.info("Train: %s (positive rate %.3f) -> %s", X_tr.shape, y_tr.mean(), out_dir / "train.parquet")
    logger.info("Test:  %s (positive rate %.3f) -> %s", X_te.shape, y_te.mean(), out_dir / "test.parquet")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Universal preprocessing for binary classification CSVs")
    p.add_argument("--target", required=True, help="Name of the target column to predict")
    p.add_argument("--input", type=Path, default=Path("data/raw/dataset.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--max-categories", type=int, default=50,
                   help="Drop text columns with more distinct values than this")
    p.add_argument("--leakage-threshold", type=float, default=0.98,
                   help="Drop features whose |correlation| with the target is >= this")
    p.add_argument("--exclude", action="append", default=[], metavar="COLUMN",
                   help="Column to leave out of training, e.g. one only known after the outcome. "
                        "Repeat for several columns; use --exclude=NAME for names starting with '-'")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def exclude_columns(df: pd.DataFrame, target: str, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Drop user-excluded columns (validated: must exist and must not be the target)."""
    excluded = [c.strip() for c in dict.fromkeys(columns) if c.strip()]  # de-duplicate, keep order
    if target in excluded:
        raise DataValidationError(f"The target column '{target}' cannot also be excluded")
    unknown = [c for c in excluded if c not in df.columns]
    if unknown:
        raise DataValidationError(f"Columns to exclude not found: {unknown}")
    if excluded:
        logger.info("Excluding %d column(s) chosen by the user: %s", len(excluded), excluded)
    return df.drop(columns=excluded), excluded


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger.info("=== Stage 1: Preprocessing started (target='%s') ===", args.target)

    df = load_csv(args.input)
    if args.target not in df.columns:
        raise DataValidationError(f"Target column '{args.target}' not found. Columns: {list(df.columns)}")
    df, excluded = exclude_columns(df, args.target, args.exclude)

    df = clean_string_cells(df)

    n_before = len(df)
    df = df.dropna(subset=[args.target]).reset_index(drop=True)
    if len(df) < n_before:
        logger.warning("Dropped %d row(s) with a missing target", n_before - len(df))

    y, mapping = encode_target(df[args.target])
    X = df.drop(columns=[args.target])

    X, dropped = drop_useless_features(X, y, args.max_categories, args.leakage_threshold)
    if X.shape[1] == 0:
        raise DataValidationError("No usable feature columns remain after cleaning")
    X = cast_numeric_strings(X)
    X = impute_missing(X)
    X = one_hot_encode(X, reserved={args.target})

    split_and_save(X, y, args.target, args.output_dir, args.test_size, args.random_state)

    meta = {"target": args.target, **mapping, "n_rows": int(len(df)),
            "n_features": int(X.shape[1]), "excluded_columns": excluded, "dropped_columns": dropped}
    (args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    logger.info("=== Stage 1: Preprocessing finished ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("Preprocessing failed: %s", exc)
        logger.debug("Traceback", exc_info=True)
        sys.exit(1)
