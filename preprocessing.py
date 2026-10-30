"""
preprocessing.py
─────────────────────────────────────────────
STEP 2 — Data Cleaning
STEP 3 — Feature Engineering

Takes the raw DataFrame from data_loader.py,
applies all cleaning rules from config.py,
label-encodes categorical columns, and returns
X (features), y (target), plus the cleaned df.
─────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from config import (
    PURPOSE_FILTER,
    PRICE_LOW_PCT,
    PRICE_HIGH_PCT,
    MAX_AREA_SIZE,
    AREA_TYPE_FILTER,
    MIN_BEDROOMS,
    MAX_BEDROOMS,
    MIN_BATHS,
    MAX_BATHS,
    COLUMNS_TO_DROP,
    CATEGORICAL_COLS,
    FEATURE_COLS,
    TARGET_COL,
)

# ── STEP 2: Cleaning ──────────────────────────────────────


def clean_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning rules and return a tidy DataFrame."""
    print("\n" + "=" * 60)
    print("  STEP 2 — DATA PREPROCESSING & CLEANING")
    print("=" * 60)

    df = df_raw.copy()

    # 2a. Keep only 'For Sale' listings
    df = df[df["purpose"] == PURPOSE_FILTER]
    print(f"✔  Filtered to '{PURPOSE_FILTER}': {len(df):,} rows")

    # 2b. Remove extreme prices (outliers via percentile clipping)
    q_low = df["price"].quantile(PRICE_LOW_PCT)
    q_high = df["price"].quantile(PRICE_HIGH_PCT)
    df = df[(df["price"] > q_low) & (df["price"] <= q_high)]
    print(f"✔  Removed price outliers (1st–99th pct): {len(df):,} rows")

    # 2c. Remove zero / extreme area sizes
    df = df[(df["Area Size"] > 0) & (df["Area Size"] < MAX_AREA_SIZE)]
    print(f"✔  Removed invalid Area Size: {len(df):,} rows")

    # 2d. Keep only Marla area type for unit uniformity
    df = df[df["Area Type"] == AREA_TYPE_FILTER]
    print(f"✔  Kept Area Type == '{AREA_TYPE_FILTER}': {len(df):,} rows")

    # 2e. Reasonable bedroom / bath counts
    df = df[(df["bedrooms"] >= MIN_BEDROOMS) & (df["bedrooms"] <= MAX_BEDROOMS)]
    df = df[(df["baths"] >= MIN_BATHS) & (df["baths"] <= MAX_BATHS)]
    print(f"✔  Valid bedrooms & baths: {len(df):,} rows")

    # 2f. Drop columns not useful for modelling
    df.drop(columns=COLUMNS_TO_DROP, inplace=True, errors="ignore")
    print(f"✔  Dropped irrelevant columns")

    # 2g. Drop any remaining nulls
    before = len(df)
    df.dropna(inplace=True)
    print(f"✔  Dropped {before - len(df)} rows with nulls")

    print(f"\n✔  Final clean shape: {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"   Remaining columns: {df.columns.tolist()}")
    return df


# ── STEP 3: Feature Engineering ──────────────────────────


def encode_features(df_clean: pd.DataFrame):
    """
    Label-encode categorical columns and return:
      X (feature matrix), y (target series), df_encoded
    """
    print("\n" + "=" * 60)
    print("  STEP 3 — FEATURE ENGINEERING")
    print("=" * 60)

    df_enc = df_clean.copy()
    le = LabelEncoder()

    for col in CATEGORICAL_COLS:
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        print(f"  ✔  Label-encoded: {col}")

    X = df_enc[FEATURE_COLS]
    y = df_enc[TARGET_COL]

    print(f"\n✔  Feature matrix shape : {X.shape}")
    print(f"✔  Target vector  shape : {y.shape}")
    print(f"\nFeature statistics:\n{X.describe().round(2)}")

    return X, y, df_enc


# ── Run standalone ────────────────────────
if __name__ == "__main__":
    from data_loader import load_data

    df_raw = load_data()
    df_clean = clean_data(df_raw)
    X, y, _ = encode_features(df_clean)
    print(f"\nDone.  X: {X.shape}  y: {y.shape}")
