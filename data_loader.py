"""
data_loader.py
─────────────────────────────────────────────
STEP 1 — Load the raw dataset from disk and
print a quick summary. Returns a raw DataFrame
that preprocessing.py will clean.
─────────────────────────────────────────────
"""

import pandas as pd
from config import DATASET_PATH


def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load raw CSV dataset and return a DataFrame."""
    print("\n" + "=" * 60)
    print("  STEP 1 — LOADING DATASET")
    print("=" * 60)

    df = pd.read_csv(path)

    print(f"✔  Dataset loaded  →  {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns:\n  {df.columns.tolist()}")
    print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
    print(f"\nMissing values per column:")
    print(df.isnull().sum().to_string())

    return df


# ── Run standalone ────────────────────────
if __name__ == "__main__":
    df_raw = load_data()
    print(f"\nDone. Shape: {df_raw.shape}")
