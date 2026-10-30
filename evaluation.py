"""
evaluation.py
─────────────────────────────────────────────
STEP 7 — Results Comparison & Evaluation

Receives the `results` dict produced by
model_training.py and prints a full metric
table, crowns the best model, and returns
the summary DataFrame for visualisation.py
─────────────────────────────────────────────
"""

import pandas as pd


def build_summary(results: dict) -> pd.DataFrame:
    """Build a tidy metrics DataFrame from the results dict."""
    summary = pd.DataFrame(
        {
            name: {
                "MAE (PKR M)": res["MAE"] / 1e6,
                "MSE": res["MSE"],
                "RMSE (PKR M)": res["RMSE"] / 1e6,
                "R²": res["R2"],
            }
            for name, res in results.items()
        }
    ).T.round(4)
    return summary


def print_summary(results: dict) -> pd.DataFrame:
    """Print the full comparison table and return it."""
    print("\n" + "=" * 60)
    print("  STEP 7 — FULL RESULTS COMPARISON")
    print("=" * 60)

    summary = build_summary(results)

    print(f"\n{summary.to_string()}")

    best = summary["R²"].idxmax()
    print(f"\n{'─'*60}")
    print(f"🏆  Best Model  :  {best}")
    print(f"    R²          :  {summary.loc[best, 'R²']:.4f}")
    print(f"    MAE         :  {summary.loc[best, 'MAE (PKR M)']:.2f} M PKR")
    print(f"    RMSE        :  {summary.loc[best, 'RMSE (PKR M)']:.2f} M PKR")
    print(f"{'─'*60}")

    return summary


# ── Run standalone ────────────────────────
if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import clean_data, encode_features
    from model_training import (
        split_data,
        train_individual_models,
        train_ensemble_models,
    )

    X, y, _ = encode_features(clean_data(load_data()))
    splits = split_data(X, y)
    results = train_individual_models(*splits)
    results = train_ensemble_models(splits[0], splits[1], splits[2], splits[3], results)
    print_summary(results)
