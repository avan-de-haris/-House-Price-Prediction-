"""
main.py
─────────────────────────────────────────────
MASTER RUNNER — House Price Prediction Project

Imports and calls every module in order:
  1. data_loader.py     → load raw data
  2. preprocessing.py   → clean + encode
  3. model_training.py  → split + train all 5 models
  4. evaluation.py      → metrics table
  5. visualisation.py   → save all charts
  6. Saves trained model + encoders for prediction app

Run this single file to execute the full pipeline:
  $ python3 main.py
─────────────────────────────────────────────
"""

import warnings, pickle, os

warnings.filterwarnings("ignore")

from data_loader import load_data
from preprocessing import clean_data, encode_features
from model_training import split_data, train_individual_models, train_ensemble_models
from evaluation import print_summary
from visualisation import generate_all
from sklearn.preprocessing import LabelEncoder
from config import CATEGORICAL_COLS, FEATURE_COLS


def run_pipeline():
    print("\n" + "★" * 60)
    print("  HOUSE PRICE PREDICTION — AI Final Semester Project")
    print("  Dataset : Pakistan Real Estate (Zameen.com)")
    print("  Models  : Ridge  |  RF  |  GB  |  Voting  |  Stacking")
    print("★" * 60)

    # Step 1 — Load
    df_raw = load_data()

    # Step 2 & 3 — Preprocess + Feature engineering
    df_clean = clean_data(df_raw)
    X, y, df_encoded = encode_features(df_clean)

    # Step 4 — Split
    X_train, X_test, y_train, y_test, X_train_sc, X_test_sc = split_data(X, y)

    # Step 5 — Individual models
    results = train_individual_models(
        X_train, X_test, y_train, y_test, X_train_sc, X_test_sc
    )

    # Step 6 — Ensemble models
    results = train_ensemble_models(X_train, X_test, y_train, y_test, results)

    # Step 7 — Evaluation table
    summary = print_summary(results)

    # Step 8 — Visualisations
    generate_all(df_clean, results, y_test)

    # ── Step 9: Save best model + encoders for prediction app ──
    print("\n" + "=" * 60)
    print("  STEP 9 — SAVING MODEL FOR PREDICTION APP")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)

    # Re-fit encoders on clean data (so we can reuse them in predict.py)
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(df_clean[col].astype(str))
        encoders[col] = le

    best_model = results["VotingEnsemble"]["model"]

    with open("models/voting_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)
    with open("models/feature_cols.pkl", "wb") as f:
        pickle.dump(FEATURE_COLS, f)

    print("✔  models/voting_model.pkl  saved")
    print("✔  models/encoders.pkl      saved")
    print("✔  models/feature_cols.pkl  saved")
    print("\n  ➜  Now run:  python3 app.py")
    print("  ➜  Then open: http://127.0.0.1:5000")

    print("\n" + "★" * 60)
    print("  ✅  PIPELINE COMPLETE")
    print(
        f"  Best model : {summary['R²'].idxmax()}"
        f"  →  R² = {summary['R²'].max():.4f}"
    )
    print("★" * 60 + "\n")


if __name__ == "__main__":
    run_pipeline()
