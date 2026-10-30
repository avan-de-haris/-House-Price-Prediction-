"""
model_training.py
─────────────────────────────────────────────
STEP 4 — Train / Test Split
STEP 5 — Individual Models
          • Ridge Regression (baseline)
          • Random Forest
          • Gradient Boosting
STEP 6 — Ensemble Learning
          • Voting Regressor  (RF + GB)
          • Stacking Regressor (RF + GB → Ridge meta)

Returns a `results` dict ready for evaluation.py
─────────────────────────────────────────────
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import (
    RANDOM_STATE,
    TEST_SIZE,
    RF_PARAMS,
    GB_PARAMS,
    GB_STACK_PARAMS,
    RIDGE_ALPHA,
    STACK_CV,
)

# ── STEP 4: Split ─────────────────────────────────────────


def split_data(X, y):
    """80/20 stratified split + StandardScaler for Ridge."""
    print("\n" + "=" * 60)
    print("  STEP 4 — TRAIN / TEST SPLIT  (80 / 20)")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print(f"✔  Training samples : {X_train.shape[0]:,}")
    print(f"✔  Testing  samples : {X_test.shape[0]:,}")

    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc


# ── Helper: fit + score one model ─────────────────────────


def _fit_and_score(name, model, X_tr, y_tr, X_te, y_te) -> dict:
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mae = mean_absolute_error(y_te, preds)
    mse = mean_squared_error(y_te, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_te, preds)
    print(f"\n  [{name}]")
    print(f"    MAE  : {mae:>15,.0f} PKR")
    print(f"    MSE  : {mse:>15,.0f}")
    print(f"    RMSE : {rmse:>15,.0f} PKR")
    print(f"    R²   : {r2:.4f}")
    return dict(model=model, preds=preds, MAE=mae, MSE=mse, RMSE=rmse, R2=r2, name=name)


# ── STEP 5: Individual models ──────────────────────────────


def train_individual_models(
    X_train, X_test, y_train, y_test, X_train_sc, X_test_sc
) -> dict:
    print("\n" + "=" * 60)
    print("  STEP 5 — INDIVIDUAL MODEL TRAINING")
    print("=" * 60)

    results = {}

    # Model 1 — Ridge Regression (baseline, uses scaled data)
    print("\n--- Algorithm 1: Ridge Regression (Baseline) ---")
    results["Ridge"] = _fit_and_score(
        "Ridge Regression",
        Ridge(alpha=RIDGE_ALPHA),
        X_train_sc,
        y_train,
        X_test_sc,
        y_test,
    )

    # Model 2 — Random Forest
    print("\n--- Algorithm 2: Random Forest ---")
    results["RandomForest"] = _fit_and_score(
        "Random Forest",
        RandomForestRegressor(**RF_PARAMS),
        X_train,
        y_train,
        X_test,
        y_test,
    )

    # Model 3 — Gradient Boosting
    print("\n--- Algorithm 3: Gradient Boosting ---")
    results["GradientBoosting"] = _fit_and_score(
        "Gradient Boosting",
        GradientBoostingRegressor(**GB_PARAMS),
        X_train,
        y_train,
        X_test,
        y_test,
    )

    return results


# ── STEP 6: Ensemble Learning ─────────────────────────────


def train_ensemble_models(X_train, X_test, y_train, y_test, results: dict) -> dict:
    print("\n" + "=" * 60)
    print("  STEP 6 — ENSEMBLE LEARNING")
    print("=" * 60)

    # Ensemble 1 — Voting Regressor (simple average of RF + GB)
    print("\n--- Ensemble 1: Voting Regressor (RF + GB) ---")
    voting = VotingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(**RF_PARAMS)),
            ("gb", GradientBoostingRegressor(**GB_PARAMS)),
        ]
    )
    results["VotingEnsemble"] = _fit_and_score(
        "Voting Ensemble", voting, X_train, y_train, X_test, y_test
    )

    # Ensemble 2 — Stacking Regressor (RF + GB → Ridge meta-learner)
    print("\n--- Ensemble 2: Stacking Regressor ---")
    print("  Base models  : Random Forest  +  Gradient Boosting")
    print("  Meta-learner : Ridge Regression")
    stacking = StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(**RF_PARAMS)),
            ("gb", GradientBoostingRegressor(**GB_STACK_PARAMS)),
        ],
        final_estimator=Ridge(alpha=RIDGE_ALPHA),
        cv=STACK_CV,
    )
    results["StackingEnsemble"] = _fit_and_score(
        "Stacking Ensemble", stacking, X_train, y_train, X_test, y_test
    )

    return results


# ── Run standalone ────────────────────────
if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import clean_data, encode_features

    X, y, _ = encode_features(clean_data(load_data()))
    X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc = split_data(X, y)

    results = train_individual_models(
        X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc
    )
    results = train_ensemble_models(X_train, X_test, y_train, y_test, results)

    print(f"\nModels trained: {list(results.keys())}")
