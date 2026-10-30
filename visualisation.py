"""
visualisation.py
─────────────────────────────────────────────
STEP 8 — Generate & Save All Charts

Charts produced:
  01_EDA.png               — Exploratory data analysis
  02_Model_Comparison.png  — MAE / RMSE / R² bar charts
  03_Actual_vs_Predicted.png
  04_Feature_Importance.png
  05_Residual_Analysis.png
─────────────────────────────────────────────
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUT_DIR, PALETTE, FEATURE_COLS
from evaluation import build_summary

plt.style.use("seaborn-v0_8-whitegrid")


# ── Fig 1: EDA ────────────────────────────────────────────


def plot_eda(df_clean: pd.DataFrame, save: bool = True) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "House Price Prediction — Exploratory Data Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # Price distribution
    axes[0, 0].hist(
        df_clean["price"] / 1e6,
        bins=60,
        color=PALETTE[0],
        edgecolor="white",
        alpha=0.85,
    )
    axes[0, 0].set_title("Price Distribution (PKR Million)")
    axes[0, 0].set_xlabel("Price (Millions)")
    axes[0, 0].set_ylabel("Count")

    # Median price by city
    city_avg = (
        df_clean.groupby("city")["price"].median().sort_values(ascending=False).head(8)
    )
    axes[0, 1].bar(city_avg.index, city_avg.values / 1e6, color=PALETTE[1], alpha=0.85)
    axes[0, 1].set_title("Median Price by City")
    axes[0, 1].set_xlabel("City")
    axes[0, 1].set_ylabel("Median Price (M PKR)")
    axes[0, 1].tick_params(axis="x", rotation=30)

    # Bedrooms vs median price
    bed_avg = df_clean.groupby("bedrooms")["price"].median().reset_index()
    axes[0, 2].plot(
        bed_avg["bedrooms"], bed_avg["price"] / 1e6, marker="o", color=PALETTE[2]
    )
    axes[0, 2].set_title("Bedrooms vs Median Price")
    axes[0, 2].set_xlabel("Bedrooms")
    axes[0, 2].set_ylabel("Median Price (M PKR)")

    # Property type distribution
    pt = df_clean["property_type"].value_counts()
    axes[1, 0].bar(pt.index, pt.values, color=PALETTE[3], alpha=0.85)
    axes[1, 0].set_title("Count by Property Type")
    axes[1, 0].set_xlabel("Property Type")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].tick_params(axis="x", rotation=20)

    # Area size vs price
    samp = df_clean.sample(min(3000, len(df_clean)), random_state=42)
    axes[1, 1].scatter(
        samp["Area Size"], samp["price"] / 1e6, alpha=0.3, s=8, color=PALETTE[4]
    )
    axes[1, 1].set_title("Area Size (Marla) vs Price")
    axes[1, 1].set_xlabel("Area Size")
    axes[1, 1].set_ylabel("Price (M PKR)")
    axes[1, 1].set_xlim(0, 50)

    # Correlation heatmap
    num_cols = ["price", "baths", "bedrooms", "Area Size", "latitude", "longitude"]
    corr = df_clean[num_cols].corr()
    sns.heatmap(
        corr,
        ax=axes[1, 2],
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
    )
    axes[1, 2].set_title("Correlation Heatmap")

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "01_EDA.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✔  Saved: {path}")
    plt.close()


# ── Fig 2: Model comparison ───────────────────────────────


def plot_model_comparison(summary: pd.DataFrame, save: bool = True) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Model Comparison — Evaluation Metrics", fontsize=15, fontweight="bold"
    )
    names = list(summary.index)

    axes[0].barh(names, summary["MAE (PKR M)"], color=PALETTE, alpha=0.85)
    axes[0].set_title("MAE  (Lower ↓ is Better)")
    axes[0].set_xlabel("MAE (Millions PKR)")

    axes[1].barh(names, summary["RMSE (PKR M)"], color=PALETTE, alpha=0.85)
    axes[1].set_title("RMSE  (Lower ↓ is Better)")
    axes[1].set_xlabel("RMSE (Millions PKR)")

    axes[2].barh(names, summary["R²"], color=PALETTE, alpha=0.85)
    axes[2].set_title("R²  (Higher ↑ is Better)")
    axes[2].set_xlabel("R² Score")
    axes[2].set_xlim(0, 1)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "02_Model_Comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✔  Saved: {path}")
    plt.close()


# ── Fig 3: Actual vs Predicted ────────────────────────────


def plot_actual_vs_predicted(results: dict, y_test, save: bool = True) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    fig.suptitle("Actual vs Predicted Prices", fontsize=15, fontweight="bold")

    for i, (name, res) in enumerate(results.items()):
        ax = axes[i]
        ax.scatter(
            y_test / 1e6,
            res["preds"] / 1e6,
            alpha=0.25,
            s=6,
            color=PALETTE[i % len(PALETTE)],
        )
        mn = min(y_test.min(), res["preds"].min()) / 1e6
        mx = max(y_test.max(), res["preds"].max()) / 1e6
        ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
        ax.set_title(f"{name}\nR²={res['R2']:.3f}")
        ax.set_xlabel("Actual (M PKR)")
        ax.set_ylabel("Predicted (M PKR)")
        ax.legend(fontsize=7)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "03_Actual_vs_Predicted.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✔  Saved: {path}")
    plt.close()


# ── Fig 4: Feature importance ─────────────────────────────


def plot_feature_importance(results: dict, save: bool = True) -> None:
    model = results["GradientBoosting"]["model"]
    importance = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    importance = importance.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    importance.plot(kind="barh", ax=ax, color="#55A868", edgecolor="white", alpha=0.88)
    ax.set_title(
        "Feature Importances — Gradient Boosting", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Importance Score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "04_Feature_Importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✔  Saved: {path}")
    plt.close()


# ── Fig 5: Residual analysis ──────────────────────────────


def plot_residuals(results: dict, y_test, save: bool = True) -> None:
    preds = results["StackingEnsemble"]["preds"]
    residuals = y_test - preds

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Residual Analysis — Stacking Ensemble", fontsize=14, fontweight="bold"
    )

    axes[0].hist(
        residuals / 1e6, bins=60, color="#C44E52", edgecolor="white", alpha=0.80
    )
    axes[0].axvline(0, color="black", linestyle="--", lw=1.5)
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Residual (M PKR)")
    axes[0].set_ylabel("Count")

    axes[1].scatter(preds / 1e6, residuals / 1e6, alpha=0.25, s=6, color="#8172B2")
    axes[1].axhline(0, color="red", linestyle="--", lw=1.5)
    axes[1].set_title("Residuals vs Predicted")
    axes[1].set_xlabel("Predicted (M PKR)")
    axes[1].set_ylabel("Residual (M PKR)")

    plt.tight_layout()
    if save:
        path = os.path.join(OUTPUT_DIR, "05_Residual_Analysis.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✔  Saved: {path}")
    plt.close()


# ── Generate all charts ───────────────────────────────────


def generate_all(
    df_clean: pd.DataFrame, results: dict, y_test, save: bool = True
) -> None:
    print("\n" + "=" * 60)
    print("  STEP 8 — GENERATING VISUALISATIONS")
    print("=" * 60)
    summary = build_summary(results)
    plot_eda(df_clean, save)
    plot_model_comparison(summary, save)
    plot_actual_vs_predicted(results, y_test, save)
    plot_feature_importance(results, save)
    plot_residuals(results, y_test, save)
    print("\n✅  All charts saved to:", OUTPUT_DIR)


# ── Run standalone ────────────────────────
if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import clean_data, encode_features
    from model_training import (
        split_data,
        train_individual_models,
        train_ensemble_models,
    )

    df_raw = load_data()
    df_clean = clean_data(df_raw)
    X, y, _ = encode_features(df_clean)
    splits = split_data(X, y)
    results = train_individual_models(*splits)
    results = train_ensemble_models(splits[0], splits[1], splits[2], splits[3], results)
    generate_all(df_clean, results, splits[3])
