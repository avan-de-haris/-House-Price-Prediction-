"""
config.py
─────────────────────────────────────────────
Central configuration for the House Price
Prediction project. All other modules import
from here so you only ever change one file.
─────────────────────────────────────────────
"""

import os

# ── Paths ─────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = "House_Price_dataset.csv"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data cleaning thresholds ──────────────
PURPOSE_FILTER = "For Sale"
PRICE_LOW_PCT = 0.01  # remove bottom 1 %
PRICE_HIGH_PCT = 0.99  # remove top 1 %
MAX_AREA_SIZE = 1000  # marlas
AREA_TYPE_FILTER = "Marla"
MIN_BEDROOMS = 1
MAX_BEDROOMS = 15
MIN_BATHS = 1
MAX_BATHS = 15

COLUMNS_TO_DROP = [
    "agency",
    "agent",
    "page_url",
    "date_added",
    "property_id",
    "location_id",
    "area",
    "purpose",
    "Area Type",
]

# ── Features & target ─────────────────────
CATEGORICAL_COLS = [
    "property_type",
    "city",
    "province_name",
    "Area Category",
    "location",
]

FEATURE_COLS = [
    "property_type",
    "city",
    "province_name",
    "baths",
    "Area Size",
    "Area Category",
    "bedrooms",
    "latitude",
    "longitude",
    "location",
]

TARGET_COL = "price"

# ── Model hyper-parameters ────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
# Random forest
RF_PARAMS = dict(n_estimators=100, max_depth=15, random_state=RANDOM_STATE, n_jobs=-1)
# Gradient boosting
GB_PARAMS = dict(
    n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE
)
# Gradient boosting (Stacking Model)
GB_STACK_PARAMS = dict(
    n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE
)
# Ridge Regression
RIDGE_ALPHA = 10.0
STACK_CV = 3

# ── Visualisation colours ─────────────────
PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#937860"]
