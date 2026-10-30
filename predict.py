"""
predict.py
─────────────────────────────────────────────
Prediction engine — loads the saved Voting
Ensemble model + encoders once at import time,
then exposes predict_price(data_dict).

Called by app.py for every web request.
─────────────────────────────────────────────
"""

import pickle, os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Load saved artefacts ──────────────────────────────────
with open(os.path.join(BASE, "models/voting_model.pkl"), "rb") as f:
    MODEL = pickle.load(f)
with open(os.path.join(BASE, "models/encoders.pkl"), "rb") as f:
    ENCODERS = pickle.load(f)
with open(os.path.join(BASE, "models/feature_cols.pkl"), "rb") as f:
    FEATURE_COLS = pickle.load(f)

# ── City default coordinates (median from dataset) ────────
CITY_COORDS = {
    "Faisalabad": (31.4260, 73.1081),
    "Karachi": (24.9266, 67.0831),
    "Lahore": (31.4637, 74.2925),
    "Islamabad": (33.6464, 73.0745),
    "Rawalpindi": (33.5564, 73.0839),
}


# ── Helper ────────────────────────────────────────────────
def _encode(col, value):
    """Encode one categorical value; unknown → 0."""
    le = ENCODERS[col]
    try:
        return int(le.transform([str(value)])[0])
    except ValueError:
        return 0


# ── Main function called by app.py ────────────────────────
def predict_price(data: dict) -> dict:
    """
    Input keys:
      property_type, city, province_name,
      baths, area_size, area_category,
      bedrooms, location

    Returns dict with formatted price + confidence band.
    """
    city = data.get("city", "Lahore")
    lat, lng = CITY_COORDS.get(city, (31.46, 74.29))

    row = {
        "property_type": _encode("property_type", data.get("property_type", "House")),
        "city": _encode("city", city),
        "province_name": _encode("province_name", data.get("province_name", "Punjab")),
        "baths": float(data.get("baths", 2)),
        "Area Size": float(data.get("area_size", 5)),
        "Area Category": _encode(
            "Area Category", data.get("area_category", "0-5 Marla")
        ),
        "bedrooms": float(data.get("bedrooms", 3)),
        "latitude": lat,
        "longitude": lng,
        "location": _encode("location", data.get("location", "DHA Defence")),
    }

    X = pd.DataFrame([row])[FEATURE_COLS]
    price = float(MODEL.predict(X)[0])
    price = max(price, 100_000)  # safety floor

    return {
        "predicted_price": price,
        "formatted": f"PKR {price:,.0f}",
        "confidence_low": f"PKR {price * 0.85:,.0f}",
        "confidence_high": f"PKR {price * 1.15:,.0f}",
        "in_millions": f"{price / 1_000_000:.2f} Million",
    }


# ── Quick self-test ───────────────────────────────────────
if __name__ == "__main__":
    tests = [
        {
            "property_type": "House",
            "city": "Lahore",
            "province_name": "Punjab",
            "baths": 3,
            "area_size": 5,
            "area_category": "0-5 Marla",
            "bedrooms": 3,
            "location": "DHA Defence",
        },
        {
            "property_type": "Flat",
            "city": "Karachi",
            "province_name": "Sindh",
            "baths": 2,
            "area_size": 4,
            "area_category": "0-5 Marla",
            "bedrooms": 2,
            "location": "Bahria Town Karachi",
        },
        {
            "property_type": "House",
            "city": "Islamabad",
            "province_name": "Islamabad Capital",
            "baths": 5,
            "area_size": 10,
            "area_category": "10-15 Marla",
            "bedrooms": 5,
            "location": "DHA Defence",
        },
        {
            "property_type": "Penthouse",
            "city": "Lahore",
            "province_name": "Punjab",
            "baths": 4,
            "area_size": 8,
            "area_category": "5-10 Marla",
            "bedrooms": 4,
            "location": "Bahria Town",
        },
    ]
    print("\n  Self-test predictions:\n")
    for t in tests:
        r = predict_price(t)
        print(
            f"  {t['property_type']:12} | {t['city']:10} | "
            f"{t['area_size']} Marla | {t['bedrooms']} bed  →  "
            f"{r['formatted']}  ({r['in_millions']})"
        )
