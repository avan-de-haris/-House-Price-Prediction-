"""
app.py  —  Flask web app
Run:   python3 app.py
Open:  http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os, pandas as pd
from predict import predict_price

app = Flask(__name__, template_folder="templates")

# ── Jinja2 helpers ────────────────────────────────────────
app.jinja_env.globals["enumerate"] = enumerate


@app.template_filter("format_num")
def format_num(value):
    return f"{int(value):,}"


BASE = os.path.dirname(os.path.abspath(__file__))


# ── Load & clean dataset once at startup ──────────────────
def _load_clean():
    df = pd.read_csv(os.path.join(BASE, "House_Price_dataset.csv"))
    df = df[df["purpose"] == "For Sale"].copy()
    q_lo = df["price"].quantile(0.01)
    q_hi = df["price"].quantile(0.99)
    df = df[(df["price"] > q_lo) & (df["price"] <= q_hi)]
    df = df[(df["Area Size"] > 0) & (df["Area Size"] < 1000)]
    df = df[df["Area Type"] == "Marla"]
    df = df[df["bedrooms"].between(1, 15)]
    df = df[df["baths"].between(1, 15)]
    df.drop(
        columns=[
            "agency",
            "agent",
            "page_url",
            "date_added",
            "property_id",
            "location_id",
            "area",
            "purpose",
            "Area Type",
        ],
        inplace=True,
        errors="ignore",
    )
    df.dropna(inplace=True)
    df = df.reset_index(drop=True)
    return df


DF_CLEAN = _load_clean()

# ── Dropdown data ─────────────────────────────────────────
CITIES = ["Faisalabad", "Islamabad", "Karachi", "Lahore", "Rawalpindi"]
PROPERTY_TYPES = [
    "House",
    "Flat",
    "Farm House",
    "Lower Portion",
    "Upper Portion",
    "Penthouse",
    "Room",
]
AREA_CATEGORIES = ["0-5 Marla", "5-10 Marla", "10-15 Marla", "15-20 Marla"]

CITY_PROVINCE = {
    "Faisalabad": "Punjab",
    "Islamabad": "Islamabad Capital",
    "Karachi": "Sindh",
    "Lahore": "Punjab",
    "Rawalpindi": "Punjab",
}

CITY_LOCATIONS = {
    "Lahore": [
        "DHA Defence",
        "Bahria Town",
        "Johar Town",
        "Gulberg",
        "Model Town",
        "Allama Iqbal Town",
        "Askari",
        "Township",
        "Wapda Town",
        "Garden Town",
        "Cavalry Ground",
        "Cantt",
        "Faisal Town",
        "Punjab University Society",
        "Paragon City",
        "Lake City",
        "Raiwind Road",
        "Bedian Road",
    ],
    "Karachi": [
        "DHA Defence",
        "Bahria Town Karachi",
        "Gulistan-e-Jauhar",
        "Gulshan-e-Iqbal Town",
        "Clifton",
        "PECHS",
        "North Nazimabad",
        "Nazimabad",
        "Saddar",
        "Korangi",
        "Malir",
        "Landhi",
        "Federal B Area",
        "Scheme 33",
        "Gadap Town",
    ],
    "Islamabad": [
        "DHA Defence",
        "Bahria Town",
        "G-13",
        "G-11",
        "F-7",
        "F-8",
        "F-10",
        "F-11",
        "E-11",
        "G-9",
        "Ghauri Town",
        "PWD Housing Scheme",
        "Soan Garden",
        "B-17",
        "Capital Smart City",
    ],
    "Rawalpindi": [
        "Bahria Town Rawalpindi",
        "DHA Defence",
        "Satellite Town",
        "Gulraiz Housing Scheme",
        "Airport Housing Society",
        "Adiala Road",
        "Chaklala Scheme",
        "PWD Colony",
        "Askari",
        "Morgah",
    ],
    "Faisalabad": [
        "DHA Faisalabad",
        "Canal Road",
        "Sargodha Road",
        "Peoples Colony",
        "Gulberg",
        "Railway Road",
        "Susan Road",
        "Ghulam Muhammad Abad",
        "Jinnah Colony",
        "Madina Town",
        "Samanabad",
    ],
}

# ── Dataset summary stats (computed once) ─────────────────
DS_STATS = {
    "total_rows": len(DF_CLEAN),
    "total_cols": len(DF_CLEAN.columns),
    "cities": int(DF_CLEAN["city"].nunique()),
    "prop_types": int(DF_CLEAN["property_type"].nunique()),
    "avg_price": f"PKR {DF_CLEAN['price'].mean()/1e6:.2f}M",
    "median_price": f"PKR {DF_CLEAN['price'].median()/1e6:.2f}M",
    "min_price": f"PKR {DF_CLEAN['price'].min()/1e6:.2f}M",
    "max_price": f"PKR {DF_CLEAN['price'].max()/1e6:.2f}M",
    "columns": DF_CLEAN.columns.tolist(),
    "city_counts": DF_CLEAN["city"].value_counts().to_dict(),
    "type_counts": DF_CLEAN["property_type"].value_counts().to_dict(),
    "area_counts": DF_CLEAN["Area Category"].value_counts().to_dict(),
}


# ── Routes ────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template(
        "index.html",
        cities=CITIES,
        property_types=PROPERTY_TYPES,
        area_categories=AREA_CATEGORIES,
        city_locations=CITY_LOCATIONS,
        city_province=CITY_PROVINCE,
        ds_stats=DS_STATS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        result = predict_price(data)
        return jsonify({"success": True, **result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/dataset")
def dataset():
    """Return paginated dataset rows as JSON."""
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 50))
    city = request.args.get("city", "")
    ptype = request.args.get("type", "")
    search = request.args.get("search", "").lower()

    df = DF_CLEAN.copy()

    # filters
    if city:
        df = df[df["city"] == city]
    if ptype:
        df = df[df["property_type"] == ptype]
    if search:
        df = df[df["location"].str.lower().str.contains(search, na=False)]

    total = len(df)
    start = (page - 1) * per_page
    end = start + per_page
    chunk = df.iloc[start:end].copy()

    # format price
    chunk["price_fmt"] = chunk["price"].apply(lambda p: f"PKR {p:,.0f}")

    rows = chunk[
        [
            "property_type",
            "city",
            "province_name",
            "location",
            "bedrooms",
            "baths",
            "Area Size",
            "Area Category",
            "price_fmt",
        ]
    ].to_dict("records")

    return jsonify({"rows": rows, "total": total, "page": page, "per_page": per_page})


@app.route("/outputs/<path:filename>")
def serve_outputs(filename):
    outputs_dir = os.path.join(BASE, "outputs")
    return send_from_directory(outputs_dir, filename)


if __name__ == "__main__":
    print("\n  🏠  House Price Predictor — Pakistan Real Estate")
    print("  ─────────────────────────────────────────────────")
    print(f"  Dataset loaded: {len(DF_CLEAN):,} clean rows")
    print("  Open in browser →  http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
