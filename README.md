# 🏠 PriceIQ Pakistan — House Price Prediction System

An end-to-end Machine Learning pipeline and interactive Flask web application for predicting residential property prices across major metropolitan cities in Pakistan. Built on historical real estate listings from Zameen.com, the system features automated data preprocessing, multi-model benchmarking (including ensemble architectures), diagnostic visualizations, and a real-time web interface.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Key Features](#-key-features)
- [Dataset & Preprocessing](#-dataset--preprocessing)
- [Machine Learning Models](#-machine-learning-models)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
  - [1. Running the Full Pipeline](#1-running-the-full-pipeline)
  - [2. Launching the Web Application](#2-launching-the-web-application)
  - [3. Standalone CLI Inference](#3-standalone-cli-inference)
- [Web Application & API](#-web-application--api)
- [Visualizations & Diagnostics](#-visualizations--diagnostics)
- [Tech Stack](#-tech-stack)

---

## 🌟 Overview

Real estate valuation in emerging markets like Pakistan is often subjective and volatile. **PriceIQ Pakistan** provides an objective, data-driven approach to estimate residential property valuations based on key structural and geographical parameters:

- **Target Geographic Coverage:** Lahore, Karachi, Islamabad, Rawalpindi, and Faisalabad.
- **Property Types:** Houses, Flats, Farm Houses, Penthouses, Upper Portions, Lower Portions, and Individual Rooms.
- **Ensemble ML Core:** Leverages a **Voting Regressor** (Random Forest + Gradient Boosting) alongside a **Stacking Regressor** (with a Ridge meta-learner) to minimize prediction error and guard against overfitting.
- **Production-Ready Web App:** A dark-themed, responsive dashboard featuring an interactive price calculator, an exploratory analytics hub, and a paginated real estate dataset explorer.

---

## 🏗️ System Architecture

```text
Raw Data (Zameen.com CSV)
        │
        ▼
[data_loader.py] ────────► Ingestion & Null Value Diagnostics
        │
        ▼
[preprocessing.py] ──────► Outlier Filtering & Unit Normalization (Marla)
                         ► Categorical Feature Encoding (LabelEncoder)
        │
        ▼
[model_training.py] ─────► 80/20 Train-Test Split & Feature Scaling
                         ► Individual Models: Ridge, Random Forest, Gradient Boosting
                         ► Ensemble Models: Voting & Stacking Regressors
        │
        ├────────────────────────────┬────────────────────────────┐
        ▼                            ▼                            ▼
[evaluation.py]              [visualisation.py]            [models/*.pkl]
Metrics Table (MAE, RMSE, R²) Diagnostic Plots (01 to 05)  voting_model.pkl & encoders
                                                                  │
                                                                  ▼
                                                      [app.py / predict.py]
                                                      Flask UI & REST API
```

---

## 🚀 Key Features

1. **Rigorous Data Cleaning Pipeline:**
   - Filters strictly for **"For Sale"** properties.
   - Outlier protection using **percentile clipping** (retains 1st to 99th percentile prices).
   - Area standardization: normalized to the **Marla** unit (1 Marla ≈ 225–272 sq ft depending on region).
   - Realistic constraints enforced on bedroom counts ($1 \le \text{bedrooms} \le 15$) and bathroom counts ($1 \le \text{baths} \le 15$).

2. **Benchmarked Machine Learning Suite:**
   - **Baseline:** Scaled Ridge Regression ($L_2$ regularization with $\alpha=10.0$).
   - **Ensemble Bagging:** Random Forest Regressor ($100$ estimators, $\text{max\_depth}=15$).
   - **Ensemble Boosting:** Gradient Boosting Regressor ($200$ estimators, $\text{learning\_rate}=0.1$).
   - **Voting Ensemble:** Combines predictions from Random Forest and Gradient Boosting.
   - **Stacking Ensemble:** Employs Random Forest and Gradient Boosting as base estimators with a Ridge Regression meta-estimator ($3$-fold cross-validation).

3. **Automated Visual Analytics Suite:**
   - Generates high-resolution diagnostic charts: Exploratory Data Analysis, Model Comparisons, Actual vs. Predicted scatter plots, Feature Importances, and Residual Distributions.

4. **Rich Interactive Web Dashboard:**
   - **Price Estimator:** Real-time property valuation with an empirical $\pm 15\%$ confidence band.
   - **Dynamic Location Autocomplete:** Automatically updates top residential areas and societies based on the selected city.
   - **Dataset Explorer:** Search and filter through thousands of cleaned properties with pagination.
   - **Analytics Gallery:** Embedded model evaluation plots and exploratory insights.

---

## 📊 Dataset & Preprocessing

The underlying dataset contains **168,000+ listings** scraped from Pakistan's premier property portal, **Zameen.com**.

### Feature Space

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `property_type` | Categorical | House, Flat, Farm House, Upper/Lower Portion, Penthouse, Room |
| `city` | Categorical | Islamabad, Rawalpindi, Lahore, Karachi, Faisalabad |
| `province_name` | Categorical | Punjab, Sindh, Islamabad Capital |
| `location` | Categorical | Specific sector, society, or phase (e.g., DHA Defence, Bahria Town, Gulberg) |
| `baths` | Numerical | Number of bathrooms ($1$ to $15$) |
| `bedrooms` | Numerical | Number of bedrooms ($1$ to $15$) |
| `Area Size` | Numerical | Area expressed in standard Marla units ($0 < \text{Area} < 1000$) |
| `Area Category` | Categorical | Binned area brackets (`0-5 Marla`, `5-10 Marla`, `10-15 Marla`, `15-20 Marla`) |
| `latitude` | Numerical | Geographic latitude coordinate |
| `longitude` | Numerical | Geographic longitude coordinate |
| **`price`** | **Target** | Listing price in Pakistani Rupees (**PKR**) |

### Feature Engineering & Transformation
- **Categorical Encoding:** `LabelEncoder` encodes discrete variables into numerical representations. Encoders are serialized into `models/encoders.pkl` to guarantee identical transformations during online inference.
- **Normalization:** `StandardScaler` standardizes continuous features for linear algorithms (Ridge Regression). Tree-based ensembles operate directly on unscaled features to preserve natural feature splits.
- **Spatial Fallbacks:** A city-level median coordinates dictionary (`CITY_COORDS`) supplies latitude and longitude if specific sub-locality coordinates are omitted.

---

## 🧠 Machine Learning Models

The training pipeline evaluates 5 distinct regression models to identify the top-performing architecture:

1. **Ridge Regression:**
   - Linear baseline incorporating $L_2$ regularization ($\alpha = 10.0$) on standardized inputs.
2. **Random Forest Regressor:**
   - Non-linear ensemble of 100 decision trees utilizing bootstrap aggregation and feature subspace sampling ($\text{max\_depth} = 15$).
3. **Gradient Boosting Regressor:**
   - Sequential boosting optimizing squared error loss with learning rate shrinkage ($\eta = 0.1$, $n = 200$).
4. **Voting Regressor (Selected for Production):**
   - Combines predictions of Random Forest and Gradient Boosting by computing the arithmetic mean, reducing individual model variance.
5. **Stacking Regressor:**
   - Multi-layer architecture utilizing out-of-fold predictions from base learners (RF & GB) to train a Ridge meta-regressor ($3$-fold CV).

### Evaluation Metrics
Performance is measured on an independent 20% holdout test set using:
- **MAE (Mean Absolute Error):** Average magnitude of errors in PKR Million.
- **RMSE (Root Mean Squared Error):** Penalizes larger valuation errors in PKR Million.
- **$R^2$ (Coefficient of Determination):** Proportion of price variance explained by the model features.

---

## 📁 Project Structure

```text
House price prediction project/
│
├── House_Price_dataset.csv     # Raw dataset from Zameen.com (~168k rows)
├── config.py                   # Central configuration, hyperparams, & paths
├── data_loader.py              # Step 1: Raw data loading & summary stats
├── preprocessing.py            # Steps 2 & 3: Outlier cleaning & label encoding
├── model_training.py           # Steps 4, 5 & 6: Data splitting & model training
├── evaluation.py               # Step 7: Performance metrics computation & tables
├── visualisation.py            # Step 8: Plot generation for EDA & diagnostics
├── main.py                     # Master runner script orchestrating the pipeline
│
├── predict.py                  # Inference module for standalone & web predictions
├── app.py                      # Production Flask web application
│
├── models/                     # Saved artifacts for inference
│   ├── voting_model.pkl        # Serialized Voting Ensemble regressor
│   ├── encoders.pkl            # LabelEncoders for all categorical features
│   └── feature_cols.pkl        # Ordered list of training feature names
│
├── outputs/                    # Exported high-resolution figures
│   ├── 01_EDA.png              # Exploratory data distributions & correlations
│   ├── 02_Model_Comparison.png # Comparative bar charts of MAE, RMSE, & R²
│   ├── 03_Actual_vs_Predicted.png # Goodness-of-fit scatter plots
│   ├── 04_Feature_Importance.png  # Relative feature importances (Gradient Boosting)
│   └── 05_Residual_Analysis.png   # Error distribution & residual plots
│
├── templates/
│   └── index.html              # Frontend dashboard, calculator, & dataset viewer
│
├── .gitignore                  # Git ignore rules for virtualenvs and caches
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9+ (Python 3.10 / 3.11 / 3.12 recommended)
- `pip` package manager
- (Optional) `virtualenv` or `venv`

### Step 1: Clone the Repository
```bash
git clone https://github.com/avan-de-haris/-House-Price-Prediction-.git
cd "-House-Price-Prediction-"
```

### Step 2: Create and Activate a Virtual Environment
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- **Windows (Command Prompt / PowerShell):**
  ```bat
  python -m venv .venv
  .venv\Scripts\activate
  ```

### Step 3: Install Required Dependencies
Install the required packages using `pip`:
```bash
pip install flask pandas numpy scikit-learn matplotlib seaborn
```

---

## 💻 Usage Guide

### 1. Running the Full Pipeline
To execute data ingestion, cleaning, feature engineering, model training, evaluation, visualization generation, and artifact serialization in one automated step:

```bash
python3 main.py
```

**Pipeline Execution Flow:**
1. Loads `House_Price_dataset.csv`.
2. Cleans records and removes price and area outliers.
3. Encodes categorical variables and creates feature matrix $X$ and target $y$.
4. Performs an 80/20 train/test split.
5. Fits **Ridge**, **Random Forest**, **Gradient Boosting**, **Voting**, and **Stacking** models.
6. Prints a comparative evaluation table.
7. Saves all 5 analytical figures to `outputs/`.
8. Serializes `voting_model.pkl`, `encoders.pkl`, and `feature_cols.pkl` into `models/`.

### 2. Launching the Web Application
Once models are trained and saved, launch the Flask web server:

```bash
python3 app.py
```
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

### 3. Standalone CLI Inference
You can verify the prediction engine directly via the terminal:

```bash
python3 predict.py
```
This executes built-in test fixtures for properties in Lahore, Karachi, and Islamabad, displaying formatted predictions and confidence bands.

---

## 🌐 Web Application & API

The Flask application (`app.py`) provides both a web interface and RESTful JSON endpoints:

### Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Renders the primary single-page application dashboard. |
| `POST` | `/predict` | Predicts price for a property specification payload. |
| `GET` | `/dataset` | Returns paginated JSON records from the cleaned dataset. |
| `GET` | `/outputs/<filename>` | Serves generated visualization images to the dashboard. |

### Prediction API Specification

**Request:**
`POST /predict`
```json
{
  "property_type": "House",
  "city": "Lahore",
  "province_name": "Punjab",
  "location": "DHA Defence",
  "area_size": 10,
  "area_category": "10-15 Marla",
  "bedrooms": 4,
  "baths": 4
}
```

**Response:**
```json
{
  "success": true,
  "predicted_price": 42500000.0,
  "formatted": "PKR 42,500,000",
  "in_millions": "42.50 Million",
  "confidence_low": "PKR 36,125,000",
  "confidence_high": "PKR 48,875,000"
}
```

---

## 📈 Visualizations & Diagnostics

All charts are generated during pipeline execution and saved directly to the `outputs/` folder:

- **`01_EDA.png`**: Multi-panel exploratory analysis showing price distribution, median prices by city, bedroom count relationships, property type breakdowns, area size vs. price scatter, and a feature correlation matrix.
- **`02_Model_Comparison.png`**: Direct horizontal comparison of MAE, RMSE, and $R^2$ across all candidate algorithms.
- **`03_Actual_vs_Predicted.png`**: Scatter plots mapping ground-truth test prices against predictions with a reference line for perfect fit.
- **`04_Feature_Importance.png`**: Ranking of input features based on Gini impurity / gradient split criteria from the Gradient Boosting model.
- **`05_Residual_Analysis.png`**: Residual error histogram and residual vs. predicted plots verifying error normality and homoscedasticity.

---

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **Machine Learning & Data Science:** Scikit-Learn, Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Backend Framework:** Flask (Jinja2 Templates)
- **Frontend:** Semantic HTML5, Vanilla Modern CSS (Dark Theme, Glassmorphism, Syne & DM Sans typography), Vanilla JavaScript (Asynchronous Fetch API)
- **Model Serialization:** Python Pickle

---

## 📄 License & Attribution

This project is developed for educational and research purposes using publicly available Pakistani real estate listing data from Zameen.com. All trademarks and listings belong to their respective owners.
