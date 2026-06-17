# -House-Price-Prediction-
AI-powered residential property price prediction system for Pakistan built with ensemble machine learning (Random Forest + Gradient Boosting). Trained on 77,027 Zameen.com listings across 5 cities. Achieves R² = 0.8656. Includes modular ML pipeline, 5 model comparison, and a live Flask web app.

---

## 📌 Project Overview

This system was built as a Final Semester AI project to solve a real-world
problem — the lack of a reliable, data-driven property valuation tool in
Pakistan. Buyers, sellers, and investors currently rely on guesswork and
agent estimates. This project replaces that with a trained ML model that
predicts fair market price instantly based on property characteristics.

---

## 🗄️ Dataset

- **Source:** Zameen.com (Pakistan's largest real estate platform)
- **Raw records:** 168,446 property listings
- **After cleaning:** 77,027 records
- **Cities covered:** Lahore, Karachi, Islamabad, Rawalpindi, Faisalabad
- **Property types:** House, Flat, Penthouse, Farm House, Upper Portion,
  Lower Portion, Room
- **Target variable:** Price (PKR)

---

## 🤖 Models Trained

| Model                  | MAE (PKR)  | RMSE (PKR) | R² Score |
|------------------------|------------|------------|----------|
| Ridge Regression       | 5.35M      | 9.14M      | 0.5287   |
| Random Forest          | 2.31M      | 4.97M      | 0.8610   |
| Gradient Boosting      | 2.58M      | 5.08M      | 0.8543   |
| **Voting Ensemble** 🏆 | **2.37M**  | **4.88M**  | **0.8656** |
| Stacking Ensemble      | 2.38M      | 4.93M      | 0.8628   |

Best model: **Voting Ensemble (RF + GB)** with R² = 0.8656

---

## 📁 Project Structure
house_price_predictor/

│

├── config.py            # All settings, paths, hyperparameters

├── data_loader.py       # Loads raw CSV dataset

├── preprocessing.py     # Data cleaning, encoding, train/test split

├── model_training.py    # Trains all 5 models

├── evaluation.py        # Metrics table and best model selection

├── visualisation.py     # Generates all 5 charts

├── main.py              # Master runner — executes full pipeline

├── predict.py           # Prediction engine (loads saved model)

├── app.py               # Flask web application

│

├── models/              # Saved model pickle files (auto-generated)

├── outputs/             # Charts and results (auto-generated)

├── templates/

│   └── index.html       # Web app frontend (Dashboard + Dataset + Predict)

│

└── House_Price_dataset.csv   # Dataset (place here before running)
---

## ⚙️ How to Run

### 1. Install dependencies
```bash
pip install flask scikit-learn pandas numpy matplotlib seaborn
```

### 2. Run the full ML pipeline (trains all 5 models + saves charts)
```bash
python main.py
```

### 3. Launch the web application
```bash
python app.py
```

### 4. Open in browser
http://127.0.0.1:5000
---

## 🌐 Web Application Features

The Flask web app has **3 tabs:**

- **📊 Dashboard** — Project overview, file structure, pipeline walkthrough,
  model performance cards, and all 5 generated charts
- **🗃️ Dataset** — Browse all 77,027 cleaned records with filters for city,
  property type, and location search. Paginated 50 rows at a time
- **🏠 Predict Price** — Enter property details (city, type, area, bedrooms,
  bathrooms, location) and get an instant AI-generated price estimate with
  a ±15% confidence range

---

## 📊 Visualisations Generated

| File | Description |
|------|-------------|
| `01_EDA.png` | Price distribution, city comparison, correlation heatmap |
| `02_Model_Comparison.png` | MAE, RMSE, R² side-by-side for all 5 models |
| `03_Actual_vs_Predicted.png` | Scatter plots for every model |
| `04_Feature_Importance.png` | Top features from Gradient Boosting |
| `05_Residual_Analysis.png` | Error distribution of best ensemble |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3 | Core language |
| Pandas | Data loading and cleaning |
| Scikit-learn | Model training and evaluation |
| Matplotlib / Seaborn | Visualisations |
| Flask | Web application backend |
| HTML / CSS / JavaScript | Frontend interface |

---

## 👨‍💻 Authors

- **Muhammad Haris Awan** — Roll No. B-28666
- **Talha Ahmed** — Roll No. B-28548

**Instructor:** Miss Rameesha Malik

---

## 📄 License

This project was developed for academic purposes as an 
Artificial Intelligence Semester project.
