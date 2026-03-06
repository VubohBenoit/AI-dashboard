# 📊 AI Business Analytics Dashboard

An end-to-end machine-learning dashboard built with **Python · Streamlit · Scikit-learn · Plotly**.

---

## 🗂 Project Structure

```
ai-dashboard/
│
├── app.py                          ← Entry point — run this
├── requirements.txt                ← All dependencies
├── .streamlit/
│   └── config.toml                 ← Dark theme + server config
│
└── src/
    ├── data/
    │   └── generator.py            ← Synthetic sales/customer/product data
    │
    ├── services/
    │   ├── ml_models.py            ← ML: forecasting, segmentation, recommendations, anomalies
    │   └── charts.py               ← All Plotly chart factories
    │
    ├── components/
    │   └── pages.py                ← Per-page UI renderers (Overview, Forecast, etc.)
    │
    └── utils/
        └── helpers.py              ← Formatting helpers + AI NLG insights
```

---

## 🚀 Setup & Run

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# or on Windows:
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> ⏳ This takes ~2 minutes — scikit-learn and plotly are large packages.

### 3. Run the dashboard

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 📈 Features

| Page | Description |
|---|---|
| 🏠 Overview | Revenue trends, category breakdown, channel mix, regional heatmap |
| 🔮 Sales Forecast | 90-day Gradient Boosting forecast with confidence intervals |
| 👥 Customers | KMeans RFM segmentation, churn risk, PCA scatter |
| 📦 Products | Bubble chart, top performers, content-based recommendations |
| 🔍 Anomalies | IsolationForest anomaly detection on daily sales |

---

## 🧠 ML Models

- **SalesForecaster** — `GradientBoostingRegressor` with calendar + lag features
- **CustomerSegmenter** — `KMeans` with auto k-selection via silhouette score
- **ProductRecommender** — Cosine similarity on normalized product attributes
- **AnomalyDetector** — `IsolationForest` on daily revenue/volume/transactions

---

## 🛠 Tech Stack

- **Streamlit** — UI framework
- **Scikit-learn** — ML models
- **Plotly** — Interactive charts
- **Pandas / NumPy** — Data manipulation
- **SciPy** — Cosine similarity
