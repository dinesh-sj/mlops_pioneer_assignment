# End-to-End MLOps Demo — NYC Airbnb Price Prediction

This project demonstrates a **production-oriented, end-to-end MLOps pipeline** using the **NYC Airbnb 2019 dataset**.  
It covers the full ML lifecycle: **data ingestion → training → tuning → deployment → monitoring → retraining trigger**.

The implementation prioritizes:
- Serving-safe features
- Reproducibility
- Reasonable training time
- Deployment-ready models

---


## 📌 Project Overview

**Business Problem**
Predict the nightly **price** of an Airbnb listing in NYC (regression).

**Primary Metrics**

* RMSE
* MAE
* R²

**Key Design Principles**

* Avoid unstable IDs and free-text
* Favor robust, numeric, production-safe features
* Explicit tradeoff between accuracy and deployability

---

## 🧱 Project Structure

```
.
├── data/
│   └── AB_NYC_2019.csv
├── models/
│   ├── model.pkl
│   └── model_meta.json
├── app.py                  # FastAPI inference service
├── Dockerfile              # Containerized FastAPI deployment
├── bentoml_service.py      # Optional BentoML service
├── nyc_airbnb_flow.py      # Metaflow pipeline
├── requirements.txt
├── README.md
└── notebook.ipynb          # End-to-end MLOps walkthrough
```

---

## ⚙️ Setup

### 1️⃣ Install Dependencies (Local)

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Dataset

Download the **NYC Airbnb 2019 dataset** and place it at:

```
data/AB_NYC_2019.csv
```

---

## 🧪 MLOps Pipeline (9 Stages)

### **Stage 1 — Business Understanding**

* Target: `price`
* Problem type: Regression
* Metrics: RMSE, MAE, R²

---

### **Stage 2 — Data Engineering**

* Drop invalid prices (`price <= 0`)
* Fill missing `reviews_per_month` with `0`
* Enforce numeric schema
* Optional dataset versioning with **DVC**

---

### **Stage 3 — EDA**

* Price distribution
* Mean price by:

  * `neighbourhood_group`
  * `room_type`

---

### **Stage 4 — Model Training + Hyperparameter Tuning**

**Pipeline Components**

* `OrdinalEncoder` for categorical features
* Numeric passthrough
* `HistGradientBoostingRegressor`
* `TransformedTargetRegressor` (log-target)

**Why this setup**

* Dense numeric inputs → fast & stable
* Log-target reduces skew
* Fully deployment-safe preprocessing

**Tuning**

* `RandomizedSearchCV`
* 3-fold CV
* Optimizes **R²**

Typical performance:

```
RMSE ≈ 105–115
MAE  ≈ 45–55
R²   ≈ 0.30–0.35
```

---

### **Stage 5 — Evaluation**

* Final test-set evaluation
* Sanity checks for non-degenerate predictions

---

### **Stage 6 — Model Packaging**

Saved artifacts:

* `models/model.pkl` — full preprocessing + model pipeline
* `models/model_meta.json` — metrics & metadata

The model returns **price in original units**.

---

## 🚀 Stage 7 — Deployment

### 🔹 FastAPI (Local)

```bash
uvicorn app:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

---

### 🔹 FastAPI with Docker (Production-Ready)

#### Build image

```bash
docker build -t nyc-airbnb-price-api .
```

#### Run container

```bash
docker run -p 8000:8000 nyc-airbnb-price-api
```

The API will be available at:

```
http://localhost:8000
```

---

### 🔹 BentoML (Optional)

```bash
bentoml serve bentoml_service:svc --reload
```

---

## 📈 Stage 8 — Monitoring (Evidently)

* Data drift detection
* Reference: training data
* Current: simulated production data
* Output artifact:

```
drift_report.html
```

---

## 🔁 Stage 9 — Continuous Retraining Trigger

Rule-based retraining trigger:

```
If share_drifted_features > 0.30 → retrain
```

Can be automated via:

* Cron
* Airflow
* CI/CD pipelines

---

## 🔁 Metaflow Integration

Run the full pipeline:

```bash
python nyc_airbnb_flow.py run --data_path data/AB_NYC_2019.csv
```

Inspect metrics:

```python
from metaflow import Flow
run = list(Flow("NYCAirbnbPriceFlow").runs())[0]
run.data.metrics
```

---

## 🧠 Key Takeaways

* Accuracy balanced with **production robustness**
* Hyperparameter tuning validates design choices
* Log-target handling fully encapsulated
* Model is:

  * Fast to train
  * Stable to deploy
  * Easy to monitor and retrain

---
