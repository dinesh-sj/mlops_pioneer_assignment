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
- RMSE
- MAE
- R²

**Key Design Principles**
- Avoid unstable IDs and free-text
- Favor robust, numeric, production-safe features
- Explicit tradeoff between accuracy and deployability

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
├── bentoml_service.py      # Optional BentoML service
├── nyc_airbnb_flow.py      # Metaflow pipeline
├── requirements.txt
├── README.md
└── notebook.ipynb          # End-to-end MLOps walkthrough

````

---

## ⚙️ Setup

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
````

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
* Fill missing `reviews_per_month` with 0
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
* `TransformedTargetRegressor` for log-target handling

**Why this setup**

* Dense numeric inputs → faster & better generalization
* Log-target stabilizes skewed price distribution
* Encoding and model choice are deployment-friendly

**Hyperparameter Tuning**

* `RandomizedSearchCV`
* 3-fold CV
* Optimizes **R²**
* Confirms chosen configuration is near-optimal

Typical tuned performance:

```
RMSE ≈ 105–115
MAE  ≈ 45–55
R²   ≈ 0.30–0.35
```

---

### **Stage 5 — Evaluation**

* Final evaluation on holdout test set
* Sanity checks ensure non-degenerate predictions

---

### **Stage 6 — Model Packaging**

Artifacts saved:

* `models/model.pkl` — full preprocessing + model pipeline
* `models/model_meta.json` — metadata & metrics

The saved model directly returns **price in original units**.

---

### **Stage 7 — Deployment**

#### FastAPI

```bash
uvicorn app:app --reload
```

#### BentoML (optional)

```bash
bentoml serve bentoml_service:svc --reload
```

---

### **Stage 8 — Monitoring (Evidently)**

* Data drift detection
* Reference: training data
* Current: simulated production data
* Output:

```
drift_report.html
```

---

### **Stage 9 — Continuous Retraining Trigger**

A simple rule-based trigger:

```text
If share_drifted_features > 0.30 → retrain
```

This logic can be automated via:

* Cron
* Airflow
* CI/CD pipeline

---

## 🔁 Metaflow Integration

Run the full pipeline:

```bash
python nyc_airbnb_flow.py run --data_path data/AB_NYC_2019.csv
```

Inspect results:

```python
from metaflow import Flow
run = list(Flow("NYCAirbnbPriceFlow").runs())[0]
run.data.metrics
```

---

## 🧠 Key Takeaways

* Accuracy was intentionally balanced with **production robustness**
* Hyperparameter tuning validates the final configuration
* Log-target handling is fully encapsulated inside the pipeline
* Model is:

  * Fast to train
  * Stable to deploy
  * Easy to monitor and retrain

---

## ✅ Status

✔ End-to-end
✔ Assignment-ready
✔ Interview-ready
✔ Production-aligned

---
