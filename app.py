import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Literal, Optional

MODEL_PATH = "models/model.pkl"

# Features used in training (must match your notebook)
CAT_FEATURES = ["neighbourhood_group", "room_type"]
NUM_FEATURES = [
    "latitude", "longitude",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "availability_365",
    "calculated_host_listings_count",
]
FEATURES = CAT_FEATURES + NUM_FEATURES

app = FastAPI(title="NYC Airbnb Price Predictor", version="1.0")

model = joblib.load(MODEL_PATH)


class AirbnbRequest(BaseModel):
    neighbourhood_group: Literal["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
    room_type: str

    latitude: float
    longitude: float

    minimum_nights: int = Field(ge=0)
    number_of_reviews: int = Field(ge=0)
    reviews_per_month: float = Field(ge=0)
    availability_365: int = Field(ge=0, le=365)
    calculated_host_listings_count: int = Field(ge=0)


class AirbnbBatchRequest(BaseModel):
    rows: List[AirbnbRequest]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(req: AirbnbRequest):
    row = req.model_dump()
    X = pd.DataFrame([row], columns=FEATURES)

    y_pred = float(model.predict(X)[0])  # already normal price units
    return {"predicted_price": y_pred}


@app.post("/predict_batch")
def predict_batch(req: AirbnbBatchRequest):
    rows = [r.model_dump() for r in req.rows]
    X = pd.DataFrame(rows, columns=FEATURES)

    preds = model.predict(X)
    return {"predicted_prices": [float(p) for p in preds]}
