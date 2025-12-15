import joblib
import pandas as pd
import bentoml
from bentoml.io import JSON

MODEL_PATH = "models/model.pkl"

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

model = joblib.load(MODEL_PATH)

svc = bentoml.Service("nyc_airbnb_price_service")


@svc.api(input=JSON(), output=JSON())
def predict(input_json):
    """
    Accepts either:
      1) a single dict with feature keys
      2) a dict {"rows": [ {..}, {..} ]}
    Returns predicted price(s) in original units.
    """
    if isinstance(input_json, dict) and "rows" in input_json:
        rows = input_json["rows"]
        X = pd.DataFrame(rows, columns=FEATURES)
        preds = model.predict(X)
        return {"predicted_prices": [float(p) for p in preds]}

    # single row
    X = pd.DataFrame([input_json], columns=FEATURES)
    pred = float(model.predict(X)[0])
    return {"predicted_price": pred}
