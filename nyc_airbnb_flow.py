import os
import json
import datetime
import numpy as np
import pandas as pd
import joblib

from metaflow import FlowSpec, step, Parameter

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class NYCAirbnbPriceFlow(FlowSpec):
    data_path = Parameter(
        "data_path",
        help="Path to AB_NYC_2019.csv",
        default="data/AB_NYC_2019.csv",
    )

    n_iter = Parameter(
        "n_iter",
        help="RandomizedSearchCV n_iter",
        default=15,
    )

    cv = Parameter(
        "cv",
        help="RandomizedSearchCV folds",
        default=3,
    )

    random_state = Parameter(
        "random_state",
        help="Random seed",
        default=42,
    )

    @step
    def start(self):
        self.TARGET = "price"
        self.CAT_FEATURES = ["neighbourhood_group", "room_type"]
        self.NUM_FEATURES = [
            "latitude", "longitude",
            "minimum_nights",
            "number_of_reviews",
            "reviews_per_month",
            "availability_365",
            "calculated_host_listings_count",
        ]
        self.FEATURES = self.CAT_FEATURES + self.NUM_FEATURES
        self.next(self.load_data)

    @step
    def load_data(self):
        assert os.path.exists(self.data_path), f"Dataset not found at: {self.data_path}"
        self.df_raw = pd.read_csv(self.data_path)
        self.next(self.clean_data)

    @step
    def clean_data(self):
        X = self.df_raw[self.FEATURES].copy()
        y = pd.to_numeric(self.df_raw[self.TARGET], errors="coerce").values

        mask = np.isfinite(y) & (y > 0)
        X = X.loc[mask].copy()
        y = y[mask]

        X["reviews_per_month"] = X["reviews_per_month"].fillna(0)

        for c in self.NUM_FEATURES:
            X[c] = pd.to_numeric(X[c], errors="coerce")

        valid = np.isfinite(X[self.NUM_FEATURES].values).all(axis=1)
        X = X.loc[valid].copy()
        y = y[valid]

        # Clip outliers to stabilize RMSE (same as notebook)
        cap = np.quantile(y, 0.995)
        y = np.clip(y, 0, cap)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        self.next(self.tune_and_train)

    @step
    def tune_and_train(self):
        preprocess = ColumnTransformer(
            transformers=[
                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), self.CAT_FEATURES),
                ("num", "passthrough", self.NUM_FEATURES),
            ],
            remainder="drop",
        )

        base_gbr = HistGradientBoostingRegressor(
            early_stopping=True,
            random_state=self.random_state,
        )

        pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", TransformedTargetRegressor(
                regressor=base_gbr,
                func=np.log1p,
                inverse_func=np.expm1
            ))
        ])

        param_dist = {
            "model__regressor__max_depth": [3, 4, 5, 6, 8],
            "model__regressor__learning_rate": [0.03, 0.05, 0.06, 0.08, 0.1],
            "model__regressor__max_iter": [150, 200, 250, 300, 400],
            "model__regressor__min_samples_leaf": [10, 20, 30, 50, 80],
            "model__regressor__l2_regularization": [0.0, 0.2, 0.5, 1.0, 2.0],
        }

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_dist,
            n_iter=int(self.n_iter),
            scoring="r2",
            cv=int(self.cv),
            random_state=self.random_state,
            n_jobs=-1,
            verbose=1,
        )

        search.fit(self.X_train, self.y_train)

        self.best_params = search.best_params_
        self.best_cv_r2 = float(search.best_score_)
        self.model = search.best_estimator_  # already fitted

        self.next(self.evaluate)

    @step
    def evaluate(self):
        preds = self.model.predict(self.X_test)

        rmse = float(np.sqrt(mean_squared_error(self.y_test, preds)))
        mae = float(mean_absolute_error(self.y_test, preds))
        r2 = float(r2_score(self.y_test, preds))

        self.metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "best_cv_r2": self.best_cv_r2,
            "best_params": self.best_params,
        }

        # sanity checks
        assert np.isfinite(rmse) and np.isfinite(mae) and np.isfinite(r2)
        assert mae > 0
        assert np.std(preds) > 1e-6

        self.next(self.package)

    @step
    def package(self):
        os.makedirs("models", exist_ok=True)

        joblib.dump(self.model, "models/model.pkl")

        meta = {
            "created_at": datetime.datetime.utcnow().isoformat(),
            "target": self.TARGET,
            "cat_features": self.CAT_FEATURES,
            "num_features": self.NUM_FEATURES,
            "features": self.FEATURES,
            "metrics": self.metrics,
        }

        with open("models/model_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        self.next(self.end)

    @step
    def end(self):
        print("Training complete.")
        print("Metrics:", self.metrics)


if __name__ == "__main__":
    NYCAirbnbPriceFlow()
