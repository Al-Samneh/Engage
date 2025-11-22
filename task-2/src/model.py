"""
Training utilities for the rating regression model (XGBoost wrapped in a scikit pipeline).
"""
import os
import time
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from config import SEED, TASK_DIR
from .data_utils import tokenizer_splitter

sns.set_style("whitegrid")


def train_xgboost(
    df_final,
    embedding_cols: List[str],
    output_dir: str = TASK_DIR,
) -> Tuple[Pipeline, float, float, float]:
    """
    End-to-end training loop:
    1) select feature subsets
    2) build preprocessing / model pipeline
    3) hyperparameter search
    4) evaluation + artifact saving
    """
    target_col = "rating"
    y = df_final[target_col]

    numerical_features = [
        "helpful_count",
        "age", "avg_rating_given", "total_reviews_written",
        "popularity_score", "avg_price", "booking_lead_time_days",
        "popularity_7_day_avg", "popularity_30_day_avg", "popularity_lag_1",
        "avg_price_7_day_avg", "popularity_7_day_growth",
        "price_avg", "price_alignment_score",
        "user_cuisine_match", "is_local_resident",
    ]

    categorical_features = [
        "resto_location", "resto_cuisine", "resto_price_bucket",
        "home_location", "preferred_price_range", "dietary_restrictions", "dining_frequency",
        "season", "day_type", "weather_impact_category",
        "review_month", "review_day_of_week", "is_holiday",
    ]

    desc_feature = "resto_description"
    amenities_feature = "resto_amenities"
    attributes_feature = "resto_attributes"

    valid_mask = y.notna()
    cols_to_keep = (
        numerical_features
        + categorical_features
        + [desc_feature, amenities_feature, attributes_feature]
        + embedding_cols
    )

    X = df_final.loc[valid_mask, cols_to_keep]
    y = y[valid_mask]
    # Weight more influential (frequently marked helpful) reviews slightly higher.
    sample_weight = 1.0 + np.log1p(df_final.loc[valid_mask, "helpful_count"])

    print("[DATA] Training matrix shape:", X.shape)

    X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
        X, y, sample_weight, test_size=0.2, random_state=SEED
    )

    # Numeric preprocessing: fill gaps, then scale to zero mean/unit variance.
    num_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical preprocessing: fill gaps, one-hot encode.
    cat_tf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    desc_tf = TfidfVectorizer(max_features=100, stop_words="english", ngram_range=(1, 2))
    amen_tf = CountVectorizer(tokenizer=tokenizer_splitter, token_pattern=None, max_features=50)
    attr_tf = CountVectorizer(tokenizer=tokenizer_splitter, token_pattern=None, max_features=50)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_tf, numerical_features),
            ("cat", cat_tf, categorical_features),
            ("embed", "passthrough", embedding_cols),
            ("desc", desc_tf, desc_feature),
            ("amen", amen_tf, amenities_feature),
            ("attr", attr_tf, attributes_feature),
        ],
        remainder="drop",
    )

    xgb = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )

    param_distributions = {
        "regressor__n_estimators": [400, 600, 800],
        "regressor__max_depth": [4, 6, 8],
        "regressor__learning_rate": [0.01, 0.03, 0.05],
        "regressor__subsample": [0.8, 0.9, 1.0],
        "regressor__colsample_bytree": [0.7, 0.8, 1.0],
        "regressor__min_child_weight": [1, 3, 5],
        "regressor__gamma": [0, 0.1, 0.3],
    }

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", xgb),
        ]
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=15,
        cv=3,
        scoring="neg_mean_squared_error",
        verbose=1,
        random_state=SEED,
        n_jobs=-1,
    )

    print("[TRAIN] Starting XGBoost hyperparameter search...")
    start = time.time()
    search.fit(X_train, y_train, regressor__sample_weight=sw_train)
    duration = time.time() - start
    best_model = search.best_estimator_
    print(f"[TRAIN] Completed in {duration:.1f}s")
    print("[TRAIN] Best params:", search.best_params_)

    y_pred = best_model.predict(X_test)
    # Clamp to valid rating range so downstream rounding makes sense.
    y_pred = np.clip(y_pred, 1.0, 5.0)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Classification-style view: treat ratings as discrete stars.
    y_true_array = y_test.to_numpy()
    pred_rounded = np.rint(y_pred)
    exact_match = np.mean(pred_rounded == np.rint(y_true_array))
    within_one = np.mean(np.abs(pred_rounded - y_true_array) <= 1.0)

    print("\n[TEST] Performance")
    print("  R2   :", round(r2, 4))
    print("  RMSE :", round(rmse, 4))
    print("  MAE  :", round(mae, 4))
    print("  Accuracy (rounded stars)        :", round(float(exact_match), 4))
    print("  Accuracy (within ±1 star)       :", round(float(within_one), 4))

    _plot_actual_vs_predicted(y_test, y_pred, output_dir)
    joblib.dump(best_model, os.path.join(output_dir, "best_restaurant_rating_model_xgboost.pkl"))
    print("[SAVE] best_restaurant_rating_model_xgboost.pkl")

    return best_model, rmse, mae, r2


def _plot_actual_vs_predicted(y_true, y_pred, output_dir):
    """
    Quick diagnostic plot to visualize calibration. Saved alongside the trained model.
    """
    plt.figure(figsize=(7, 7))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.4)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "r--", label="Perfect Prediction")
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title("XGBoost – Actual vs Predicted User Ratings")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "xgboost_actual_vs_predicted.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[PLOT] Saved {plot_path}")

