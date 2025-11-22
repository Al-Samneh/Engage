"""
Utility helpers for reading raw CSV/JSON data plus a few shared feature transforms.
Modified for SageMaker Inference (removed config dependency).
"""
import json
import os
import re
from typing import List, Tuple
import pandas as pd

# DATA_DIR default removed to avoid config import error
DATA_DIR = "./data"

# Quick lookup table for user preference alignment, values used in user_data csv file
PRICE_BUCKET_MAP = {
    "AED 50 - 100": "Low",
    "AED 100 - 150": "Medium",
    "AED 150 - 200": "High",
    "AED 200 - 300+": "Luxury",
}

def load_data(data_dir: str = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read every source file needed for Task 2 into pandas DataFrames.
    Returns: restaurants, reviews, users, and trends in that order.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found")

    with open(os.path.join(data_dir, "restaurant.json"), "r") as f:
        restaurants = json.load(f)

    df_restaurants = pd.DataFrame(restaurants)
    df_reviews = pd.read_csv(os.path.join(data_dir, "reviews.csv"))
    df_users = pd.read_csv(os.path.join(data_dir, "user_data.csv"))
    df_trends = pd.read_csv(os.path.join(data_dir, "dining_trends.csv"))

    return df_restaurants, df_reviews, df_users, df_trends


def parse_price_range_to_avg(price_str: str) -> float:
    """
    Convert free-form ranges like 'AED 120 - 200' into a single numeric midpoint.
    This allows tree-based models to compare restaurants by effective spend.
    """
    if pd.isna(price_str):
        return float("nan")
    cleaned = re.sub(r"[^\d\-]", " ", str(price_str))
    match = re.search(r"(\d+)\s*-\s*(\d+)", cleaned)
    if match:
        low, high = float(match.group(1)), float(match.group(2))
        return (low + high) / 2.0
    return float("nan")


def map_restaurant_price_bucket(price_str: str) -> str:
    """
    Map the textual price bucket to a smaller vocabulary the user-preference logic can work with.
    """
    if pd.isna(price_str):
        return "Unknown"
    return PRICE_BUCKET_MAP.get(str(price_str), "Unknown")


def tokenizer_splitter(text: str) -> List[str]:
    """
    Helper for scikit-learn's CountVectorizer. Splits comma-separated amenities/attributes
    into clean tokens without relying on regex token_pattern.
    """
    if pd.isna(text):
        return []
    return [token.strip() for token in str(text).split(",") if token.strip()]

