"""
Feature engineering helpers for Task 2.
"""
import pandas as pd

from .data_utils import (
    map_restaurant_price_bucket,
    parse_price_range_to_avg,
)


def engineer_trend_features(df_trends: pd.DataFrame) -> pd.DataFrame:
    """
    Produce rolling averages and lagged popularity signals for each cuisine.
    These capture momentum (7-day) and seasonality (30-day) that impact ratings.
    """
    df_trends = df_trends.copy()
    df_trends["date"] = pd.to_datetime(df_trends["date"])
    df_trends = df_trends.sort_values(["cuisine_type", "date"]).reset_index(drop=True)

    grouped_pop = df_trends.groupby("cuisine_type")["popularity_score"]
    grouped_price = df_trends.groupby("cuisine_type")["avg_price"]

    df_trends["popularity_lag_1"] = grouped_pop.shift(1)
    df_trends["popularity_7_day_avg"] = grouped_pop.shift(1).rolling(7, min_periods=1).mean()
    df_trends["popularity_30_day_avg"] = grouped_pop.shift(1).rolling(30, min_periods=1).mean()
    df_trends["avg_price_7_day_avg"] = grouped_price.shift(1).rolling(7, min_periods=1).mean()

    prev_7 = grouped_pop.shift(7)
    growth = (df_trends["popularity_lag_1"] - prev_7) / prev_7.replace(0, pd.NA)
    growth = growth.replace([pd.NA, pd.NaT, float("inf"), float("-inf")], 0).fillna(0)
    df_trends["popularity_7_day_growth"] = growth

    return df_trends


def engineer_merged_features(
    df_restaurants: pd.DataFrame,
    df_reviews: pd.DataFrame,
    df_users: pd.DataFrame,
    df_trends: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join all datasets and derive interaction features so the model sees both context and preferences.
    """
    df_trends = engineer_trend_features(df_trends)
    df_reviews = df_reviews.copy()
    df_reviews["date"] = pd.to_datetime(df_reviews["date"])
    df_trends["date"] = pd.to_datetime(df_trends["date"])

    # Prefix restaurant columns to avoid collisions after the merge.
    df_restaurants = df_restaurants.add_prefix("resto_")

    df = (
        df_reviews
        .merge(df_restaurants, left_on="restaurant_id", right_on="resto_id", how="left")
        .merge(df_users, on="user_id", how="left")
        .merge(df_trends, left_on=["date", "resto_cuisine"], right_on=["date", "cuisine_type"], how="left")
    )

    # Pricing features: both numeric midpoint and bucketed label.
    df["price_avg"] = df["resto_price_range"].apply(parse_price_range_to_avg)
    df["resto_price_bucket"] = df["resto_price_range"].apply(map_restaurant_price_bucket)

    # User preference matching.
    df["favorite_cuisines"] = df["favorite_cuisines"].fillna("").astype(str)
    df["favorite_cuisines_list"] = (
        df["favorite_cuisines"].str.lower()
        .apply(lambda s: [c.strip() for c in s.split(",") if c.strip()])
    )
    df["resto_cuisine_lower"] = df["resto_cuisine"].str.lower()
    df["user_cuisine_match"] = df.apply(
        lambda row: int(row["resto_cuisine_lower"] in row["favorite_cuisines_list"]),
        axis=1,
    )

    def price_alignment(row):
        user_pref = str(row["preferred_price_range"]).strip().lower()
        resto_bucket = str(row["resto_price_bucket"]).strip().lower()
        if user_pref in ["", "nan"] or resto_bucket == "unknown":
            return 0
        return int(user_pref == resto_bucket)

    df["price_alignment_score"] = df.apply(price_alignment, axis=1)

    df["is_local_resident"] = (df["home_location"] == df["resto_location"]).astype(int)

    # Calendar signals to capture seasonality and weekend effects.
    df["review_month"] = df["date"].dt.month.astype("int8")
    df["review_day_of_week"] = df["date"].dt.dayofweek.astype("int8")

    # Text fields feed downstream vectorizers; replace NaNs to avoid TypeErrors.
    for col in ["resto_amenities", "resto_attributes", "resto_description", "review_text"]:
        df[col] = df[col].fillna("")

    return df

