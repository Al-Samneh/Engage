import os

# Explicitly set the environment variable for SageMaker
os.environ['SAGEMAKER_PROGRAM'] = 'inference.py'
os.environ['SAGEMAKER_SUBMIT_DIRECTORY'] = '/opt/ml/model/code'

import json
import joblib
import pandas as pd
import numpy as np
import sys

# Ensure local imports work
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.dirname(__file__))

# We import this to ensure the unpickler can find the function 'tokenizer_splitter'
# if it was pickled as a reference to 'src.data_utils.tokenizer_splitter'
try:
    import src.data_utils
except ImportError:
    try:
        import data_utils
        sys.modules['src.data_utils'] = data_utils
        sys.modules['src'] = type('src', (), {'data_utils': data_utils})
    except ImportError:
        pass

def model_fn(model_dir):
    """
    Load the model from the model directory.
    """
    print("Loading model from", model_dir)
    model_path = os.path.join(model_dir, "best_restaurant_rating_model_xgboost.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading joblib model: {e}")
        raise
        
    print("Model loaded successfully")
    
    # Retrieve embedding columns if possible
    try:
        preprocessor = model.named_steps["preprocessor"]
        embed_entry = next(
            (entry for entry in preprocessor.transformers if entry[0] == "embed"), None
        )
        embedding_cols = embed_entry[2] if embed_entry else []
    except Exception as e:
        print(f"Warning: could not extract embedding cols from pipeline: {e}")
        embedding_cols = []
    
    # Attach metadata to model object so we can use it in predict_fn
    model.embedding_cols = embedding_cols
    return model

def input_fn(request_body, request_content_type):
    """
    Parse the incoming request body.
    """
    if request_content_type == "application/json":
        return json.loads(request_body)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Transform input data to DataFrame and generate prediction.
    """
    # Expect input_data to be the JSON structure from RatingPredictionRequest
    
    rest = input_data.get("restaurant", {})
    user = input_data.get("user", {})
    ctx = input_data.get("review_context", {})
    embeddings = input_data.get("embeddings", {})
    
    trend_features = rest.get("trend_features", {})
    
    row = {
        "helpful_count": ctx.get("helpful_count", 0),
        "age": user.get("age", 30),
        "avg_rating_given": user.get("avg_rating_given", 0),
        "total_reviews_written": user.get("total_reviews_written", 0),
        "popularity_score": rest.get("popularity_score", 0),
        "avg_price": rest.get("avg_price", 0),
        "booking_lead_time_days": ctx.get("booking_lead_time_days", 0),
        "popularity_7_day_avg": trend_features.get("popularity_7_day_avg", 0),
        "popularity_30_day_avg": trend_features.get("popularity_30_day_avg", 0),
        "popularity_lag_1": trend_features.get("popularity_lag_1", 0),
        "avg_price_7_day_avg": trend_features.get("avg_price_7_day_avg", 0),
        "popularity_7_day_growth": trend_features.get("popularity_7_day_growth", 0),
        "price_avg": rest.get("avg_price", 0),
        "price_alignment_score": user.get("user_cuisine_match", 0),
        "user_cuisine_match": user.get("user_cuisine_match", 0),
        "dietary_conflict": user.get("dietary_conflict", 0),
        "is_local_resident": 1 if user.get("is_local_resident", False) else 0,
        "resto_location": rest.get("location", ""),
        "resto_cuisine": rest.get("cuisine", ""),
        "resto_price_bucket": rest.get("price_bucket", ""),
        "home_location": user.get("home_location", ""),
        "preferred_price_range": user.get("preferred_price_range", ""),
        "dietary_restrictions": user.get("dietary_restrictions", "none"),
        "dining_frequency": user.get("dining_frequency", ""),
        "season": ctx.get("season", ""),
        "day_type": ctx.get("day_type", ""),
        "weather_impact_category": ctx.get("weather_impact_category", ""),
        "review_month": ctx.get("review_month", 1),
        "review_day_of_week": ctx.get("review_day_of_week", 0),
        "is_holiday": int(ctx.get("is_holiday", False)),
        "resto_description": rest.get("description", ""),
        "resto_amenities": ", ".join(rest.get("amenities", [])),
        "resto_attributes": ", ".join(rest.get("attributes", [])),
    }
    
    embedding_vector = embeddings.get("review_text_embedding", [])
    if hasattr(model, "embedding_cols") and model.embedding_cols:
        if len(embedding_vector) != len(model.embedding_cols):
             if not embedding_vector:
                 embedding_vector = [0.0] * len(model.embedding_cols)
        
        for idx, col in enumerate(model.embedding_cols):
            if idx < len(embedding_vector):
                row[col] = embedding_vector[idx]
            else:
                row[col] = 0.0

    df = pd.DataFrame([row])
    
    try:
        pred_raw = float(model.predict(df)[0])
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")
        
    return pred_raw

def output_fn(prediction, accept):
    if accept == "application/json":
        response = {"rating_prediction": prediction}
        return json.dumps(response), "application/json"
    raise ValueError(f"Unsupported accept type: {accept}")
