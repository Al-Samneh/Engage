# Model Performance Experiment Log

This document tracks the performance of different models, embedding strategies, and training objectives used during the development of the Restaurant Rating Predictor.

## Final Deployed Model (Regression with Rounding)
**Configuration:**
- **Model:** XGBoost Regressor (Objective: `reg:squarederror`)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
- **Dimensions:** 384
- **Strategy:** Regression output is clamped to [1, 5] and rounded to nearest integer for star rating.

**Performance:**
- **Accuracy (Exact Star):** ~66% (0.658)
- **Accuracy (±1 Star):** ~99.5%
- **RMSE:** ~0.50
- **R²:** ~0.87

---

## Experiment History

### 1. Baseline: Qwen Embeddings (Regression)
- **Model:** XGBoost Regressor
- **Embeddings:** `Qwen/Qwen3-Embedding-0.6B` (1024 dims)
- **Performance:**
  - Accuracy (Exact): ~60% (0.605)
  - Accuracy (±1 Star): ~99.5%
  - RMSE: ~0.54
  - R²: ~0.85
- **Notes:** High dimensionality (1024) led to slight overfitting/noise with the small dataset (1000 rows).

### 2. High-Performance: MPNet Embeddings (Regression)
- **Model:** XGBoost Regressor
- **Embeddings:** `sentence-transformers/all-mpnet-base-v2` (768 dims)
- **Performance:**
  - Accuracy (Exact): ~60% (0.60)
  - Accuracy (±1 Star): ~99.5%
  - RMSE: ~0.49 (Best raw regression error)
  - R²: ~0.88
- **Notes:** Achieved the best "raw" mathematical error (RMSE), but the continuous predictions often landed just across a rounding boundary (e.g., 3.49 vs 3.51), causing lower "exact star" accuracy.

### 3. Compact: MiniLM Embeddings (Regression) - **WINNER**
- **Model:** XGBoost Regressor
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
- **Performance:**
  - Accuracy (Exact): ~66% (0.658)
  - Accuracy (±1 Star): ~99.5%
- **Notes:** Lower dimensionality (384) acted as better regularization for this small dataset. Best practical "Star Rating" accuracy.

### 4. Classification Approach (XGBClassifier)
- **Model:** XGBoost Classifier (Objective: `multi:softmax`)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Performance:**
  - Accuracy (Exact): ~65% (0.65)
  - Accuracy (±1 Star): ~99.5%
- **Notes:** Performed very similarly to the regression approach (~65-66%). We chose **Regression** for the final deployment because it preserves the "magnitude" of errors better (predicting 4 when truth is 5 is better than predicting 1 when truth is 5; classification treats all errors equally by default).

---

## Deployment Strategy
We deploy the **Regression** model because:
1. **Robustness:** It understands that 4 stars is "close" to 5 stars.
2. **Simplicity:** It works natively with the continuous nature of many input features.
3. **Performance:** It achieved the highest exact match accuracy (66%) in our experiments.

