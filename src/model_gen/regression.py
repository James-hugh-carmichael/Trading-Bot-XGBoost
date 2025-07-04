import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

"""
XGBoost Regressor Training Script for Predicting Target Returns

This script trains and evaluates XGBoost regression models to predict the `target_return` — a continuous value representing the future return of an asset — using historical features engineered from market data.

Main Workflow:
--------------
1. Loads preprocessed data from a Parquet file (`training_data.parquet`).
2. Defines feature columns (excluding label columns, symbol, and future-close leakage).
3. Applies quantile filtering to exclude extreme outliers in the target variable.
4. Splits filtered data chronologically into training and testing sets.
5. Trains multiple XGBoost regression models with different learning rates.
6. Evaluates models using:
    - Mean Absolute Error (MAE)
    - R² Score
7. Saves all models and identifies the best one based on MAE.

Key Parameters:
---------------
- Quantile Cutoffs: Used to limit training data to a specific return range (e.g., 0.065th to 95th percentile)
- Model: `xgb.XGBRegressor` with 200 estimators, max depth 6, and 80% subsample
- Metrics: `mean_absolute_error`, `r2_score` from `sklearn`
- Output Directory: All models saved under `models/`

File I/O:
---------
- Input: `data/dataprocessed/training_data.parquet`
- Output:
    - All trained models: `models/xgb_regressor_lr_<lr>_q_<lower>_<upper>.pkl`
    - Best model: `models/best_xgb_regressor.pkl`

Dependencies:
-------------
- pandas
- xgboost
- scikit-learn
- joblib
- parquet support (pyarrow or fastparquet)
"""

# Load processed data
df = pd.read_parquet("data/dataprocessed/training_data.parquet")

# Define features (exclude target columns and future data)
features = [
    col for col in df.columns
    if col not in ['target', 'target_return', 'future_close', 'symbol', 'index']
]

X_full = df[features]
y_full = df['target_return']

os.makedirs("models", exist_ok=True)

# Define quantile cutoff pairs to try (lower, upper)
quantile_ranges = [
    (0.00065, 0.95),
   
]

learning_rates = [0.04]

best_overall_mae = float('inf')
best_overall_model = None
best_overall_lr = None
best_overall_quantiles = None

for lower_q, upper_q in quantile_ranges:
    print(f"\n=== Trying quantile cutoffs: lower={lower_q}, upper={upper_q} ===")

    lower = y_full.quantile(lower_q)
    upper = y_full.quantile(upper_q)
    mask = (y_full >= lower) & (y_full <= upper)

    X = X_full[mask]
    y = y_full[mask]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    best_model = None
    best_mae = float('inf')
    best_lr = None

    for lr in learning_rates:
        print(f"\nTraining regression model with learning_rate = {lr}")

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mae'
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"MAE: {mae:.6f} | R²: {r2:.4f}")

        joblib.dump(model, f"models/xgb_regressor_lr_{lr}_q_{int(lower_q*10000)}_{int(upper_q*10000)}.pkl")

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_lr = lr

    print(f"Best MAE for quantiles {lower_q}-{upper_q} is {best_mae:.6f} at lr={best_lr}")

    if best_mae < best_overall_mae:
        best_overall_mae = best_mae
        best_overall_model = best_model
        best_overall_lr = best_lr
        best_overall_quantiles = (lower_q, upper_q)

if best_overall_model:
    joblib.dump(best_overall_model, "models/best_xgb_regressor.pkl")
    print(f"\nOverall best model saved with learning_rate={best_overall_lr} and quantiles={best_overall_quantiles} (MAE: {best_overall_mae:.6f})")
