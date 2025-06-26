import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

"""
XGBoost Classifier Training Script for Financial Market Prediction

This script trains and evaluates multiple XGBoost classification models using labeled financial market data.
The goal is to predict the `target` variable — typically a directional signal (e.g., buy/sell/hold) — based on engineered features.

Main Workflow:
--------------
1. Loads preprocessed training data from a Parquet file (`training_data.parquet`).
2. Selects relevant feature columns (excludes metadata and target-related columns).
3. Splits data chronologically (no shuffle) into training and testing sets.
4. Trains multiple XGBoost models with varying learning rates.
5. Evaluates models using:
    - Accuracy on "confident" predictions (proba > 0.7 or < 0.3)
    - Full classification report on all test samples
6. Saves all models and identifies the best one based on confident prediction accuracy.

Key Components:
---------------
- Model: `xgb.XGBClassifier` (100 trees, depth 6, 80% subsample)
- Confidence filtering: Only evaluate predictions with high probability to reduce noise
- Metrics: `accuracy_score`, `classification_report` from `sklearn`
- Model persistence: Saved as `.pkl` files using `joblib` in the `models/` directory

Directory Requirements:
-----------------------
- Input: `data/dataprocessed/training_data.parquet`
- Output: `models/best_xgb_classifier.pkl` and other models for each learning rate

Dependencies:
-------------
- pandas
- xgboost
- scikit-learn
- joblib
- parquet file support (e.g., pyarrow or fastparquet)
"""
# Load processed data
df = pd.read_parquet("data/dataprocessed/training_data.parquet")

# Define feature columns (excluding labels and metadata)
features = [
    col for col in df.columns 
    if col not in ['target', 'target_return', 'future_close', 'symbol', 'index']
]

X = df[features]
y = df['target']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Try different learning rates
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]

best_model = None
best_acc = 0
best_lr = None

os.makedirs("models", exist_ok=True)

for lr in learning_rates:
    print(f"\nTraining model with learning_rate = {lr}")
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=lr,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    y_proba = model.predict_proba(X_test)[:, 1]
    confident_mask = (y_proba > 0.7) | (y_proba < 0.3)
    if confident_mask.sum() > 0:
        filtered_preds = (y_proba[confident_mask] > 0.5).astype(int)
        filtered_actual = y_test[confident_mask]
        acc = accuracy_score(filtered_actual, filtered_preds)
    else:
        acc = 0

    
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save all models if needed                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    joblib.dump(model, f"models/xgb_model_lr_{lr}.pkl")
    
    # Track best
    if acc > best_acc:
        best_model = model
        best_acc = acc
        best_lr = lr

# Save the best model
if best_model:
    joblib.dump(best_model, "models/best_xgb_classifier.pkl")
    print(f"\nBest model saved: learning_rate={best_lr} with Accuracy={best_acc:.4f}")
