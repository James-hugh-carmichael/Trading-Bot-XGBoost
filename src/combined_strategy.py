import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_parquet("data/dataprocessed/training_data.parquet")
features = [
    col for col in df.columns
    if col not in ['target', 'target_return', 'future_close', 'symbol', 'index']
]

X = df[features]
true_returns = df['target_return']
true_labels = df['target']

# Load models
clf_model = joblib.load("models/best_xgb_classifier.pkl")  
reg_model = joblib.load("models/best_xgb_regressor.pkl")

# ----------- Feature Importance Analysis ------------
def plot_feature_importance(model, title, feature_names, top_n=20):
    booster = model.get_booster()
    importance_dict = booster.get_score(importance_type='gain')
    importance = pd.Series(importance_dict).reindex(feature_names).fillna(0)
    importance = importance.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    importance.plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Importance (gain)")
    plt.tight_layout()
    plt.show()

print("\nTop Feature Importances - Classifier")
plot_feature_importance(clf_model, "Top Classifier Feature Importances", features)

print("\nTop Feature Importances - Regressor")
plot_feature_importance(reg_model, "Top Regressor Feature Importances", features)


# ----------- Predictions & Strategy Evaluation ------------

# Classifier predictions
clf_probs = clf_model.predict_proba(X)[:, 1]  # Probability of "up"
clf_preds = clf_model.predict(X)

# Regression predictions
reg_preds = reg_model.predict(X)

# Thresholds
clf_threshold = 0.58
reg_threshold = 0.002

# Long trade conditions
long_clf_mask = clf_probs > clf_threshold
long_reg_mask = reg_preds > reg_threshold
long_mask = long_clf_mask & long_reg_mask

# Short trade conditions 
short_clf_mask = clf_probs < 0.23
short_reg_mask = reg_preds < -0.0065
short_mask = short_clf_mask & short_reg_mask

# Combine all trades 
final_mask = long_mask | short_mask

# Create direction array for selected trades
direction = np.full(len(df), 'none', dtype=object)
direction[long_mask] = 'long'
direction[short_mask] = 'short'

# Selected trades 
selected = df[final_mask].copy()
selected['predicted_return'] = reg_preds[final_mask]
selected['true_return'] = true_returns[final_mask]
selected['direction'] = direction[final_mask]

# Adjust returns for shorts
selected['adjusted_return'] = np.where(
    selected['direction'] == 'short',
    -selected['true_return'],
    selected['true_return']
)

selected['was_profitable'] = selected['adjusted_return'] > 0

# ----------- Reporting ------------

print(f"\nTotal samples: {len(df)}")
print(f"Long trades: {long_mask.sum()}")
print(f"Short trades: {short_mask.sum()}")
print(f"Final trades selected: {final_mask.sum()}")

print(f"\nAverage predicted return: {selected['predicted_return'].mean():.4f}")
print(f"Average true return (adjusted): {selected['adjusted_return'].mean():.4f}")
print(f"Win rate: {selected['was_profitable'].mean():.2%}")

# ----------- Plots ------------

selected['cum_return'] = (1 + selected['adjusted_return']).cumprod()
plt.figure(figsize=(10, 4))
plt.title("Cumulative Return (Long + Short Strategy)")
plt.plot(selected['cum_return'])
plt.xlabel("Trade Index")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.hist(clf_probs, bins=50, alpha=0.6, label='Classifier Probabilities')
plt.axvline(clf_threshold, color='red', linestyle='--', label='Long Threshold')
plt.axvline(0.23, color='purple', linestyle='--', label='Short Threshold')
plt.title("Classifier Probability Distribution")
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.hist(reg_preds, bins=50, alpha=0.6, label='Regression Predicted Returns')
plt.axvline(reg_threshold, color='green', linestyle='--', label='Long Reg Threshold')
plt.axvline(-0.0085, color='orange', linestyle='--', label='Short Reg Threshold')
plt.title("Regressor Predicted Return Distribution")
plt.legend()
plt.show()
