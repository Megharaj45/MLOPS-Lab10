import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib
import json
import os

print("🚀 Starting retraining...")

# Load datasets
train = pd.read_csv("data/training_data.csv")
new = pd.read_csv("data/new_data.csv")

print("✅ Data loaded")

# Combine
combined = pd.concat([train, new])

# Prepare
X = combined.drop("wait_time_min", axis=1)
y = combined["wait_time_min"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train new model
model = RandomForestRegressor()
model.fit(X_train, y_train)

print("✅ Model trained")

# Evaluate new model
preds = model.predict(X_test)
retrained_mae = mean_absolute_error(y_test, preds)

# Load old model
old_model = joblib.load("models/best_model.pkl")
old_preds = old_model.predict(X_test)
champion_mae = mean_absolute_error(y_test, old_preds)

# Compare
improvement = champion_mae - retrained_mae
action = "promoted" if improvement > 0 else "kept_champion"

# Save model if better
if action == "promoted":
    joblib.dump(model, "models/best_model.pkl")

# Prepare output
output = {
    "original_data_rows": len(train),
    "new_data_rows": len(new),
    "combined_data_rows": len(combined),
    "champion_mae": float(champion_mae),
    "retrained_mae": float(retrained_mae),
    "improvement": float(improvement),
    "min_improvement_threshold": 0,
    "action": action,
    "comparison_metric": "mae"
}

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# Save JSON
with open("results/step4_s8.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Task 4 Completed")
print("OUTPUT:", output)