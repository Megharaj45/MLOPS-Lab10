import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import os
import joblib

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("wait_time_min", axis=1)
y = df["wait_time_min"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MLflow setup
mlflow.set_experiment("mediqueue-wait-time-min")

results = []

models = {
    "RandomForest": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor()
}

best_rmse = float("inf")
best_model_name = None
best_model = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Log to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("project_phase", "model_selection")

        results.append({
            "name": name,
            "mae": float(mae),
            "rmse": float(rmse)
        })

        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = model

# Save best model
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")

# Save JSON output
output = {
    "experiment_name": "mediqueue-wait-time-min",
    "models": results,
    "best_model": best_model_name,
    "best_metric_name": "rmse",
    "best_metric_value": float(best_rmse)
}

os.makedirs("results", exist_ok=True)
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Task 1 Completed")