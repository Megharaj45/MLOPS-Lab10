import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import json

# Load data
df = pd.read_csv("data/training_data.csv")

X = df.drop("wait_time_min", axis=1)
y = df["wait_time_min"]

# Parameter grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5],
    "max_features": ["auto", "sqrt"]
}

mlflow.set_experiment("mediqueue-wait-time-min")

with mlflow.start_run(run_name="tuning-mediqueue") as parent_run:

    grid = GridSearchCV(
        RandomForestRegressor(),
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error"
    )

    grid.fit(X, y)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    best_rmse = np.sqrt(-grid.best_score_)

    preds = best_model.predict(X)
    best_mae = mean_absolute_error(y, preds)

    output = {
        "search_type": "grid",
        "n_folds": 5,
        "total_trials": len(grid.cv_results_["params"]),
        "best_params": best_params,
        "best_mae": float(best_mae),
        "best_cv_mae": float(best_rmse),
        "parent_run_name": "tuning-mediqueue"
    }

    with open("results/step2_s2.json", "w") as f:
        json.dump(output, f, indent=4)

print("✅ Task 2 Completed")