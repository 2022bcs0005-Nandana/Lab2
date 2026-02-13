import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# -------------------------
# Paths (MATCH JENKINSFILE)
# -------------------------
DATA_PATH = "data/winequality-red.csv"
MODEL_PATH = "app/artifacts/model.pkl"
METRICS_PATH = "app/artifacts/metrics.json"

os.makedirs("app/artifacts", exist_ok=True)

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop("quality", axis=1)
y = df["quality"]

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Random Forest Model
# -------------------------
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------
# Evaluation
# -------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# -------------------------
# Save model and metrics (Jenkins compatible)
# -------------------------
joblib.dump(model, MODEL_PATH)

metrics = {
    "accuracy": float(r2),
    "mse": float(mse)
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print("Training completed - Random Forest (100 trees)")
print("MSE:", mse)
print("R2 (accuracy):", r2)
print("Metrics saved to app/artifacts/metrics.json")