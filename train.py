import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# -------------------------
# Paths
# -------------------------
DATA_PATH = "data/winequality-red.csv"
MODEL_PATH = "outputs/model/model.pkl"
METRICS_PATH = "outputs/metrics/results.json"

os.makedirs("outputs/model", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)

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
# Pipeline (Preprocessing + Model)
# -------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

pipeline.fit(X_train, y_train)

# -------------------------
# Evaluation
# -------------------------
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# -------------------------
# Save model
# -------------------------
joblib.dump(pipeline, MODEL_PATH)

# -------------------------
# Save metrics
# -------------------------
metrics = {
    "MSE": mse,
    "R2": r2
}

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

# -------------------------
# Print metrics (for logs)
# -------------------------
print("Training completed")
print("MSE:", mse)
print("R2:", r2)