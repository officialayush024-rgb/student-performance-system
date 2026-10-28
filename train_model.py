
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Generate synthetic example data
rng = np.random.default_rng(42)
n = 800

df = pd.DataFrame({
    "attendance_pct": rng.uniform(35, 100, n),
    "study_hours_per_day": rng.uniform(0, 9, n),
    "previous_score": rng.uniform(20, 100, n),
    "assignment_score": rng.uniform(15, 100, n),
    "sleep_hours": rng.uniform(4, 10, n),
    "participation_level": rng.integers(1, 6, n)
})

# Generate demonstration target values
noise = rng.normal(0, 6, n)

df["final_score"] = np.clip(
    0.20 * df["attendance_pct"]
    + 2.5 * df["study_hours_per_day"]
    + 0.30 * df["previous_score"]
    + 0.22 * df["assignment_score"]
    + 1.1 * df["sleep_hours"]
    + 1.2 * df["participation_level"]
    - 8
    + noise,
    0,
    100
)

df["final_score"] = df["final_score"].round(1)

# Save the synthetic dataset
df.to_csv("data/sample_student_data.csv", index=False)

features = [
    "attendance_pct",
    "study_hours_per_day",
    "previous_score",
    "assignment_score",
    "sleep_hours",
    "participation_level"
]

X = df[features]
y = df["final_score"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Train the regression model
model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate the model on held-out data
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.3f}")

# Save model and feature order
joblib.dump(
    {"model": model, "features": features},
    "model/student_model.pkl"
)

print("Model saved successfully!")
