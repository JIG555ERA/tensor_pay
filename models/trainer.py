# train_model.py
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data
df = pd.read_csv("datasets/income_census_data_with_salary.csv")

# Columns to encode
categorical_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex"]
encoders = {}

# Encode categorical features
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Save encoders
joblib.dump(encoders, "models/encoders.pkl")

# Split features and target
X = df.drop("salary", axis=1)
y = df["salary"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# Train model
model = RandomForestRegressor()
model.fit(X_scaled, y)

# Save model
joblib.dump(model, "models/sklearn_salary_model.pkl")

print("✅ Model, encoders, and scaler saved successfully!")
