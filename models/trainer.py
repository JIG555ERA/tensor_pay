import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv("datasets/income_census_data_with_salary.csv")

# Encode categorical columns
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Split into features and target
X = df.drop("salary", axis=1)
y = df["salary"]

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Scikit-learn model (no TensorFlow)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model + preprocessing
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/salary_predictor.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("✅ Model and preprocessing files saved successfully.")
