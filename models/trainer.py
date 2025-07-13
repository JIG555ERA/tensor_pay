import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import os

# ====== Suppress Warnings ======
warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")

# ====== Load Dataset ======
df = pd.read_csv("datasets/income_census_data_with_salary.csv")

# ====== Encode Categorical Columns ======
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ====== Feature-Target Split ======
X = df.drop("salary", axis=1)
y = df["salary"]

# ====== Scale Numeric Features ======
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ====== Train-Test Split ======
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ====== Build Model ======
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Regression output
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])

# ====== Train Model ======
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1, verbose=1)

# ====== Create Model Directory ======
os.makedirs("models", exist_ok=True)

# ====== Save Model and Preprocessing ======
model.save("models/salary_predictor.h5")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("✅ Training complete. Model and preprocessing saved to 'models/' directory.")
