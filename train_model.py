import os
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

DATA_PATH  = "data/raw/mumbai_train.csv"
MODEL_PATH = "models/lgbm_no2_model.pkl"

TARGET = "tropospheric_NO2_column_number_density"

FEATURES = [
    "elevation",
    "temperature_2m",
    "dewpoint_temperature_2m",
    "surface_pressure",
    "wind_speed",
    "wind_direction_sin",
    "wind_direction_cos",
    "day_of_year",
    "month",
    "cloud_fraction",
    "boundary_layer_height",
    "total_precipitation_7d_mm",
]

# ── 1. CREATE MODELS DIRECTORY ────────────────────────────────────────────────

os.makedirs("models", exist_ok=True)

# ── 2. LOAD DATA ──────────────────────────────────────────────────────────────

print("📂 Loading:", DATA_PATH)
df = pd.read_excel(DATA_PATH)
print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ── 3. DEFINE X AND y ────────────────────────────────────────────────────────

X = df[FEATURES].copy()
y = df[TARGET].copy()

# Drop rows with missing values in features or target
mask = X.notna().all(axis=1) & y.notna()
X, y = X[mask], y[mask]
print(f"   Clean samples: {len(X):,}")

# ── 4. 80 / 20 TRAIN-TEST SPLIT ──────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n📊 Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ── 5. TRAIN LIGHTGBM ────────────────────────────────────────────────────────

print("\n🚀 Training LightGBM Regressor ...")

model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)
print("   Training complete.")

# ── 6. EVALUATE ───────────────────────────────────────────────────────────────

y_pred = model.predict(X_test)
r2     = r2_score(y_test, y_pred)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n" + "=" * 40)
print("  📈  MODEL EVALUATION RESULTS")
print("=" * 40)
print(f"  R² Score  :  {r2:.6f}")
print(f"  RMSE      :  {rmse:.6e}")
print("=" * 40)

# ── 7. SAVE MODEL ─────────────────────────────────────────────────────────────

joblib.dump(model, MODEL_PATH)
print(f"\n💾 Model saved → {MODEL_PATH}")
print("✅ Done!")
