# ============================================================
#   WALMART PRICE FORECASTING - MODEL TRAINING (ALL METRICS)
#   Python 3.13 + sklearn compatible
# ============================================================

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------
DATA_PATH = r"C:\Users\Vibhu\Desktop\Sales_Forecasting-Inventory_Management\data\Walmart_preprocessed_completed_PurchaseDate_2019_2024.csv"
df = pd.read_csv(DATA_PATH)

print("Dataset Loaded:", df.shape)

# ------------------------------------------------------------
# 2. HANDLE YES / NO / BOOLEAN VALUES
# ------------------------------------------------------------
bool_map = {
    "Yes": 1, "No": 0,
    "yes": 1, "no": 0,
    True: 1, False: 0
}

for col in df.columns:
    if df[col].dtype == object:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset(bool_map.keys()):
            df[col] = df[col].map(bool_map)

# ------------------------------------------------------------
# 3. DATE FEATURE ENGINEERING
# ------------------------------------------------------------
df['Purchase_Date'] = pd.to_datetime(df['Purchase_Date'], errors='coerce')

df['Year'] = df['Purchase_Date'].dt.year
df['Month'] = df['Purchase_Date'].dt.month
df['Week'] = df['Purchase_Date'].dt.isocalendar().week.astype(float)

# ------------------------------------------------------------
# 4. ENCODE CATEGORICAL FEATURES
# ------------------------------------------------------------
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col].astype(str))

# ------------------------------------------------------------
# 5. FEATURE SELECTION
# ------------------------------------------------------------
FEATURES = [
    'Year', 'Month', 'Week',
    'Purchase_Amount', 'Discount_Applied', 'Rating',
    'Age', 'Repeat_Customer',
    'Competitor_Price', 'Market_Share',
    'Competitor_Rating', 'Promotion_Competitor'
]

TARGET = 'Market_Price'

X = df[FEATURES]
y = df[TARGET]

# ------------------------------------------------------------
# 6. HANDLE MISSING VALUES
# ------------------------------------------------------------
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=FEATURES)

# ------------------------------------------------------------
# 7. TRAIN–TEST SPLIT (TIME AWARE)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# ------------------------------------------------------------
# 8. MODELS & HYPERPARAMETERS
# ------------------------------------------------------------
models = {
    "LinearRegression": (LinearRegression(), {}),

    "RandomForest": (
        RandomForestRegressor(random_state=42),
        {
            "n_estimators": [100, 200],
            "max_depth": [10, 20]
        }
    ),

    "GradientBoosting": (
        GradientBoostingRegressor(random_state=42),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1]
        }
    ),

    "XGBoost": (
        XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        ),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [4, 6]
        }
    )
}

# ------------------------------------------------------------
# 9. TRAINING + MULTI-METRIC EVALUATION
# ------------------------------------------------------------
best_model = None
best_rmse = float("inf")
results = []

for name, (model, params) in models.items():
    print(f"\nTraining {name}...")

    if params:
        grid = GridSearchCV(
            model,
            params,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        trained_model = grid.best_estimator_
    else:
        trained_model = model
        trained_model.fit(X_train, y_train)

    preds = trained_model.predict(X_test)

    # ---------------- METRICS ----------------
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)  # ✅ sklearn-safe RMSE
    mae = mean_absolute_error(y_test, preds)

    # Safe MAPE (avoids divide-by-zero)
    mape = np.mean(
        np.abs((y_test - preds) / np.clip(y_test, 1e-8, None))
    ) * 100

    r2 = r2_score(y_test, preds)

    print(
        f"{name} -> "
        f"RMSE: {rmse:.2f}, "
        f"MAE: {mae:.2f}, "
        f"MAPE: {mape:.2f}%, "
        f"R2: {r2:.3f}"
    )

    results.append([name, rmse, mae, mape, r2])

    # Best model selection by RMSE
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = trained_model

# ------------------------------------------------------------
# 10. SAVE BEST MODEL & METRICS
# ------------------------------------------------------------
joblib.dump(best_model, "best_price_forecast_model.pkl")
joblib.dump(FEATURES, "model_features.pkl")

results_df = pd.DataFrame(
    results,
    columns=["Model", "RMSE", "MAE", "MAPE (%)", "R2"]
)

results_df.to_csv("model_comparison_metrics.csv", index=False)

print("\n===================================")
print(" BEST MODEL SAVED SUCCESSFULLY")
print(" BEST RMSE:", round(best_rmse, 2))
print(" Metrics saved to model_comparison_metrics.csv")
print("===================================")
