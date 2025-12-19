import os
import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ===============================
# 0. Create Model Directory
# ===============================
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


# ===============================
# 1. Load Dataset
# ===============================
DATA_PATH = r"C:\Users\Vibhu\Desktop\Sales_Forecasting-Inventory_Management\data\Walmart_preprocessed_completed_PurchaseDate_2019_2024.csv"
df = pd.read_csv(DATA_PATH)


# ===============================
# 2. Age Segmentation
# ===============================
def age_to_group(age):
    if age <= 25:
        return 0
    elif age <= 35:
        return 1
    elif age <= 45:
        return 2
    elif age <= 60:
        return 3
    else:
        return 4

df["Age_Group"] = df["Age"].apply(age_to_group)


# ===============================
# 3. Features & Targets
# ===============================
FEATURES = [
    "Product_Name",
    "Brand",
    "Category",
    "Market_Price",
    "Discount_Applied",
    "Rating",
    "Market_Share",
    "Promotion_Competitor"
]

TARGETS = ["Age_Group", "Gender"]

X = df[FEATURES].copy()
y = df[TARGETS].copy()


# ===============================
# 4. Encode Categorical Features
# ===============================
label_encoders = {}
most_common_values = {}

for col in ["Product_Name", "Brand", "Category"]:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    most_common_values[col] = le.classes_[0]  # fallback value


# ===============================
# 5. Fix Yes / No Columns
# ===============================
for col in ["Promotion_Competitor", "Discount_Applied"]:
    X[col] = (
        X[col]
        .astype(str)
        .str.lower()
        .map({"yes": 1, "no": 0})
        .fillna(0)
        .astype(int)
    )


# ===============================
# 6. Numeric Safety
# ===============================
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)


# ===============================
# 7. Encode Target (Gender)
# ===============================
gender_le = LabelEncoder()
y["Gender"] = gender_le.fit_transform(y["Gender"].astype(str))
label_encoders["Gender"] = gender_le


# ===============================
# 8. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ===============================
# 9. Models
# ===============================
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42
    )
}


# ===============================
# 10. Training
# ===============================
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\n🚀 Training {name}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MultiOutputClassifier(model))
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    age_acc = accuracy_score(y_test["Age_Group"], preds[:, 0])
    gender_acc = accuracy_score(y_test["Gender"], preds[:, 1])

    print(f"Age Accuracy    : {age_acc:.3f}")
    print(f"Gender Accuracy : {gender_acc:.3f}")

    avg_score = (age_acc + gender_acc) / 2

    if avg_score > best_score:
        best_score = avg_score
        best_model = pipeline


# ===============================
# 11. Save Model
# ===============================
joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
joblib.dump(label_encoders, f"{MODEL_DIR}/label_encoders.pkl")
joblib.dump(most_common_values, f"{MODEL_DIR}/fallback_values.pkl")

print("\n✅ Model & encoders saved successfully!")


# ===============================
# 12. Safe Encoder Function
# ===============================
def safe_encode(value, encoder, fallback):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return encoder.transform([fallback])[0]


# ===============================
# 13. Prediction Function
# ===============================
def predict_customer_segment(product_name, brand, category):
    model = joblib.load(f"{MODEL_DIR}/best_model.pkl")
    encoders = joblib.load(f"{MODEL_DIR}/label_encoders.pkl")
    fallback = joblib.load(f"{MODEL_DIR}/fallback_values.pkl")

    input_df = pd.DataFrame([{
        "Product_Name": safe_encode(product_name, encoders["Product_Name"], fallback["Product_Name"]),
        "Brand": safe_encode(brand, encoders["Brand"], fallback["Brand"]),
        "Category": safe_encode(category, encoders["Category"], fallback["Category"]),
        "Market_Price": X["Market_Price"].mean(),
        "Discount_Applied": 0,
        "Rating": X["Rating"].mean(),
        "Market_Share": X["Market_Share"].mean(),
        "Promotion_Competitor": 0
    }])

    pred = model.predict(input_df)

    age_map = {
        0: "18–25",
        1: "26–35",
        2: "36–45",
        3: "46–60",
        4: "60+"
    }

    return {
        "Age_Group": age_map[pred[0][0]],
        "Gender": encoders["Gender"].inverse_transform([pred[0][1]])[0]
    }


# ===============================
# 14. Local Test
# ===============================
if __name__ == "__main__":
    print("\n🎯 Sample Prediction:")
    print(
        predict_customer_segment(
            product_name="Smartwatch",
            brand="Sony",
            category="Electronics"
        )
    )
