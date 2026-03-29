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
# Helper: train and save model (only when needed)
# ===============================
def train_and_save():
    DATA_PATH = r"C:\Users\Vibhu\Desktop\Sales_Forecasting-Inventory_Management\data\Walmart_preprocessed_completed_PurchaseDate_2019_2024.csv"
    df = pd.read_csv(DATA_PATH)

    def age_to_group(age):
        try:
            age = float(age)
        except Exception:
            return 4
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

    # Encode categorical features
    label_encoders = {}
    most_common_values = {}
    for col in ["Product_Name", "Brand", "Category"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        most_common_values[col] = le.classes_[0] if len(le.classes_)>0 else ''

    # Fix Yes/No
    for col in ["Promotion_Competitor", "Discount_Applied"]:
        if col in X.columns:
            X[col] = (
                X[col]
                .astype(str)
                .str.lower()
                .map({"yes": 1, "no": 0})
                .fillna(0)
            )

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

    gender_le = LabelEncoder()
    y["Gender"] = gender_le.fit_transform(y["Gender"].astype(str))
    label_encoders["Gender"] = gender_le

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    }

    best_model = None
    best_score = -1
    for name, model in models.items():
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", MultiOutputClassifier(model))])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        age_acc = accuracy_score(y_test["Age_Group"], preds[:, 0])
        gender_acc = accuracy_score(y_test["Gender"], preds[:, 1])
        avg_score = (age_acc + gender_acc) / 2
        if avg_score > best_score:
            best_score = avg_score
            best_model = pipeline

    # Save artifacts
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'best_model.pkl'))
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, 'label_encoders.pkl'))
    joblib.dump(most_common_values, os.path.join(MODEL_DIR, 'fallback_values.pkl'))

    return best_model, label_encoders, most_common_values


def safe_encode(value, encoder, fallback):
    try:
        if value in encoder.classes_:
            return int(encoder.transform([value])[0])
    except Exception:
        pass
    # if fallback not in classes, return 0
    try:
        return int(encoder.transform([fallback])[0])
    except Exception:
        return 0


def predict_customer_segment(product_name, brand, category):
    # Load saved model and encoders if present; else train
    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    enc_path = os.path.join(MODEL_DIR, 'label_encoders.pkl')
    fallback_path = os.path.join(MODEL_DIR, 'fallback_values.pkl')

    if not (os.path.exists(model_path) and os.path.exists(enc_path) and os.path.exists(fallback_path)):
        train_and_save()

    model = joblib.load(model_path)
    encoders = joblib.load(enc_path)
    fallback = joblib.load(fallback_path)

    # Build input vector similar to training
    # Use average numeric stats by reloading dataset minimal fields
    DATA_PATH = r"C:\Users\Vibhu\Desktop\Sales_Forecasting-Inventory_Management\data\Walmart_preprocessed_completed_PurchaseDate_2019_2024.csv"
    df = pd.read_csv(DATA_PATH)
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

    input_df = pd.DataFrame([{
        "Product_Name": safe_encode(product_name, encoders["Product_Name"], fallback.get("Product_Name", '')),
        "Brand": safe_encode(brand, encoders["Brand"], fallback.get("Brand", '')),
        "Category": safe_encode(category, encoders["Category"], fallback.get("Category", '')),
        "Market_Price": float(df["Market_Price"].mean()) if "Market_Price" in df.columns else 0,
        "Discount_Applied": 0,
        "Rating": float(df["Rating"].mean()) if "Rating" in df.columns else 0,
        "Market_Share": float(df["Market_Share"].mean()) if "Market_Share" in df.columns else 0,
        "Promotion_Competitor": 0
    }])

    pred = model.predict(input_df)

    age_map = {0: "18–25", 1: "26–35", 2: "36–45", 3: "46–60", 4: "60+"}
    age_label = age_map.get(int(pred[0][0]), "36–45")

    try:
        gender_label = encoders["Gender"].inverse_transform([int(pred[0][1])])[0]
    except Exception:
        gender_label = str(pred[0][1])

    return {"Age_Group": age_label, "Gender": gender_label}


if __name__ == "__main__":
    print("\n🎯 Sample Prediction:")
    print(predict_customer_segment(product_name="Smartwatch", brand="Sony", category="Electronics"))
