# =========================================================
#   WALMART MULTI-AGENT INTELLIGENCE SYSTEM (FASTAPI)
# =========================================================

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import re
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from dotenv import load_dotenv
import os
import requests
from typing import Optional, Dict

# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---------------------------------------------------------
# Load dataset and model
# ---------------------------------------------------------
DATA_PATH = r"C:\Users\Vibhu\Desktop\Sales_Forecasting-Inventory_Management\data\Walmart_preprocessed_completed_PurchaseDate_2019_2024.csv"
df = pd.read_csv(DATA_PATH)

model = joblib.load("best_price_forecast_model.pkl")
FEATURES = joblib.load("model_features.pkl")

# ---------------------------------------------------------
# Preserve Product_Name mapping BEFORE encoding
# ---------------------------------------------------------
product_encoder = LabelEncoder()
df["Product_Name_Original"] = df["Product_Name"]
df["Product_Name"] = product_encoder.fit_transform(df["Product_Name"].astype(str))

product_decoder = dict(
    zip(df["Product_Name"], df["Product_Name_Original"])
)

# ---------------------------------------------------------
# Boolean mapping
# ---------------------------------------------------------
bool_map = {"Yes": 1, "No": 0, "yes": 1, "no": 0, True: 1, False: 0}
for col in df.columns:
    if df[col].dtype == object:
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset(bool_map.keys()):
            df[col] = df[col].map(bool_map)

# ---------------------------------------------------------
# Date features
# ---------------------------------------------------------
df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"], errors="coerce")
df["Year"] = df["Purchase_Date"].dt.year
df["Month"] = df["Purchase_Date"].dt.month
df["Week"] = df["Purchase_Date"].dt.isocalendar().week.astype(float)

# ---------------------------------------------------------
# Encode remaining categorical columns (EXCEPT Product_Name_Original)
# ---------------------------------------------------------
categorical_cols = df.select_dtypes(include=["object"]).columns
encoder = LabelEncoder()
for col in categorical_cols:
    if col != "Product_Name_Original":
        df[col] = encoder.fit_transform(df[col].astype(str))

# ---------------------------------------------------------
# Handle missing values
# ---------------------------------------------------------
imputer = SimpleImputer(strategy="median")
df[FEATURES] = imputer.fit_transform(df[FEATURES])

# ---------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------
app = FastAPI(title="Walmart Multi-Agent Intelligence API")

# ---------------------------------------------------------
# Request model
# ---------------------------------------------------------
class AnalyzeRequest(BaseModel):
    future_start: str
    future_end: str

# =========================================================
# AGENT 1: Total Future Cost Forecasting
# =========================================================
def price_forecast_agent(start_date, end_date):
    future_dates = pd.date_range(start=start_date, end=end_date)
    if future_dates.empty:
        return None

    all_products = df["Product_Name_Original"].unique()
    future_list = [{"Purchase_Date": d, "Product_Name": p} for p in all_products for d in future_dates]

    future_df = pd.DataFrame(future_list)
    # Safe encoding
    future_df["Product_Name"] = future_df["Product_Name"].apply(
        lambda x: product_encoder.transform([x])[0] if x in product_encoder.classes_ else 0
    )
    future_df["Year"] = future_df["Purchase_Date"].dt.year
    future_df["Month"] = future_df["Purchase_Date"].dt.month
    future_df["Week"] = future_df["Purchase_Date"].dt.isocalendar().week.astype(float)

    for col in FEATURES:
        if col not in future_df.columns:
            future_df[col] = df[col].median()

    predictions = model.predict(future_df[FEATURES])
    total_predicted_cost = predictions.sum()
    return round(float(total_predicted_cost), 2)

# =========================================================
# AGENT 2: Historical Top Products (Past Years, Same Months)
# =========================================================
def top_selling_agent(future_start, future_end, top_n=5):
    start = pd.to_datetime(future_start)
    end = pd.to_datetime(future_end)
    past_df = df[(df["Month"] >= start.month) & (df["Month"] <= end.month) & (df["Year"] < start.year)]
    if past_df.empty:
        return pd.DataFrame()
    grouped = (
        past_df.groupby("Product_Name")["Purchase_Amount"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    grouped["Product_Name"] = grouped["Product_Name"].map(product_decoder)
    return grouped

# =========================================================
# AGENT 3: Sales Growth Advice Agent
# =========================================================
# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")  # Get your NewsAPI key from newsapi.org

def advice_generation_agent(product, sales, top_articles=5):
    """
    Generates advice using recent news related to the product.
    """
    target_sales = round(sales * 1.3, 2)
    
    # Query News API for relevant articles
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"Walmart {product} sales OR marketing OR retail strategy",
        "apiKey": NEWSAPI_KEY,
        "pageSize": top_articles,
        "language": "en",
        "sortBy": "relevancy"
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        articles = data.get("articles", [])

        if not articles:
            return f"No recent news found for {product}. Consider traditional retail strategies."

        # Collect article titles + descriptions
        advice_snippets = []
        for a in articles:
            title = a.get("title", "")
            desc = a.get("description", "")
            snippet = f"- {title}. {desc}" if desc else f"- {title}"
            advice_snippets.append(snippet)

        # Combine into advice message
        advice_text = f"Product: {product}\nHistorical Sales: ${sales}\nTarget Sales: ${target_sales}\n\nRecent industry news insights:\n"
        advice_text += "\n".join(advice_snippets)
        advice_text += "\n\nConsider these insights when planning promotions, inventory, and marketing strategies."
        
        return advice_text

    except Exception as e:
        print("Error fetching news:", str(e))
        return "News service unavailable. Try again later."

# =========================================================
# AGENT 4: Predicted Top Products for Future Period
# =========================================================
def predicted_top_selling_agent(start_date, end_date, top_n=5):
    future_dates = pd.date_range(start=start_date, end=end_date)
    all_products = df["Product_Name_Original"].unique()
    future_list = [{"Purchase_Date": d, "Product_Name": p} for p in all_products for d in future_dates]

    future_df = pd.DataFrame(future_list)
    # Safe encoding
    future_df["Product_Name"] = future_df["Product_Name"].apply(
        lambda x: product_encoder.transform([x])[0] if x in product_encoder.classes_ else 0
    )
    future_df["Year"] = future_df["Purchase_Date"].dt.year
    future_df["Month"] = future_df["Purchase_Date"].dt.month
    future_df["Week"] = future_df["Purchase_Date"].dt.isocalendar().week.astype(float)

    for col in FEATURES:
        if col not in future_df.columns:
            future_df[col] = df[col].median()

    future_df["Predicted_Amount"] = model.predict(future_df[FEATURES])

    top_products = (
        future_df.groupby("Product_Name")["Predicted_Amount"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    top_products["Product_Name"] = top_products["Product_Name"].map(product_decoder)
    return top_products

def important_advice(top_products_future, future_start, future_end):
    """
    Generates instant advice for the user based on predicted top-selling products
    and the user-selected future period.
    """
    if top_products_future.empty:
        return "No predicted top products available for this period."
    
    product_names = top_products_future["Product_Name"].tolist()
    period_str = f"{future_start} → {future_end}"
    
    advice_text = (
        f"Important Advice for period {period_str}:\n"
        f"Consider keeping large stock of the following top products:\n"
        f"- " + "\n- ".join(product_names)
    )
    
    return advice_text


# =========================================================
# ORCHESTRATION ENDPOINT
# =========================================================
@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        # 1. Total future cost
        future_total_cost = price_forecast_agent(req.future_start, req.future_end)
        
        # 2. Historical top products
        top_products_historical = top_selling_agent(req.future_start, req.future_end)
        
        # 3. Predicted top products for future period
        top_products_future = predicted_top_selling_agent(req.future_start, req.future_end)
        
        if future_total_cost is None or top_products_historical.empty:
            return {"message": "Insufficient data for analysis"}
        
        # 4. AI-generated advice (News API)
        best_product = top_products_historical.iloc[0]
        ai_advice = advice_generation_agent(
            best_product["Product_Name"],
            best_product["Purchase_Amount"]
        )
        
        # 5. Instant important advice based on top 5 future products
        important = important_advice(top_products_future, req.future_start, req.future_end)
        
        # Build forecast part
        forecast_resp = {
            "future_total_cost_prediction": future_total_cost,
            "historical_top_products": top_products_historical.to_dict(orient="records"),
            "predicted_top_products": top_products_future.to_dict(orient="records"),
            "ai_generated_advice": ai_advice,
            "important_advice": important
        }

        # Try to build a campaign plan for the top historical product (best-effort)
        campaign_resp = {"plan": {}, "csv_path": "N/A", "json_path": "N/A"}
        try:
            best_product = top_products_historical.iloc[0] if not top_products_historical.empty else None
            if best_product is not None:
                best_product_name = best_product["Product_Name"]

                # Try to locate reasonable brand/category columns in the dataset
                brand_col = next((c for c in ["Brand", "brand", "Brand_Name", "Manufacturer"] if c in df.columns), None)
                category_col = next((c for c in ["Category", "category", "Product_Category", "Category_Name"] if c in df.columns), None)

                if brand_col and category_col:
                    # Find the most common brand/category for the product
                    product_rows = df[df["Product_Name_Original"] == best_product_name]
                    if not product_rows.empty:
                        brand_val = str(product_rows[brand_col].mode().iloc[0]) if brand_col in product_rows.columns else ""
                        category_val = str(product_rows[category_col].mode().iloc[0]) if category_col in product_rows.columns else ""

                        # Lazy import planner and produce plan + save reports
                        from backend.plan_campaign import plan_for_input, write_report_csv

                        plan_output = plan_for_input(best_product_name, brand_val, category_val, impressions_by_platform=None)
                        csv_path, json_path = write_report_csv(plan_output, cpms_override=None, impressions_override=None, out_dir=os.path.dirname(__file__))

                        # Return only the filename (basename) to the client
                        csv_name = os.path.basename(csv_path) if isinstance(csv_path, str) else csv_path
                        json_name = os.path.basename(json_path) if isinstance(json_path, str) else json_path
                        campaign_resp = {"plan": plan_output, "csv_path": csv_name, "json_path": json_name}
                else:
                    # No brand/category columns found -- leave campaign_resp as N/A
                    campaign_resp = {"plan": {}, "csv_path": "N/A", "json_path": "N/A"}
        except Exception as e:
            # If any error occurs, include minimal error info but don't break the main response
            campaign_resp = {"plan": {}, "csv_path": "N/A", "json_path": "N/A", "error": str(e)}

        # Return forecast fields at top-level for backward compatibility,
        # and keep the `forecast` and `campaign` sections for the new UI.
        combined = {**forecast_resp, "forecast": forecast_resp, "campaign": campaign_resp}
        return combined
        
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------
# Campaign planning endpoint (uses customer segmentation)
# ---------------------------------------------------------
class CampaignRequest(BaseModel):
    product_name: str
    brand: str
    category: str
    impressions: Optional[Dict[str, int]] = None
    cpms: Optional[Dict[str, float]] = None


@app.post('/campaign')
def campaign_planner(req: CampaignRequest):
    try:
        # import planner lazily
        from backend.plan_campaign import plan_for_input, write_report_csv

        output = plan_for_input(req.product_name, req.brand, req.category, impressions_by_platform=req.impressions)
        csv_path, json_path = write_report_csv(output, cpms_override=req.cpms, impressions_override=req.impressions, out_dir=os.path.dirname(__file__))

        # Return filenames (basename) rather than full filesystem paths
        csv_name = os.path.basename(csv_path) if isinstance(csv_path, str) else csv_path
        json_name = os.path.basename(json_path) if isinstance(json_path, str) else json_path

        return {
            'output': output,
            'csv_path': csv_name,
            'json_path': json_name
        }
    except Exception as e:
        return {'error': str(e)}


@app.get('/reports/{filename}')
def get_report(filename: str):
    """Serve campaign report files saved in the backend directory.
    Only allows files with the pattern `campaign_report_YYYYMMDD_HHMMSS.(csv|json)`.
    """
    safe_dir = os.path.dirname(__file__)

    # Validate filename pattern to avoid path traversal
    if not re.match(r'^campaign_report_\d{8}_\d{6}\.(csv|json)$', filename):
        raise HTTPException(status_code=400, detail='Invalid filename')

    file_path = os.path.abspath(os.path.join(safe_dir, filename))
    if not file_path.startswith(os.path.abspath(safe_dir) + os.sep):
        raise HTTPException(status_code=400, detail='Invalid path')

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail='File not found')

    return FileResponse(path=file_path, media_type='application/octet-stream', filename=filename)


@app.get('/assets/ai_robo.webp')
def get_ai_robo():
    """Serve the AI robo image from the project's `data` folder.
    This allows the frontend to reference `/assets/ai_robo.webp` without copying images.
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    # original filename contains a space; normalize to the expected name
    candidates = [
        os.path.join(data_dir, 'AI robo.webp'),
        os.path.join(data_dir, 'AI_robo.webp'),
        os.path.join(data_dir, 'ai_robo.webp')
    ]
    for p in candidates:
        if os.path.exists(p):
            return FileResponse(path=os.path.abspath(p), media_type='image/webp', filename=os.path.basename(p))

    raise HTTPException(status_code=404, detail='AI robo image not found')
