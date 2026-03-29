"""
fastapi_app.py  —  FastAPI Customer Churn Detection API + Web UI
Run: python -m uvicorn fastapi_app:app --reload
Then open: http://127.0.0.1:8000
"""
import io
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC_DIR   = Path("data/processed")
MODELS_DIR = Path("models")

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Detection API",
    description="ML-powered churn prediction for telecom customers",
    version="1.0.0",
)

# Mount templates
templates = Jinja2Templates(directory="templates")

# ── Load Artifacts ────────────────────────────────────────────────────────────
def load_artifacts():
    try:
        with open(MODELS_DIR / "best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(PROC_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(PROC_DIR / "encoders.pkl", "rb") as f:
            encoders = pickle.load(f)
        with open(PROC_DIR / "feature_columns.pkl", "rb") as f:
            feature_cols = pickle.load(f)
        with open(MODELS_DIR / "best_model_name.txt") as f:
            model_name = f.read().strip()
        return model, scaler, encoders, feature_cols, model_name
    except FileNotFoundError as e:
        return None, None, None, None, None

MODEL, SCALER, ENCODERS, FEATURE_COLS, MODEL_NAME = load_artifacts()

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_input(customer: dict) -> np.ndarray:
    df = pd.DataFrame([customer])
    binary_cols = ["Gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in binary_cols:
        if col in df.columns and col in ENCODERS:
            df[col] = ENCODERS[col].transform(df[col].astype(str))
    df["AvgMonthlySpend"] = (df["TotalCharges"] / df["Tenure"].replace(0, 1)).round(2)
    df["TenureGroup"] = pd.cut(
        df["Tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"]
    )
    multi_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "TechSupport", "StreamingTV", "Contract", "PaymentMethod", "TenureGroup"
    ]
    df = pd.get_dummies(df, columns=multi_cols)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_COLS]
    return SCALER.transform(df)


def risk_label(prob: float) -> str:
    return "High" if prob >= 0.7 else ("Medium" if prob >= 0.4 else "Low")

# ── Pydantic Schema ───────────────────────────────────────────────────────────
class CustomerInput(BaseModel):
    Gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    Age: int
    Tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    TechSupport: str
    StreamingTV: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    NumServices: int
    MonthlyCharges: float
    TotalCharges: float

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main web UI."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model_name": MODEL_NAME or "Model Not Loaded",
            "model_ready": MODEL is not None,
        }
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "model_name": MODEL_NAME}


@app.post("/predict")
def predict_single(customer: CustomerInput):
    """Predict churn for a single customer."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run the training pipeline first.")
    X = preprocess_input(customer.dict())
    prob = float(MODEL.predict_proba(X)[0][1])
    pred = int(MODEL.predict(X)[0])
    return {
        "churn_prediction": pred,
        "churn_probability": round(prob, 4),
        "risk_level": risk_label(prob),
        "model_used": MODEL_NAME,
    }


@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Predict churn for a batch CSV upload."""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    content = await file.read()
    df_raw = pd.read_csv(io.BytesIO(content))
    id_col = "CustomerID" if "CustomerID" in df_raw.columns else None
    ids = df_raw[id_col] if id_col else None
    records = df_raw.drop(columns=[c for c in ["CustomerID", "Churn"] if c in df_raw.columns])

    results = []
    for _, row in records.iterrows():
        X = preprocess_input(row.to_dict())
        prob = float(MODEL.predict_proba(X)[0][1])
        pred = int(MODEL.predict(X)[0])
        results.append({
            "churn_prediction": pred,
            "churn_probability": round(prob, 4),
            "risk_level": risk_label(prob),
        })

    out = pd.DataFrame(results)
    if ids is not None:
        out.insert(0, "CustomerID", ids.values)

    csv_bytes = out.to_csv(index=False).encode()
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=churn_predictions.csv"},
    )


@app.get("/model/summary")
def model_summary():
    """Return model comparison metrics."""
    path = MODELS_DIR / "results_summary.csv"
    if not path.exists():
        return {"available": False, "data": []}
    df = pd.read_csv(path, index_col=0)
    records = []
    for name, row in df.iterrows():
        records.append({
            "model": name,
            "auc": round(row.get("AUC", 0), 4),
            "accuracy": round(row.get("Accuracy", 0), 4),
            "f1": round(row.get("F1", 0), 4),
        })
    return {"available": True, "data": records}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
