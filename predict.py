"""
predict.py
Makes churn predictions for new customers.
Usage: python predict.py
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

PROC_DIR   = Path('data/processed')
MODELS_DIR = Path('models')


def load_model_artifacts():
    with open(MODELS_DIR / 'best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(PROC_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(PROC_DIR / 'encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    with open(PROC_DIR / 'feature_columns.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    return model, scaler, encoders, feature_cols


def predict_single(customer: dict) -> dict:
    """
    Predict churn for a single customer.

    Args:
        customer: dict with raw customer fields (same as raw CSV columns
                  excluding CustomerID and Churn).
    Returns:
        dict with 'churn_prediction' (0/1) and 'churn_probability' (float).
    """
    model, scaler, encoders, feature_cols = load_model_artifacts()

    df = pd.DataFrame([customer])

    # Binary label encoding
    binary_cols = ['Gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns and col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))

    # Feature engineering
    df['AvgMonthlySpend'] = (df['TotalCharges'] / df['Tenure'].replace(0, 1)).round(2)
    df['TenureGroup'] = pd.cut(df['Tenure'], bins=[0, 12, 24, 48, 72],
                                labels=['0-1yr', '1-2yr', '2-4yr', '4+yr'])

    # One-hot encoding
    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                  'TechSupport', 'StreamingTV', 'Contract',
                  'PaymentMethod', 'TenureGroup']
    df = pd.get_dummies(df, columns=multi_cols)

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]

    X_scaled = scaler.transform(df)
    pred     = model.predict(X_scaled)[0]
    prob     = model.predict_proba(X_scaled)[0][1]

    return {
        'churn_prediction': int(pred),
        'churn_probability': round(float(prob), 4),
        'risk_level': 'High' if prob >= 0.7 else ('Medium' if prob >= 0.4 else 'Low')
    }


def predict_batch(csv_path: str) -> pd.DataFrame:
    """Predict churn for a batch of customers from a CSV file."""
    df_raw = pd.read_csv(csv_path)
    id_col = 'CustomerID' if 'CustomerID' in df_raw.columns else None
    ids    = df_raw[id_col] if id_col else None

    records  = df_raw.drop(columns=[c for c in ['CustomerID', 'Churn'] if c in df_raw.columns])
    results  = []
    for _, row in records.iterrows():
        res = predict_single(row.to_dict())
        results.append(res)

    out = pd.DataFrame(results)
    if ids is not None:
        out.insert(0, 'CustomerID', ids.values)
    return out


# ---------- Demo ----------
SAMPLE_CUSTOMER = {
    'Gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'Age': 35,
    'Tenure': 5,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'NumServices': 3,
    'MonthlyCharges': 95.50,
    'TotalCharges': 477.50,
}

if __name__ == '__main__':
    result = predict_single(SAMPLE_CUSTOMER)
    print("\n>>> Churn Prediction Result <<<")
    print("=" * 40)
    for k, v in result.items():
        print(f"  {k:<25} : {v}")
    print("=" * 40)
