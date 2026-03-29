"""
preprocess.py
Loads raw churn data, cleans, encodes, and scales features.
Saves processed data and artifacts to data/processed/
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_PATH   = Path('data/raw/churn_data.csv')
PROC_DIR   = Path('data/processed')
PROC_DIR.mkdir(parents=True, exist_ok=True)


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.drop(columns=['CustomerID'], inplace=True)

    # Fill missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Feature engineering
    df['AvgMonthlySpend'] = (df['TotalCharges'] / df['Tenure'].replace(0, 1)).round(2)
    df['TenureGroup'] = pd.cut(df['Tenure'], bins=[0, 12, 24, 48, 72],
                                labels=['0-1yr', '1-2yr', '2-4yr', '4+yr'])
    return df


def encode_features(df: pd.DataFrame):
    binary_cols = ['Gender', 'Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling']
    multi_cols  = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                   'TechSupport', 'StreamingTV', 'Contract',
                   'PaymentMethod', 'TenureGroup']

    encoders = {}
    for col in binary_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    df = pd.get_dummies(df, columns=multi_cols)
    return df, encoders


def preprocess(raw_path: Path = RAW_PATH):
    print("📦 Loading raw data …")
    df = load_and_clean(raw_path)

    print("🔧 Encoding features …")
    df, encoders = encode_features(df)

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # Save artifacts
    pd.DataFrame(X_train_sc, columns=X.columns).to_csv(PROC_DIR / 'X_train.csv', index=False)
    pd.DataFrame(X_test_sc,  columns=X.columns).to_csv(PROC_DIR / 'X_test.csv',  index=False)
    y_train.reset_index(drop=True).to_csv(PROC_DIR / 'y_train.csv', index=False)
    y_test.reset_index(drop=True).to_csv(PROC_DIR / 'y_test.csv',  index=False)

    with open(PROC_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(PROC_DIR / 'encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    with open(PROC_DIR / 'feature_columns.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)

    print(f"✅ Processed data saved to {PROC_DIR}")
    print(f"   Train: {X_train_sc.shape}  |  Test: {X_test_sc.shape}")
    return X_train_sc, X_test_sc, y_train, y_test, scaler, list(X.columns)


if __name__ == '__main__':
    preprocess()
