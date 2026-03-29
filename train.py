"""
train.py
Trains Logistic Regression, Random Forest, and XGBoost models.
Picks best model by ROC-AUC and saves all models to models/
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    classification_report, confusion_matrix
)

warnings.filterwarnings('ignore')

PROC_DIR   = Path('data/processed')
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print(" XGBoost not installed — skipping XGBClassifier.")


def load_processed():
    X_train = pd.read_csv(PROC_DIR / 'X_train.csv').values
    X_test  = pd.read_csv(PROC_DIR / 'X_test.csv').values
    y_train = pd.read_csv(PROC_DIR / 'y_train.csv').values.ravel()
    y_test  = pd.read_csv(PROC_DIR / 'y_test.csv').values.ravel()
    return X_train, X_test, y_train, y_test


def get_models():
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
    }
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            scale_pos_weight=3, eval_metric='logloss',
            tree_method='hist', device='cpu', random_state=42
        )
    return models


def train_and_evaluate():
    print(" Loading processed data …")
    X_train, X_test, y_train, y_test = load_processed()

    models   = get_models()
    results  = {}
    best_auc = 0
    best_name = None

    for name, model in models.items():
        print(f"\n  Training {name} …")
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average='binary')

        results[name] = {'AUC': auc, 'Accuracy': acc, 'F1': f1, 'model': model}

        print(f"   AUC: {auc:.4f}  |  Accuracy: {acc:.4f}  |  F1: {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

        # Save each model
        safe_name = name.lower().replace(' ', '_')
        with open(MODELS_DIR / f'{safe_name}.pkl', 'wb') as f:
            pickle.dump(model, f)

        if auc > best_auc:
            best_auc  = auc
            best_name = name

    print(f"\n Best model: {best_name}  (AUC = {best_auc:.4f})")
    with open(MODELS_DIR / 'best_model.pkl', 'wb') as f:
        pickle.dump(results[best_name]['model'], f)
    with open(MODELS_DIR / 'best_model_name.txt', 'w') as f:
        f.write(best_name)

    # Save results summary
    summary = pd.DataFrame(
        {k: {m: v for m, v in vv.items() if m != 'model'}
         for k, vv in results.items()}
    ).T
    summary.to_csv(MODELS_DIR / 'results_summary.csv')
    print(f"\n All models saved to {MODELS_DIR}/")
    return results, best_name


if __name__ == '__main__':
    train_and_evaluate()
print('acc')    
