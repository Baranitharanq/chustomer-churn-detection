
# 📉 Customer Churn Detection

A complete end-to-end machine learning pipeline for predicting customer churn, featuring data generation, preprocessing, model training (Logistic Regression, Random Forest, XGBoost), evaluation reports, and an interactive Streamlit dashboard.

---

## 🗂 Project Structure

```
customer churn detection project/
├── data/
│   ├── generate_data.py      # Synthetic dataset generator
│   ├── raw/                  # Raw CSV data
│   └── processed/            # Scaled features & artifacts
├── models/                   # Saved trained models
├── reports/                  # Evaluation plots (PNG)
├── preprocess.py             # Cleaning, encoding, scaling
├── train.py                  # Train & compare models
├── evaluate.py               # Plots: ROC, CM, PR, FI
├── predict.py                # Single / batch inference
├── app.py                    # Streamlit dashboard
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset
```bash
python data/generate_data.py
```

### 3. Preprocess data
```bash
python preprocess.py
```

### 4. Train models
```bash
python train.py
```

### 5. Evaluate & generate plots
```bash
python evaluate.py
```

### 6. Launch Web Server
```bash
python fastapi_app.py
```

### 7. Predict a single customer
```bash
python predict.py
```

---

## 🧠 Models Trained

| Model | Description |
|---|---|
| Logistic Regression | Fast baseline with L2 regularization |
| Random Forest | Ensemble of 200 decision trees |
| XGBoost | Gradient boosted trees (best AUC) |

---

## 📊 Features Used

| Category | Features |
|---|---|
| Demographics | Gender, Age, SeniorCitizen, Partner, Dependents |
| Services | PhoneService, InternetService, OnlineSecurity, TechSupport, StreamingTV |
| Billing | Contract, PaymentMethod, MonthlyCharges, TotalCharges, Tenure |
| Engineered | AvgMonthlySpend, TenureGroup, NumServices |

---

## 📈 Dashboard Features

- **Real-time churn probability gauge** for any customer profile
- **Risk level classification** (High / Medium / Low)
- **Key risk factor analysis**
- **Model performance comparison charts**
- **Batch CSV prediction** with downloadable results

---

## 📋 Reports Generated

- `reports/confusion_matrix.png`
- `reports/roc_curve.png`
- `reports/precision_recall.png`
- `reports/feature_importance.png`
>>>>>>> 2c9f9df (initialcommit)
