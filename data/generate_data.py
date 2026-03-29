"""
generate_data.py
Generates synthetic customer churn dataset and saves to data/raw/churn_data.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 5000

def generate_churn_data(n=N):
    tenure         = np.random.randint(1, 72, n)
    age            = np.random.randint(18, 75, n)
    monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
    total_charges  = np.round(monthly_charges * tenure + np.random.normal(0, 50, n), 2)
    total_charges  = np.clip(total_charges, 0, None)

    contract       = np.random.choice(['Month-to-month', 'One year', 'Two year'], n,
                                       p=[0.55, 0.25, 0.20])
    internet       = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22])
    payment        = np.random.choice(['Electronic check', 'Mailed check',
                                        'Bank transfer', 'Credit card'], n)
    gender         = np.random.choice(['Male', 'Female'], n)
    senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner        = np.random.choice(['Yes', 'No'], n)
    dependents     = np.random.choice(['Yes', 'No'], n)
    phone_service  = np.random.choice(['Yes', 'No'], n, p=[0.90, 0.10])
    multiple_lines = np.random.choice(['Yes', 'No', 'No phone service'], n)
    online_security= np.random.choice(['Yes', 'No', 'No internet service'], n)
    tech_support   = np.random.choice(['Yes', 'No', 'No internet service'], n)
    streaming_tv   = np.random.choice(['Yes', 'No', 'No internet service'], n)
    paperless      = np.random.choice(['Yes', 'No'], n)
    num_services   = np.random.randint(1, 7, n)

    # Churn probability based on realistic factors
    churn_prob = (
        0.05
        + 0.25 * (contract == 'Month-to-month')
        + 0.15 * (internet == 'Fiber optic')
        + 0.10 * (payment == 'Electronic check')
        + 0.08 * (senior_citizen == 1)
        - 0.15 * (tenure > 36)
        - 0.10 * (num_services > 4)
        + 0.05 * (monthly_charges > 80)
        - 0.08 * (online_security == 'Yes')
        - 0.06 * (tech_support == 'Yes')
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn = (np.random.rand(n) < churn_prob).astype(int)

    customer_ids = [f'CUST-{str(i).zfill(5)}' for i in range(1, n + 1)]

    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'Age': age,
        'Tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet,
        'OnlineSecurity': online_security,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'NumServices': num_services,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn
    })
    return df


if __name__ == '__main__':
    raw_dir = Path(__file__).parent / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    df = generate_churn_data()
    out_path = raw_dir / 'churn_data.csv'
    df.to_csv(out_path, index=False)
    print(f"[OK] Dataset saved to {out_path}  |  Shape: {df.shape}")
    print(f"   Churn rate: {df['Churn'].mean():.2%}")
