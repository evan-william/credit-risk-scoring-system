import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

# 1. Load Data (Pastikan file csv ada di folder yang sama atau sesuaikan path)
# Gunakan fungsi generate data dari chat sebelumnya jika belum ada csv
df = pd.read_csv('credit_risk_dataset.csv')

# 2. Feature Engineering Dasar
df['loan_to_income_ratio'] = df['loan_amount'] / df['annual_income']
df['monthly_burden'] = (df['loan_amount'] * (df['interest_rate']/100)) / 12

X = df.drop(['loan_status', 'customer_id'], axis=1, errors='ignore')
y = df['loan_status']

# 3. Pipeline Setup
numeric_features = ['age', 'annual_income', 'emp_length_years', 'loan_amount', 
                    'interest_rate', 'credit_history_years', 'loan_to_income_ratio', 'monthly_burden']
categorical_features = ['home_ownership', 'loan_intent']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Hitung scale_pos_weight
ratio = float(np.sum(y == 0)) / np.sum(y == 1)

# Full Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=ratio,
        eval_metric='logloss',
        use_label_encoder=False
    ))
])

# 4. Fit & Save
print("Sedang training model...")
model_pipeline.fit(X, y)
print("Training selesai. Menyimpan model ke 'credit_model_xgb.pkl'...")

# Simpan pipeline lengkap (termasuk scaler dan encoder)
joblib.dump(model_pipeline, 'credit_model_xgb.pkl')
print("Model berhasil disimpan!")