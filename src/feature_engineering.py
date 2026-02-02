"""
Feature Engineering for Credit Risk Model
Advanced feature creation and transformation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class CreditFeatureEngine:
    """Advanced feature engineering for credit risk prediction"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = None
        
    def create_features(self, df, is_training=True):
        """Create advanced features from raw data"""
        
        df = df.copy()
        
        print("🔧 Creating advanced features...")
        
        # 1. DEBT-TO-INCOME RATIO (Critical for credit risk)
        df['monthly_payment'] = df['loan_amount'] / df['loan_term']
        df['dti_ratio'] = (df['existing_debt'] + df['monthly_payment']) / df['monthly_income']
        df['dti_ratio'] = df['dti_ratio'].clip(0, 10)  # Cap extreme values
        
        # 2. LOAN-TO-INCOME RATIO
        df['loan_to_income'] = df['loan_amount'] / (df['monthly_income'] * 12)
        df['loan_to_income'] = df['loan_to_income'].clip(0, 20)
        
        # 3. AVAILABLE INCOME (after existing debts and new loan)
        df['available_income'] = df['monthly_income'] - df['existing_debt'] - df['monthly_payment']
        df['available_income_ratio'] = df['available_income'] / df['monthly_income']
        
        # 4. CREDIT UTILIZATION
        # Estimate credit limit based on income
        df['estimated_credit_limit'] = df['monthly_income'] * 3
        df['credit_utilization'] = (df['existing_debt'] / df['estimated_credit_limit']).clip(0, 2)
        
        # 5. RISK SCORE COMPONENTS
        df['payment_risk_score'] = (100 - df['payment_history_pct']) + (df['late_payments'] * 5)
        df['inquiry_risk_score'] = df['credit_inquiries'] * 10
        
        # 6. CREDIT SCORE CATEGORIES
        df['credit_score_category'] = pd.cut(
            df['credit_score'],
            bins=[0, 580, 670, 740, 800, 850],
            labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
        )
        
        # 7. AGE GROUPS
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['Young', 'Early Career', 'Mid Career', 'Senior', 'Retirement']
        )
        
        # 8. INCOME STABILITY INDICATOR
        df['income_stability'] = (
            (df['employment_type'] == 'Full-Time').astype(int) * 0.4 +
            (df['employment_length'] > 2).astype(int) * 0.3 +
            (df['home_ownership'].isin(['Own', 'Mortgage'])).astype(int) * 0.3
        )
        
        # 9. DEBT BURDEN CATEGORY
        df['debt_burden'] = pd.cut(
            df['dti_ratio'],
            bins=[0, 0.3, 0.5, 1.0, 10],
            labels=['Low', 'Moderate', 'High', 'Very High']
        )
        
        # 10. CREDIT EXPERIENCE
        df['credit_experience'] = df['credit_history_length'] * df['existing_loans']
        df['avg_loan_size'] = np.where(
            df['existing_loans'] > 0,
            df['existing_debt'] / df['existing_loans'],
            0
        )
        
        # 11. FAMILY FINANCIAL PRESSURE
        df['income_per_person'] = df['monthly_income'] / (df['dependents'] + 1)
        
        # 12. LOAN BURDEN SCORE
        df['loan_burden_score'] = (
            df['dti_ratio'] * 0.3 +
            df['loan_to_income'] * 0.2 +
            (1 - df['available_income_ratio']) * 0.25 +
            df['credit_utilization'] * 0.25
        )
        
        # 13. STABILITY SCORE
        df['stability_score'] = (
            (df['employment_length'] / 40) * 0.3 +
            (df['credit_history_length'] / 30) * 0.3 +
            df['income_stability'] * 0.4
        )
        
        # 14. RISK FLAGS
        df['high_dti_flag'] = (df['dti_ratio'] > 0.5).astype(int)
        df['bankruptcy_flag'] = df['bankruptcy_history']
        df['recent_inquiries_flag'] = (df['credit_inquiries'] > 3).astype(int)
        df['poor_payment_flag'] = (df['payment_history_pct'] < 80).astype(int)
        df['multiple_late_payments_flag'] = (df['late_payments'] > 2).astype(int)
        
        # 15. POLYNOMIAL FEATURES for important ratios
        df['dti_squared'] = df['dti_ratio'] ** 2
        df['credit_score_normalized'] = (df['credit_score'] - 300) / 550
        df['credit_score_squared'] = df['credit_score_normalized'] ** 2
        
        # 16. INTERACTION FEATURES
        df['score_x_dti'] = df['credit_score_normalized'] * (1 / (df['dti_ratio'] + 0.1))
        df['income_x_stability'] = df['monthly_income'] * df['stability_score']
        df['age_x_credit_history'] = df['age'] * df['credit_history_length']
        
        print(f"✅ Created {len(df.columns)} total features")
        
        return df
    
    def encode_and_scale(self, df, is_training=True):
        """Encode categorical variables and scale numerical features"""
        
        df = df.copy()
        
        # Separate features and target
        if 'default' in df.columns:
            target = df['default']
            df = df.drop('default', axis=1)
        else:
            target = None
        
        # Drop ID and date columns
        cols_to_drop = ['application_id', 'application_date']
        df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1)
        
        # Identify categorical and numerical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"📊 Encoding {len(categorical_cols)} categorical features...")
        print(f"📊 Scaling {len(numerical_cols)} numerical features...")
        
        # Encode categorical variables
        for col in categorical_cols:
            if is_training:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                le = self.encoders.get(col)
                if le:
                    # Handle unseen categories
                    df[col + '_encoded'] = df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
            df = df.drop(col, axis=1)
        
        # Scale numerical features
        if is_training:
            scaler = StandardScaler()
            df_scaled = pd.DataFrame(
                scaler.fit_transform(df),
                columns=df.columns,
                index=df.index
            )
            self.scalers['standard'] = scaler
            self.feature_names = df.columns.tolist()
        else:
            scaler = self.scalers['standard']
            df_scaled = pd.DataFrame(
                scaler.transform(df),
                columns=df.columns,
                index=df.index
            )
        
        # Add target back if exists
        if target is not None:
            df_scaled['default'] = target
        
        print(f"✅ Final feature count: {len(df_scaled.columns) - (1 if target is not None else 0)}")
        
        return df_scaled
    
    def get_feature_importance_names(self):
        """Get feature names for importance plotting"""
        return self.feature_names if self.feature_names else []


if __name__ == "__main__":
    # Test the feature engineering
    print("Testing Feature Engineering...")
    
    df = pd.read_csv('../data/credit_train.csv')
    
    engine = CreditFeatureEngine()
    df_featured = engine.create_features(df, is_training=True)
    df_final = engine.encode_and_scale(df_featured, is_training=True)
    
    print(f"\n📋 Final Dataset Shape: {df_final.shape}")
    print(f"\n🎯 Target Distribution:")
    print(df_final['default'].value_counts())
    print(f"\n📊 Sample Features:\n{df_final.head()}")