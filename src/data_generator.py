import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os
from pathlib import Path

np.random.seed(67)
random.seed(67)

class CreditDataGenerator:
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        
    def generate_data(self):
        print(f"Generating {self.n_samples} credit applications...")
        
        # Basic Demographics
        age = np.random.normal(40, 12, self.n_samples).clip(18, 75)
        
        # Education levels with realistic distribution
        education_levels = np.random.choice(
            ['High School', 'Bachelor', 'Master', 'PhD', 'Vocational'],
            size=self.n_samples,
            p=[0.30, 0.40, 0.20, 0.05, 0.05]
        )
        
        # Employment type
        employment_types = np.random.choice(
            ['Full-Time', 'Part-Time', 'Self-Employed', 'Unemployed', 'Retired'],
            size=self.n_samples,
            p=[0.60, 0.15, 0.15, 0.05, 0.05]
        )
        
        # Marital status
        marital_status = np.random.choice(
            ['Single', 'Married', 'Divorced', 'Widowed'],
            size=self.n_samples,
            p=[0.35, 0.45, 0.15, 0.05]
        )
        
        # Number of dependents
        dependents = np.random.choice([0, 1, 2, 3, 4, 5], 
                                     size=self.n_samples,
                                     p=[0.25, 0.20, 0.25, 0.15, 0.10, 0.05])
        
        # Income - correlated with education and employment
        base_income = np.random.lognormal(10.5, 0.6, self.n_samples)
        education_multiplier = np.where(education_levels == 'PhD', 1.5,
                               np.where(education_levels == 'Master', 1.3,
                               np.where(education_levels == 'Bachelor', 1.1, 1.0)))
        
        employment_multiplier = np.where(employment_types == 'Full-Time', 1.2,
                                np.where(employment_types == 'Self-Employed', 1.1,
                                np.where(employment_types == 'Part-Time', 0.7,
                                np.where(employment_types == 'Retired', 0.6, 0.3))))
        
        monthly_income = (base_income * education_multiplier * employment_multiplier).clip(500, 50000)
        
        # Loan amount - typically correlated with income
        loan_amount = (monthly_income * np.random.uniform(3, 15, self.n_samples)).clip(1000, 500000)
        
        # Loan term in months
        loan_term = np.random.choice([12, 24, 36, 48, 60, 72], 
                                    size=self.n_samples,
                                    p=[0.10, 0.20, 0.30, 0.20, 0.15, 0.05])
        
        # Interest rate - varies by risk
        interest_rate = np.random.uniform(5, 25, self.n_samples)
        
        # Credit history length in years
        credit_history_length = np.random.gamma(3, 2, self.n_samples).clip(0, 30)
        
        # Number of existing loans
        existing_loans = np.random.choice([0, 1, 2, 3, 4, 5], 
                                         size=self.n_samples,
                                         p=[0.30, 0.30, 0.20, 0.10, 0.07, 0.03])
        
        # Total existing debt
        existing_debt = np.where(existing_loans > 0,
                                monthly_income * np.random.uniform(0.5, 5, self.n_samples),
                                0)
        
        # Credit score (300-850)
        base_credit_score = np.random.normal(680, 80, self.n_samples).clip(300, 850)
        
        # Number of credit inquiries in last 6 months
        credit_inquiries = np.random.poisson(2, self.n_samples).clip(0, 15)
        
        # Payment history - percentage of on-time payments
        payment_history = np.random.beta(8, 2, self.n_samples) * 100
        
        # Number of late payments in last 2 years
        late_payments = np.random.poisson(1.5, self.n_samples).clip(0, 20)
        
        # Bankruptcy history
        bankruptcy = np.random.choice([0, 1], size=self.n_samples, p=[0.95, 0.05])
        
        # Home ownership
        home_ownership = np.random.choice(['Rent', 'Own', 'Mortgage', 'Other'],
                                         size=self.n_samples,
                                         p=[0.35, 0.25, 0.35, 0.05])
        
        # Employment length in years
        employment_length = np.random.gamma(2, 3, self.n_samples).clip(0, 40)
        
        # Loan purpose
        loan_purpose = np.random.choice(
            ['Business', 'Education', 'Home', 'Medical', 'Personal', 'Debt Consolidation', 'Auto'],
            size=self.n_samples,
            p=[0.15, 0.10, 0.20, 0.10, 0.20, 0.15, 0.10]
        )
        
        # CREATE DATAFRAME BASED ON THE DATA
        df = pd.DataFrame({
            'application_id': [f'APP{str(i).zfill(6)}' for i in range(self.n_samples)],
            'age': age.astype(int),
            'education': education_levels,
            'employment_type': employment_types,
            'marital_status': marital_status,
            'dependents': dependents,
            'monthly_income': monthly_income.round(2),
            'loan_amount': loan_amount.round(2),
            'loan_term': loan_term,
            'interest_rate': interest_rate.round(2),
            'loan_purpose': loan_purpose,
            'credit_score': base_credit_score.astype(int),
            'credit_history_length': credit_history_length.round(1),
            'existing_loans': existing_loans,
            'existing_debt': existing_debt.round(2),
            'credit_inquiries': credit_inquiries,
            'payment_history_pct': payment_history.round(2),
            'late_payments': late_payments,
            'bankruptcy_history': bankruptcy,
            'home_ownership': home_ownership,
            'employment_length': employment_length.round(1)
        })
        
        # Generate DEFAULT target with realistic imbalanced ratio (8-10% default rate)
        df['default'] = self._generate_default_target(df)
        
        # Add application timestamp
        start_date = datetime(2020, 1, 1)
        df['application_date'] = [
            start_date + timedelta(days=random.randint(0, 1460))
            for _ in range(self.n_samples)
        ]
        
        return df
    
    def _generate_default_target(self, df):
        # Calculate risk score based on multiple factors
        risk_score = np.zeros(len(df))
        
        # Credit score impact (higher score = lower risk)
        risk_score += (850 - df['credit_score']) / 100
        
        # Debt-to-income ratio impact
        dti_ratio = (df['existing_debt'] + (df['loan_amount'] / df['loan_term'])) / df['monthly_income']
        risk_score += dti_ratio * 3
        
        # Late payments impact
        risk_score += df['late_payments'] * 0.5
        
        # Bankruptcy impact
        risk_score += df['bankruptcy_history'] * 5
        
        # Credit inquiries impact
        risk_score += df['credit_inquiries'] * 0.3
        
        # Payment history impact (inverse)
        risk_score += (100 - df['payment_history_pct']) / 10
        
        # Employment stability
        risk_score += np.where(df['employment_type'] == 'Unemployed', 3, 0)
        risk_score += np.where(df['employment_length'] < 1, 2, 0)
        
        # High loan amount relative to income
        loan_to_income = df['loan_amount'] / (df['monthly_income'] * 12)
        risk_score += np.where(loan_to_income > 5, 2, 0)
        
        # Normalize risk score
        risk_score = (risk_score - risk_score.min()) / (risk_score.max() - risk_score.min())
        
        # Generate default with probability based on risk score
        # Adjusted to create ~8-10% default rate
        default_probability = risk_score ** 2 * 0.3  # Non-linear transformation
        
        default = np.random.binomial(1, default_probability)
        
        return default
    
    def save_data(self, df, filepath):
        # Create directory if it doesn't exist
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(filepath, index=False)
        print(f"\n Data saved to {filepath}")
        print(f"   Total samples: {len(df)}")
        print(f"   Default rate: {df['default'].mean()*100:.2f}%")
        print(f"   Default cases: {df['default'].sum()}")
        print(f"   Non-default cases: {(df['default']==0).sum()}")


if __name__ == "__main__":
    # Get root directory -> FINDS 'data'
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / 'data'
    
    # Create data directory if it doesn't EXIST !!
    data_dir.mkdir(exist_ok=True)
    
    # Training data (15,000 samples)
    print("Generating training data...")
    generator = CreditDataGenerator(n_samples=15000)
    train_df = generator.generate_data()
    generator.save_data(train_df, str(data_dir / 'credit_train.csv'))
    
    # Test data (5,000 samples)
    print("\nGenerating test data...")
    generator_test = CreditDataGenerator(n_samples=5000)
    test_df = generator_test.generate_data()
    generator_test.save_data(test_df, str(data_dir / 'credit_test.csv'))
    
    print("\n  Data Generation Complete!")
    print(f"   Files saved in: {data_dir}")