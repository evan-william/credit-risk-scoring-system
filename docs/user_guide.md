# User Guide - Intelligent Credit Risk Scoring System

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Using the Web Application](#using-the-web-application)
4. [Understanding Results](#understanding-results)
5. [Advanced Usage](#advanced-usage)
6. [FAQ](#faq)

---

## Introduction

### What is the Credit Risk Scoring System?

This system is an AI-powered tool that helps assess the risk of loan default for credit applications. It uses machine learning to analyze various factors such as:

- Credit history and score
- Income and employment stability
- Debt levels and payment behavior
- Loan characteristics

### Who Should Use This?

- **Banks & Financial Institutions**: Automate credit decisions
- **Fintech Companies**: Quick loan approvals
- **Credit Analysts**: Support decision-making
- **Risk Managers**: Portfolio risk assessment

---

## Getting Started

### Installation

#### Option 1: Quick Setup (Recommended)

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup script
./setup.sh
```

The script will:
1. Check Python installation
2. Install all dependencies
3. Optionally create virtual environment
4. Train the model
5. Launch the web application

#### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Generate data and train model
cd src
python train_pipeline.py

# Launch web app
cd ../streamlit_app
streamlit run app.py
```

### First Launch

When you first launch the application:
1. Open your browser to `http://localhost:8501`
2. You'll see the main interface with 4 modes
3. Ensure the model is trained (check models/ folder)

---

## Using the Web Application

### Mode 1: Single Application Assessment

This mode allows you to assess individual credit applications.

#### Step-by-Step Guide:

**1. Personal Information**
- **Age**: Applicant's age (18-75)
- **Education**: Highest level of education
- **Marital Status**: Current marital status
- **Dependents**: Number of financial dependents

**2. Employment & Income**
- **Employment Type**: Current employment status
- **Employment Length**: Years in current job
- **Monthly Income**: Gross monthly income in dollars
- **Home Ownership**: Housing situation

**3. Loan Information**
- **Loan Amount**: Requested loan amount
- **Loan Term**: Repayment period in months
- **Interest Rate**: Proposed interest rate
- **Loan Purpose**: Reason for the loan

**4. Credit History**
- **Credit Score**: FICO score (300-850)
- **Credit History Length**: Years of credit history
- **Payment History**: % of on-time payments

**5. Existing Obligations**
- **Existing Loans**: Number of active loans
- **Existing Debt**: Total current debt
- **Late Payments**: Late payments in last 2 years

**6. Additional Factors**
- **Credit Inquiries**: Recent credit checks
- **Bankruptcy History**: Any past bankruptcies

#### Interpreting Results:

After clicking "Assess Credit Risk", you'll see:

**Risk Score Gauge (0-100)**
- **0-30 (Green)**: Low Risk - Likely to approve
- **30-70 (Orange)**: Medium Risk - Careful consideration
- **70-100 (Red)**: High Risk - Likely to reject

**Decision**
- ✅ **APPROVE**: Low default probability, safe to approve
- ❌ **REJECT**: High default probability, recommend rejection

**Risk Assessment Breakdown**
- Detailed explanation of contributing factors
- Specific metrics that influenced the decision

**Financial Metrics**
- **DTI Ratio**: Should be <50% for healthy finances
- **Loan-to-Income**: Lower is better
- **Monthly Payment**: Impact on cash flow
- **Available Income**: Remaining income after obligations

#### Example: Good Applicant

```
Age: 35
Credit Score: 720
Monthly Income: $6,000
Loan Amount: $20,000
DTI Ratio: 25%
Payment History: 98%
Late Payments: 0

Result: ✅ APPROVE
Risk Score: 18/100
Decision: Low Risk - Excellent candidate
```

#### Example: High-Risk Applicant

```
Age: 28
Credit Score: 580
Monthly Income: $3,000
Loan Amount: $30,000
DTI Ratio: 65%
Payment History: 75%
Late Payments: 5

Result: ❌ REJECT
Risk Score: 82/100
Decision: High Risk - Significant default probability
```

---

### Mode 2: Batch Processing

Process multiple applications at once using CSV files.

#### CSV File Format:

Your CSV should have these columns:

```
age,education,employment_type,marital_status,dependents,monthly_income,
loan_amount,loan_term,interest_rate,loan_purpose,credit_score,
credit_history_length,existing_loans,existing_debt,credit_inquiries,
payment_history_pct,late_payments,bankruptcy_history,home_ownership,
employment_length
```

#### Sample CSV Row:

```
35,Bachelor,Full-Time,Married,2,6000,20000,36,8.5,Personal,720,
5.0,1,3000,1,95,0,0,Mortgage,7.0
```

#### Processing Steps:

1. Click "Choose a CSV file"
2. Upload your CSV file
3. Preview the data to ensure correct format
4. Click "Process All Applications"
5. View results and download processed file

#### Results Include:

- Total applications processed
- Number approved/rejected
- Risk score distribution chart
- Detailed results table
- Downloadable CSV with decisions

---

### Mode 3: Model Analytics

View detailed model performance and insights.

#### Available Analytics:

**Performance Metrics**
- **Accuracy**: Overall prediction accuracy
- **ROC-AUC**: Model's ability to distinguish classes
- **F1 Score**: Balance of precision and recall
- **Threshold**: Decision boundary

**Confusion Matrix**
- True Positives (TP): Correctly predicted defaults
- True Negatives (TN): Correctly predicted non-defaults
- False Positives (FP): Incorrectly predicted defaults
- False Negatives (FN): Missed defaults

**Feature Importance**
- Shows which factors matter most
- Top 20 features ranked by importance
- Helps understand model decisions

---

### Mode 4: About

Information about the system, technical details, and documentation.

---

## Understanding Results

### Risk Score Interpretation

| Score Range | Risk Level | Meaning | Action |
|------------|-----------|---------|--------|
| 0-30 | Low | Very safe to approve | ✅ Approve |
| 30-50 | Low-Medium | Generally safe | ✅ Approve with monitoring |
| 50-70 | Medium-High | Requires careful review | ⚠️ Manual review |
| 70-85 | High | Significant risk | ❌ Likely reject |
| 85-100 | Very High | Extreme risk | ❌ Reject |

### Key Risk Factors

**1. Debt-to-Income (DTI) Ratio**
- **Formula**: (Total Monthly Debt) / (Monthly Income)
- **Healthy**: <30%
- **Acceptable**: 30-50%
- **Risky**: >50%

**2. Credit Score**
- **Poor**: 300-579
- **Fair**: 580-669
- **Good**: 670-739
- **Very Good**: 740-799
- **Excellent**: 800-850

**3. Payment History**
- **Excellent**: 95-100% on-time
- **Good**: 85-94% on-time
- **Fair**: 75-84% on-time
- **Poor**: <75% on-time

**4. Late Payments**
- **None**: 0 late payments
- **Few**: 1-2 late payments
- **Several**: 3-5 late payments
- **Many**: >5 late payments

---

## Advanced Usage

### Customizing Risk Tolerance

You can adjust the system's risk tolerance by modifying the threshold:

```python
# In src/model_training.py
# Conservative (fewer approvals, lower risk)
model.threshold = 0.6

# Balanced (default)
model.threshold = 0.5

# Aggressive (more approvals, higher risk)
model.threshold = 0.4
```

### Batch Processing Tips

**For Large Files:**
- Process in chunks of 1,000-5,000 applications
- Ensure CSV is properly formatted
- Remove any special characters from column names

**For Best Results:**
- Clean your data first
- Remove duplicate applications
- Ensure all required fields are present

### Integration with Existing Systems

**API Development** (Future Enhancement):
```python
# Example API endpoint
POST /api/v1/assess
{
  "applicant_data": {...},
  "return_explanation": true
}
```

**Database Integration**:
```python
# Save results to database
import sqlite3
predictions.to_sql('credit_decisions', conn)
```

---

## FAQ

### General Questions

**Q: How accurate is the model?**
A: The model achieves 80%+ accuracy with ROC-AUC >0.85. It's been optimized for financial industry standards.

**Q: Can I use this for real credit decisions?**
A: This is a demonstration system. For production use, ensure compliance with local regulations and conduct thorough testing.

**Q: How is my data handled?**
A: Data is processed locally and not stored unless you explicitly save results.

### Technical Questions

**Q: How do I retrain the model with new data?**
A: Place your data in CSV format in the data/ folder and run `python src/train_pipeline.py`

**Q: Can I add new features?**
A: Yes! Modify `src/feature_engineering.py` to add custom features.

**Q: What if the model files are missing?**
A: Run the training pipeline: `cd src && python train_pipeline.py`

**Q: How do I change the port for Streamlit?**
A: Run: `streamlit run app.py --server.port 8502`

### Performance Questions

**Q: How fast is the prediction?**
A: Single predictions: <100ms
   Batch (1000 apps): <5 seconds

**Q: Can it handle millions of applications?**
A: For production scale, consider implementing batch processing and caching.

**Q: Does it work offline?**
A: Yes! Once trained, the model works completely offline.

### Business Questions

**Q: What's the recommended rejection rate?**
A: Typically 10-15% for balanced risk-reward. Adjust based on your risk appetite.

**Q: How do I explain rejections to customers?**
A: Use the "Risk Assessment Breakdown" section which provides clear, specific reasons.

**Q: Can I override the model's decision?**
A: Yes! The system provides recommendations. Final decisions should involve human judgment.

---

## Support

### Getting Help

1. **Documentation**: Check README.md and this guide
2. **GitHub Issues**: Report bugs or request features
3. **Code Comments**: Most functions have detailed comments

### Best Practices

1. **Always review high-risk decisions manually**
2. **Monitor model performance over time**
3. **Update model quarterly with new data**
4. **Keep audit logs of all decisions**
5. **Ensure compliance with local regulations**

---

## Appendix

### Sample Test Cases

**Test Case 1: Ideal Applicant**
```
Age: 40, Income: $8000, Credit: 780, DTI: 20%
Expected: APPROVE (Risk: ~15)
```

**Test Case 2: Marginal Applicant**
```
Age: 30, Income: $4000, Credit: 650, DTI: 45%
Expected: BORDERLINE (Risk: ~55)
```

**Test Case 3: High Risk**
```
Age: 25, Income: $2500, Credit: 550, DTI: 70%
Expected: REJECT (Risk: ~85)
```

### Glossary

- **DTI**: Debt-to-Income ratio
- **ROC-AUC**: Area Under the Receiver Operating Characteristic curve
- **Default**: Failure to repay a loan
- **Feature**: Input variable used by the model
- **Threshold**: Decision boundary for classification

---

**Last Updated**: 2024
**Version**: 1.0.0

For more information, see the main [README.md](README.md)