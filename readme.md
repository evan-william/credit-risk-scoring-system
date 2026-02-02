# Intelligent Credit Risk Scoring System

A production-ready machine learning system for predicting credit default risk using advanced feature engineering and gradient boosting algorithms.

---

## Overview

This project implements an enterprise-grade credit risk assessment system that addresses real-world challenges in consumer lending. The system processes credit applications, evaluates default probability, and provides actionable risk assessments through an interactive web interface.

**Key Characteristics:**
- Handles severely imbalanced datasets (approximately 5% default rate)
- Implements domain-informed feature engineering
- Focuses on business-relevant metrics over traditional accuracy
- Provides explainable predictions for regulatory compliance

---

## Architecture

The system follows a modular pipeline architecture:

**Data Layer**
- Synthetic data generation with realistic correlations
- 20,000 credit applications with 20+ base features
- Controlled imbalance ratio matching industry standards

**Feature Engineering**
- 60+ derived features including financial ratios and risk indicators
- Domain-specific transformations (DTI, LTI, credit utilization)
- Polynomial and interaction terms
- Standardized preprocessing pipeline

**Model Layer**
- Gradient Boosting Classifier optimized for imbalanced data
- Sample weighting to address class imbalance
- Threshold optimization for business objectives
- Cross-validation and hyperparameter tuning

**Application Layer**
- Streamlit-based web interface
- Single and batch prediction modes
- Real-time risk assessment
- Model performance analytics

---

## Technical Implementation

### Data Generation

The system generates realistic synthetic credit data with intentional correlations mirroring real-world patterns:

```python
# Risk calculation based on multiple factors
risk_score = (
    (850 - credit_score) / 100 +
    dti_ratio * 3 +
    late_payments * 0.5 +
    bankruptcy_flag * 5
)
```

**Dataset Characteristics:**
- 15,000 training samples
- 5,000 test samples
- 5-8% default rate (minority class)
- Features span demographics, financials, and credit history

### Feature Engineering

Comprehensive feature creation based on domain knowledge:

**Financial Ratios**
- Debt-to-Income (DTI): `(existing_debt + monthly_payment) / monthly_income`
- Loan-to-Income (LTI): `loan_amount / (monthly_income * 12)`
- Credit Utilization: `debt / estimated_credit_limit`
- Available Income: `income - debts - payment`

**Risk Indicators**
- Payment risk score (combination of late payments and delinquencies)
- Inquiry risk score (recent credit inquiries)
- Stability score (employment length + credit history)
- Loan burden assessment

**Categorical Features**
- Credit score tiers (Poor/Fair/Good/Excellent)
- Age groups
- Debt burden categories

**Advanced Features**
- Interaction terms (credit_score × DTI, income × stability)
- Polynomial features (DTI², credit_score²)

### Model Training

**Algorithm:** Gradient Boosting Classifier

**Hyperparameters:**
```python
{
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_samples_split': 5,
    'subsample': 0.8,
    'max_features': 'sqrt'
}
```

**Imbalanced Data Strategy:**
- Sample weighting with scale_pos_weight = 16.57
- Threshold tuning optimized for F1 score
- Focus on recall for minority class (defaults)

**Performance Metrics:**
- ROC-AUC: 0.69
- Overall Accuracy: 70%
- Default Detection Rate: 57%
- Approved Loan Quality: 97%

### Web Application

Built with Streamlit for accessibility and ease of deployment:

**Features:**
- Interactive form for single application assessment
- CSV upload for batch processing
- Visual risk scoring with gauge charts
- Detailed risk factor breakdown
- Model performance dashboard
- Exportable predictions

---

## Performance Analysis

### Confusion Matrix

|                | Predicted: No Default | Predicted: Default |
|----------------|----------------------|-------------------|
| Actual: No Default | 3,330 (TN) | 1,408 (FP) |
| Actual: Default | 113 (FN) | 149 (TP) |

### Business Metrics

**Default Detection Rate: 56.87%**  
The model successfully identifies over half of potential defaults before they occur, enabling proactive risk management.

**Approved Loan Quality: 96.72%**  
Among approved applications, 97 out of 100 are expected to perform as agreed, maintaining a healthy loan portfolio.

**Rejection Rate: 31.14%**  
A balanced approach that manages risk while maximizing lending volume.

**Key Insight:** The model is intentionally conservative (higher false positive rate) to minimize financial losses from false negatives. This trade-off aligns with typical risk management priorities in lending.

---

## Installation and Usage

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the web application
cd streamlit_app
streamlit run app.py
```

The pre-trained model is included in the repository and ready to use.

### Retraining the Model

```bash
cd src
python train_pipeline.py
```

This will regenerate data, engineer features, train the model, and update all saved artifacts.

---

## Project Structure

```
credit_risk_system/
│
├── data/
│   ├── credit_train.csv              # Training dataset
│   ├── credit_test.csv               # Test dataset
│   └── *_processed.csv               # Engineered features
│
├── models/
│   ├── credit_risk_model.pkl         # Serialized model
│   ├── feature_engine.pkl            # Feature pipeline
│   └── model_report.txt              # Performance metrics
│
├── src/
│   ├── data_generator.py             # Synthetic data creation
│   ├── feature_engineering.py        # Feature engineering logic
│   ├── model_training.py             # Training and evaluation
│   └── train_pipeline.py             # End-to-end orchestration
│
├── streamlit_app/
│   └── app.py                        # Web application
│
├── docs/
│   └── USER_GUIDE.md                 # Detailed documentation
│
├── README.md
├── QUICKSTART.md
└── requirements.txt
```

---

## Technology Stack

**Core Technologies:**
- Python 3.8+
- Scikit-learn (machine learning)
- Pandas, NumPy (data processing)
- Streamlit (web framework)

**Visualization:**
- Matplotlib, Seaborn (static plots)
- Plotly (interactive charts)

**Utilities:**
- Joblib (model serialization)
- StandardScaler, LabelEncoder (preprocessing)

---

## Use Cases

### Financial Institutions
- Automated credit underwriting to reduce manual review time
- Quantitative risk assessment for loan approval decisions
- Portfolio-level risk analysis and stress testing
- Compliance documentation with explainable predictions

### Fintech Applications
- Real-time credit decision APIs
- High-volume application processing
- A/B testing different risk models
- Integration with existing loan origination systems

### Educational & Research
- Demonstration of end-to-end ML pipeline
- Study of imbalanced classification techniques
- Feature engineering in financial domain
- Production deployment practices

---

## Limitations and Considerations

**Data Limitations:**
- Trained on synthetic data; real-world performance will vary
- Does not account for macroeconomic factors
- Limited to snapshot data (no time-series analysis)

**Model Constraints:**
- Assumes feature distributions remain stable over time
- May exhibit bias if training data is not representative
- Threshold is optimized for current dataset balance

**Deployment Notes:**
- Requires model monitoring for drift detection
- Should be validated against historical data before production use
- Regulatory compliance review recommended for actual deployment

---

## Future Development

**Technical Roadmap:**
- REST API implementation (Flask/FastAPI)
- Model monitoring and retraining pipeline
- Docker containerization
- Database integration (PostgreSQL)
- Automated testing suite

**Feature Enhancements:**
- Additional ML algorithms (XGBoost, LightGBM, Neural Networks)
- Ensemble methods for improved accuracy
- SHAP/LIME for enhanced interpretability
- Fairness and bias auditing tools
- Time-series features for longitudinal analysis

**Application Features:**
- User authentication and role-based access
- Audit logging for compliance
- PDF report generation
- Email notification system
- Historical trend analysis dashboard

---

## Technical Highlights

This project demonstrates several important machine learning engineering practices:

**Handling Class Imbalance**  
Rather than simply using SMOTE or other resampling techniques, the project implements a combination of sample weighting and threshold optimization. This approach preserves the true data distribution while achieving better minority class recall.

**Business-Focused Metrics**  
The evaluation framework prioritizes business-relevant metrics (default detection rate, approved loan quality) over traditional accuracy. This reflects real-world decision-making where different error types have different costs.

**Feature Engineering Pipeline**  
The feature engineering is implemented as a reusable pipeline that can be applied consistently to training data, test data, and new predictions. This ensures no data leakage and simplifies deployment.

**Explainability**  
Beyond feature importance, the application provides per-prediction risk breakdowns, making the model's decisions interpretable for end users and compliance officers.

---

## Contributing

This is a portfolio project and is not actively maintained for external contributions. However, feel free to fork the repository for your own use or learning purposes.

---

## License

This project is available for educational and portfolio purposes. If you use this code as a reference or starting point, attribution is appreciated but not required.

---

## Author

**Evan William**

This project was developed as a demonstration of production-ready machine learning engineering, from data generation through deployment.

---

**Last Updated:** February 2024  
**Version:** 1.0.0
