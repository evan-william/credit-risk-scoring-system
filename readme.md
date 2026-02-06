# Intelligent Credit Risk Scoring System

A production-ready machine learning system for predicting credit default risk using advanced feature engineering and gradient boosting algorithms.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat)
![License](https://img.shields.io/badge/License-Educational-green?style=flat)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technical Implementation](#technical-implementation)
- [Performance Metrics](#performance-metrics)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Use Cases](#use-cases)
- [Limitations](#limitations)
- [Future Development](#future-development)

---

## Overview

This project implements an enterprise-grade credit risk assessment system that addresses real-world challenges in consumer lending. The system processes credit applications, evaluates default probability, and provides actionable risk assessments through an interactive web interface.

### Key Characteristics

- Handles severely imbalanced datasets (approximately 5% default rate)
- Implements domain-informed feature engineering with 60+ derived features
- Focuses on business-relevant metrics over traditional accuracy
- Provides explainable predictions for regulatory compliance
- Production-ready with modular architecture and web interface

---

## Key Features

**Advanced Feature Engineering**
- Financial ratios: DTI, LTI, credit utilization, available income
- Risk indicators: payment risk, inquiry risk, stability scores
- Interaction and polynomial features for complex relationships

**Robust Model Training**
- Gradient Boosting Classifier optimized for imbalanced data
- Sample weighting strategy (scale_pos_weight = 16.57)
- Threshold optimization for business objectives
- Cross-validation and hyperparameter tuning

**Interactive Web Application**
- Single application assessment with real-time predictions
- Batch processing via CSV upload
- Visual risk scoring with detailed breakdowns
- Model performance analytics dashboard
- Exportable predictions and reports

**Business-Focused Metrics**
- Default Detection Rate: 57%
- Approved Loan Quality: 97%
- ROC-AUC: 0.69
- Balanced approach between risk mitigation and lending volume

---

## Architecture

### System Pipeline

```
Data Generation → Feature Engineering → Model Training → Web Application
```

**1. Data Layer**
- Synthetic data generation with realistic correlations
- 20,000 credit applications (15K train, 5K test)
- 20+ base features spanning demographics, financials, and credit history
- Controlled 5-8% default rate matching industry standards

**2. Feature Engineering**
- 60+ derived features using domain knowledge
- Financial ratios and risk indicators
- Categorical encodings and binning
- Polynomial and interaction terms
- Standardized preprocessing pipeline

**3. Model Layer**
- Gradient Boosting Classifier with 300 estimators
- Optimized for imbalanced classification
- Sample weighting and threshold tuning
- Comprehensive evaluation framework

**4. Application Layer**
- Streamlit-based interactive interface
- Real-time and batch prediction modes
- Visual analytics and reporting
- Model performance monitoring

---

## Technical Implementation

### Data Generation

Synthetic credit data with intentional correlations mirroring real-world patterns:

```python
# Risk score calculation
risk_score = (
    (850 - credit_score) / 100 +
    dti_ratio * 3 +
    late_payments * 0.5 +
    bankruptcy_flag * 5
)
```

**Dataset Characteristics:**
- 20,000 total samples (15K train / 5K test)
- 5-8% default rate (minority class)
- Realistic correlations between features
- No data leakage between train and test sets

### Feature Engineering

**Financial Ratios**
```python
DTI = (existing_debt + monthly_payment) / monthly_income
LTI = loan_amount / (monthly_income * 12)
Credit_Utilization = debt / estimated_credit_limit
Available_Income = income - debts - payment
```

**Risk Indicators**
- Payment Risk Score: Combination of late payments and delinquencies
- Inquiry Risk Score: Recent credit inquiries weighted by recency
- Stability Score: Employment length + credit history age
- Loan Burden: Categorical assessment of debt load

**Categorical Features**
- Credit Score Tiers: Poor / Fair / Good / Excellent
- Age Groups: Young / Middle / Senior
- Debt Burden: Low / Medium / High / Very High

**Advanced Features**
- Interaction terms: credit_score × DTI, income × stability
- Polynomial features: DTI², credit_score²
- All features standardized using StandardScaler

### Model Configuration

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

**Imbalanced Data Handling:**
- Sample weighting: scale_pos_weight = 16.57
- Threshold optimization: Tuned for F1 score maximization
- Focus on recall for minority class detection

---

## Performance Metrics

### Confusion Matrix Results

|                    | Predicted: No Default | Predicted: Default |
|--------------------|----------------------|-------------------|
| **Actual: No Default** | 3,330 (TN)           | 1,408 (FP)        |
| **Actual: Default**    | 113 (FN)             | 149 (TP)          |

### Key Business Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | 0.69 | Good discrimination ability |
| **Overall Accuracy** | 70% | Balanced performance |
| **Default Detection Rate** | 57% | Identifies over half of defaults |
| **Approved Loan Quality** | 97% | High portfolio quality |
| **Rejection Rate** | 31% | Conservative but balanced |

### Business Impact

**Default Detection Rate (56.87%)**  
Successfully identifies over half of potential defaults before they occur, enabling proactive risk management and loss prevention.

**Approved Loan Quality (96.72%)**  
Among approved applications, 97 out of 100 are expected to perform as agreed, maintaining a healthy and profitable loan portfolio.

**Strategic Trade-off**  
The model is intentionally conservative with a higher false positive rate to minimize costly false negatives. This aligns with standard risk management priorities in consumer lending.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/[USERNAME]/credit-risk-system.git
cd credit-risk-system

# Install dependencies
pip install -r requirements.txt

# Launch web application
cd streamlit_app
streamlit run app.py
```

The pre-trained model is included and ready to use immediately.

### Retraining the Model

```bash
cd src
python train_pipeline.py
```

This regenerates data, engineers features, trains the model, and updates all artifacts.

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
│   ├── credit_risk_model.pkl         # Trained model
│   ├── feature_engine.pkl            # Feature pipeline
│   └── model_report.txt              # Performance metrics
│
├── src/
│   ├── data_generator.py             # Synthetic data creation
│   ├── feature_engineering.py        # Feature engineering logic
│   ├── model_training.py             # Training and evaluation
│   └── train_pipeline.py             # End-to-end pipeline
│
├── streamlit_app/
│   └── app.py                        # Web application
│
├── docs/
│   └── USER_GUIDE.md                 # Detailed documentation
│
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── LICENSE                           # License information
```

---

## Technology Stack

### Core Libraries

- **Python 3.8+**: Primary programming language
- **scikit-learn**: Machine learning framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Streamlit**: Web application framework

### Visualization

- **Matplotlib**: Static plotting
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive charts

### Utilities

- **Joblib**: Model serialization
- **StandardScaler**: Feature normalization
- **LabelEncoder**: Categorical encoding

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
- Integration with loan origination systems

### Educational & Research

- End-to-end ML pipeline demonstration
- Imbalanced classification techniques
- Feature engineering in financial domain
- Production deployment best practices

---

## Limitations

### Data Constraints

- Trained on synthetic data; real-world performance may vary
- Does not account for macroeconomic factors or market conditions
- Limited to snapshot data without time-series analysis
- May not capture all real-world feature relationships

### Model Constraints

- Assumes feature distributions remain stable over time
- May exhibit bias if training data is not representative
- Threshold optimized for current dataset balance
- Performance degrades with significant distribution shift

### Deployment Considerations

- Requires continuous monitoring for model drift
- Should be validated against historical data before production
- Regulatory compliance review recommended
- Periodic retraining necessary to maintain performance

---

## Future Development

### Technical Enhancements

- REST API implementation using Flask or FastAPI
- Automated model monitoring and retraining pipeline
- Docker containerization for easy deployment
- Database integration (PostgreSQL, MongoDB)
- Comprehensive automated testing suite
- CI/CD pipeline setup

### Model Improvements

- Alternative algorithms: XGBoost, LightGBM, Neural Networks
- Ensemble methods for improved accuracy
- SHAP and LIME for enhanced interpretability
- Fairness and bias auditing tools
- Time-series features for longitudinal analysis
- Calibrated probability estimates

### Application Features

- User authentication and role-based access control
- Comprehensive audit logging for compliance
- PDF report generation for decisions
- Email notification system
- Historical trend analysis dashboard
- Batch processing queue system

---

## Technical Highlights

### Handling Class Imbalance

Rather than simple resampling techniques like SMOTE, this project implements a combination of sample weighting and threshold optimization. This preserves the true data distribution while achieving better minority class recall.

### Business-Focused Evaluation

The evaluation framework prioritizes business-relevant metrics (default detection rate, approved loan quality) over traditional accuracy. This reflects real-world decision-making where different error types have different financial costs.

### Production-Ready Pipeline

The feature engineering is implemented as a reusable, serializable pipeline that ensures consistent processing across training, testing, and prediction. This design prevents data leakage and simplifies deployment.

### Explainable Predictions

Beyond global feature importance, the application provides per-prediction risk factor breakdowns, making model decisions interpretable for end users, loan officers, and compliance teams.

---

## Contributing

This is a portfolio project and is not actively maintained for external contributions. However, you are welcome to fork the repository for your own learning or use.

If you find this project helpful, attribution is appreciated but not required.

---

## License

This project is available for educational and portfolio purposes under the MIT License.

---

## Author

**Evan William**

Developed as a demonstration of production-ready machine learning engineering, covering the full lifecycle from data generation to deployment.

**Contact:**
- GitHub: [Your GitHub Profile]
- LinkedIn: [Your LinkedIn Profile]
- Email: [Your Email]

---

**Version:** 1.0.0  
**Last Updated:** February 2024
