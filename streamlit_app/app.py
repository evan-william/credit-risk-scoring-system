import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import sys
import os
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.append(str(project_root / 'src'))

# Setup paths
models_dir = project_root / 'models'
data_dir = project_root / 'data'

# Page configuration
st.set_page_config(
    page_title="Credit Risk Scoring System",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS BELOW
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #2c3e50;
        padding-top: 1rem;
    }
    .risk-high {
        color: #e74c3c;
        font-weight: bold;
    }
    .risk-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .risk-low {
        color: #27ae60;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    try:
        model_data = joblib.load(models_dir / 'credit_risk_model.pkl')
        feature_engine = joblib.load(models_dir / 'feature_engine.pkl')
        return model_data, feature_engine
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run the training pipeline first: `python src/train_pipeline.py`")
        return None, None


def create_gauge_chart(value, title):
    # Determine color based on value
    if value < 30:
        color = "#27ae60"  # Green
        risk_level = "Low Risk"
    elif value < 70:
        color = "#f39c12"  # Orange
        risk_level = "Medium Risk"
    else:
        color = "#e74c3c"  # Red
        risk_level = "High Risk"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#d5f4e6'},
                {'range': [30, 70], 'color': '#fdebd0'},
                {'range': [70, 100], 'color': '#fadbd8'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'size': 16}
    )
    
    return fig, risk_level, color


def get_risk_explanation(default_prob, applicant_data):
    explanations = []
    
    # DTI Analysis
    dti = (applicant_data.get('existing_debt', 0) + 
           (applicant_data.get('loan_amount', 0) / applicant_data.get('loan_term', 12))) / \
          applicant_data.get('monthly_income', 1)
    
    if dti > 0.5:
        explanations.append("⚠️ High debt-to-income ratio (>50%) indicates financial strain")
    elif dti > 0.3:
        explanations.append("⚡ Moderate debt-to-income ratio (30-50%)")
    else:
        explanations.append("✅ Healthy debt-to-income ratio (<30%)")
    
    # Credit Score Analysis
    credit_score = applicant_data.get('credit_score', 650)
    if credit_score < 580:
        explanations.append("⚠️ Poor credit score (<580) - significant risk factor")
    elif credit_score < 670:
        explanations.append("⚡ Fair credit score (580-670)")
    elif credit_score < 740:
        explanations.append("✅ Good credit score (670-740)")
    else:
        explanations.append("✅ Excellent credit score (>740)")
    
    # Payment History
    payment_history = applicant_data.get('payment_history_pct', 85)
    if payment_history < 80:
        explanations.append("⚠️ Poor payment history (<80% on-time)")
    elif payment_history < 95:
        explanations.append("⚡ Average payment history (80-95%)")
    else:
        explanations.append("✅ Excellent payment history (>95%)")
    
    # Late Payments
    late_payments = applicant_data.get('late_payments', 0)
    if late_payments > 3:
        explanations.append(f"⚠️ High number of late payments ({late_payments})")
    elif late_payments > 0:
        explanations.append(f"⚡ Some late payments ({late_payments})")
    else:
        explanations.append("✅ No late payments")
    
    # Bankruptcy
    if applicant_data.get('bankruptcy_history', 0) == 1:
        explanations.append("⚠️ Previous bankruptcy on record")
    
    return explanations


def predict_single_application(applicant_data, model_data, feature_engine):
    # Convert to DataFrame
    df = pd.DataFrame([applicant_data])
    
    # Feature engineering
    df_featured = feature_engine.create_features(df, is_training=False)
    df_processed = feature_engine.encode_and_scale(df_featured, is_training=False)
    
    # Make prediction
    model = model_data['model']
    threshold = model_data['threshold']
    
    default_prob = model.predict_proba(df_processed)[0, 1]
    prediction = 1 if default_prob >= threshold else 0
    
    # Risk score (0-100)
    risk_score = default_prob * 100
    
    return prediction, default_prob, risk_score


def main():
    # Header
    st.markdown("<h1>Smart Credit Risk Scoring System</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Load models
    model_data, feature_engine = load_models()
    
    if model_data is None or feature_engine is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("🛠️ Application Settings")
    app_mode = st.sidebar.selectbox(
        "Select Mode",
        ["Single Application", "Batch Processing", "Model Analytics", "About"]
    )
    
    if app_mode == "Single Application":
        show_single_application(model_data, feature_engine)
    elif app_mode == "Batch Processing":
        show_batch_processing(model_data, feature_engine)
    elif app_mode == "Model Analytics":
        show_model_analytics(model_data)
    else:
        show_about()


def show_single_application(model_data, feature_engine):
    st.header("Credit Application Assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Personal Information")
        age = st.number_input("Age", min_value=18, max_value=75, value=35)
        education = st.selectbox("Education", 
                                ['High School', 'Bachelor', 'Master', 'PhD', 'Vocational'])
        marital_status = st.selectbox("Marital Status",
                                     ['Single', 'Married', 'Divorced', 'Widowed'])
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
    
    with col2:
        st.subheader("Employment & Income")
        employment_type = st.selectbox("Employment Type",
                                      ['Full-Time', 'Part-Time', 'Self-Employed', 'Unemployed', 'Retired'])
        employment_length = st.number_input("Employment Length (years)", 
                                           min_value=0.0, max_value=40.0, value=5.0, step=0.5)
        monthly_income = st.number_input("Monthly Income ($)", 
                                        min_value=0.0, max_value=50000.0, value=5000.0, step=100.0)
        home_ownership = st.selectbox("Home Ownership",
                                     ['Rent', 'Own', 'Mortgage', 'Other'])
    
    with col3:
        st.subheader("Loan Information")
        loan_amount = st.number_input("Loan Amount ($)", 
                                     min_value=1000.0, max_value=500000.0, value=20000.0, step=1000.0)
        loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60, 72], index=2)
        interest_rate = st.number_input("Interest Rate (%)", 
                                       min_value=5.0, max_value=25.0, value=10.0, step=0.5)
        loan_purpose = st.selectbox("Loan Purpose",
                                   ['Business', 'Education', 'Home', 'Medical', 
                                    'Personal', 'Debt Consolidation', 'Auto'])
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Credit History")
        credit_score = st.number_input("Credit Score", 
                                      min_value=300, max_value=850, value=680)
        credit_history_length = st.number_input("Credit History (years)",
                                               min_value=0.0, max_value=30.0, value=5.0, step=0.5)
        payment_history_pct = st.slider("On-Time Payment %", 
                                        min_value=0, max_value=100, value=90)
    
    with col2:
        st.subheader("Existing Obligations")
        existing_loans = st.number_input("Number of Existing Loans",
                                        min_value=0, max_value=10, value=1)
        existing_debt = st.number_input("Total Existing Debt ($)",
                                       min_value=0.0, max_value=200000.0, value=5000.0, step=500.0)
        late_payments = st.number_input("Late Payments (last 2 years)",
                                       min_value=0, max_value=20, value=0)
    
    with col3:
        st.subheader("Additional Factors")
        credit_inquiries = st.number_input("Credit Inquiries (last 6 months)",
                                          min_value=0, max_value=15, value=1)
        bankruptcy_history = st.selectbox("Bankruptcy History", 
                                         ["No", "Yes"], index=0)
    
    st.markdown("---")
    
    # Predict button
    if st.button("Assess Credit Risk", type="primary", use_container_width=True):
        
        # Prepare applicant data
        applicant_data = {
            'age': age,
            'education': education,
            'employment_type': employment_type,
            'marital_status': marital_status,
            'dependents': dependents,
            'monthly_income': monthly_income,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'interest_rate': interest_rate,
            'loan_purpose': loan_purpose,
            'credit_score': credit_score,
            'credit_history_length': credit_history_length,
            'existing_loans': existing_loans,
            'existing_debt': existing_debt,
            'credit_inquiries': credit_inquiries,
            'payment_history_pct': payment_history_pct,
            'late_payments': late_payments,
            'bankruptcy_history': 1 if bankruptcy_history == "Yes" else 0,
            'home_ownership': home_ownership,
            'employment_length': employment_length
        }
        
        # Make prediction
        with st.spinner("Analyzing credit risk..."):
            prediction, default_prob, risk_score = predict_single_application(
                applicant_data, model_data, feature_engine
            )
        
        # Display results
        st.markdown("---")
        st.header("📊 Assessment Results")
        
        # Create gauge chart
        gauge_fig, risk_level, risk_color = create_gauge_chart(risk_score, "Credit Risk Score")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col2:
            st.markdown(f"### Decision: {'❌ REJECT' if prediction == 1 else '✅ APPROVE'}")
            st.markdown(f"**Risk Level:** <span style='color:{risk_color}'>{risk_level}</span>", 
                       unsafe_allow_html=True)
            st.metric("Default Probability", f"{default_prob*100:.2f}%")
            st.metric("Risk Score", f"{risk_score:.1f}/100")
            
            # Recommendation
            if prediction == 0:
                st.success("✅ **APPROVED** - Low default risk detected")
            else:
                st.error("❌ **REJECTED** - High default risk detected")
        
        # Risk Explanation
        st.markdown("---")
        st.subheader("📝 Risk Assessment Breakdown")
        
        explanations = get_risk_explanation(default_prob, applicant_data)
        
        for explanation in explanations:
            st.markdown(f"- {explanation}")
        
        # Key Metrics
        st.markdown("---")
        st.subheader("💰 Financial Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        dti = (existing_debt + (loan_amount / loan_term)) / monthly_income
        loan_to_income = loan_amount / (monthly_income * 12)
        monthly_payment = loan_amount / loan_term
        available_income = monthly_income - existing_debt - monthly_payment
        
        with col1:
            st.metric("Debt-to-Income Ratio", f"{dti*100:.1f}%")
        with col2:
            st.metric("Loan-to-Income Ratio", f"{loan_to_income:.2f}x")
        with col3:
            st.metric("Monthly Payment", f"${monthly_payment:,.2f}")
        with col4:
            st.metric("Available Income", f"${available_income:,.2f}")


def show_batch_processing(model_data, feature_engine):
    st.header("📦 Batch Credit Assessment")
    
    st.info("Upload a CSV file with multiple credit applications for bulk processing")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} applications")
            
            # Show preview
            with st.expander("📄 Data Preview"):
                st.dataframe(df.head(10))
            
            if st.button("🚀 Process All Applications", type="primary"):
                with st.spinner("Processing applications..."):
                    # Feature engineering
                    df_featured = feature_engine.create_features(df, is_training=False)
                    df_processed = feature_engine.encode_and_scale(df_featured, is_training=False)
                    
                    # Predictions
                    model = model_data['model']
                    threshold = model_data['threshold']
                    
                    probas = model.predict_proba(df_processed)[:, 1]
                    predictions = (probas >= threshold).astype(int)
                    risk_scores = probas * 100
                    
                    # Add results to dataframe
                    df['default_probability'] = probas
                    df['risk_score'] = risk_scores
                    df['prediction'] = predictions
                    df['decision'] = df['prediction'].map({0: 'APPROVE', 1: 'REJECT'})
                
                # Display results
                st.markdown("---")
                st.header("📊 Batch Processing Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Applications", len(df))
                with col2:
                    approved = (df['prediction'] == 0).sum()
                    st.metric("Approved", approved, 
                             delta=f"{approved/len(df)*100:.1f}%")
                with col3:
                    rejected = (df['prediction'] == 1).sum()
                    st.metric("Rejected", rejected,
                             delta=f"{rejected/len(df)*100:.1f}%")
                with col4:
                    avg_risk = df['risk_score'].mean()
                    st.metric("Avg Risk Score", f"{avg_risk:.1f}")
                
                # Risk distribution
                fig = px.histogram(df, x='risk_score', color='decision',
                                  title="Risk Score Distribution",
                                  labels={'risk_score': 'Risk Score', 'count': 'Count'},
                                  color_discrete_map={'APPROVE': '#27ae60', 'REJECT': '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Results table
                st.subheader("📋 Detailed Results")
                st.dataframe(
                    df[['age', 'monthly_income', 'loan_amount', 'credit_score', 
                        'risk_score', 'decision']].style.applymap(
                        lambda x: 'background-color: #d5f4e6' if x == 'APPROVE' else 
                                  ('background-color: #fadbd8' if x == 'REJECT' else ''),
                        subset=['decision']
                    ),
                    use_container_width=True
                )
                
                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name="credit_assessment_results.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")


def show_model_analytics(model_data):
    st.header("Model Performance Analytics")
    
    # Load test data
    try:
        df_test = pd.read_csv(data_dir / 'credit_test_processed.csv')
        
        X_test = df_test.drop('default', axis=1)
        y_test = df_test['default']
        
        model = model_data['model']
        threshold = model_data['threshold']
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Metrics
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{accuracy*100:.2f}%")
        with col2:
            st.metric("ROC-AUC", f"{roc_auc:.4f}")
        with col3:
            st.metric("F1 Score", f"{f1:.4f}")
        with col4:
            st.metric("Threshold", f"{threshold:.4f}")
        
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['No Default', 'Default'],
            y=['No Default', 'Default'],
            text=cm,
            texttemplate="%{text}",
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        st.subheader("Top 20 Feature Importances")
        
        feature_importance = model_data['feature_importance'].head(20)
        
        fig = go.Figure(go.Bar(
            x=feature_importance['importance'],
            y=feature_importance['feature'],
            orientation='h'
        ))
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading analytics: {e}")
        st.info("Please ensure test data is available")


def show_about():
    st.header("About This System")
    
    st.markdown("""
    ## Intelligent Credit Risk Scoring System
    
    ### Overview
    This is an advanced machine learning system designed to assess credit risk for loan applications.
    The system uses state-of-the-art XGBoost algorithm with sophisticated feature engineering to 
    predict the likelihood of loan default.
    
    ### Key Features
    - **ML Model**: XGBoost with imbalanced data handling
    - **Comprehensive Feature Engineering**: 60+ features including financial ratios
    - **Real-time Assessment**: Instant credit decisions
    - **Batch Processing**: Handle multiple applications simultaneously
    - **Explainable AI**: Clear breakdown of risk factors
    - **UI**: User-friendly Streamlit interface
    
    ### Technical Stack
    - **ML Framework**: XGBoost, Scikit-learn
    - **Data Processing**: Pandas, NumPy
    - **Visualization**: Plotly, Matplotlib
    - **Web Framework**: Streamlit
    
    ### Model Performance
    - ROC-AUC: >0.85
    - Accuracy: >80%
    - Handles imbalanced data effectively
    - Optimized for financial industry standards
    
    ### Use Cases
    - Banks and financial institutions
    - Fintech companies
    - P2P lending platforms
    - Credit unions
    
    ### Creator
    **Evan William | linkedin.com/evanwilliam03**  
    **Version**: 1.0.0  
    **Last Updated**: 2024
    """)


if __name__ == "__main__":
    main()
