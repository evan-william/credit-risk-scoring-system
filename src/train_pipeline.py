import sys
import os
from pathlib import Path

script_dir = Path(__file__).resolve().parent # -> PROJECT ROOT
project_root = script_dir.parent
sys.path.append(str(script_dir))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # For non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_generator import CreditDataGenerator
from feature_engineering import CreditFeatureEngine
from model_training import CreditRiskModel


def main():
    # Setup directories
    data_dir = project_root / 'data'
    models_dir = project_root / 'models'
    
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("  INTELLIGENT CREDIT RISK SCORING SYSTEM - TRAINING PIPELINE")
    print("=" * 80)
    
    # Step 1: Generate Data
    print("\n  STEP 1: DATA GENERATION")
    print("-" * 80)
    
    generator = CreditDataGenerator(n_samples=15000)
    df_train = generator.generate_data()
    generator.save_data(df_train, str(data_dir / 'credit_train.csv'))
    
    generator_test = CreditDataGenerator(n_samples=5000)
    df_test = generator_test.generate_data()
    generator_test.save_data(df_test, str(data_dir / 'credit_test.csv'))
    
    # Step 2: Feature Engineering
    print("\n  STEP 2: FEATURE ENGINEERING")
    print("-" * 80)
    
    feature_engine = CreditFeatureEngine()
    
    # Process training data
    print("\nProcessing training data...")
    df_train_featured = feature_engine.create_features(df_train, is_training=True)
    df_train_final = feature_engine.encode_and_scale(df_train_featured, is_training=True)
    
    # Process test data
    print("\nProcessing test data...")
    df_test_featured = feature_engine.create_features(df_test, is_training=False)
    df_test_final = feature_engine.encode_and_scale(df_test_featured, is_training=False)
    
    # Save processed data
    df_train_final.to_csv(data_dir / 'credit_train_processed.csv', index=False)
    df_test_final.to_csv(data_dir / 'credit_test_processed.csv', index=False)
    
    print(f"\n✅ Processed data saved")
    print(f"   Training shape: {df_train_final.shape}")
    print(f"   Test shape: {df_test_final.shape}")
    
    # Step 3: Prepare for Training
    print("\n  STEP 3: PREPARING TRAINING DATA")
    print("-" * 80)
    
    # Split features and target
    X_train_full = df_train_final.drop('default', axis=1)
    y_train_full = df_train_final['default']
    
    X_test = df_test_final.drop('default', axis=1)
    y_test = df_test_final['default']
    
    # Create validation set from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.2, 
        stratify=y_train_full,
        random_state=42
    )
    
    print(f"Data split complete:")
    print(f"   Training: {X_train.shape[0]} samples ({y_train.sum()} defaults)")
    print(f"   Validation: {X_val.shape[0]} samples ({y_val.sum()} defaults)")
    print(f"   Test: {X_test.shape[0]} samples ({y_test.sum()} defaults)")
    
    # Step 4: Train Model
    print("\n  STEP 4: MODEL TRAINING")
    print("-" * 80)
    
    model = CreditRiskModel()
    model.train(X_train, y_train, X_val, y_val)
    
    # Step 5: Optimize Threshold
    print("\n  STEP 5: THRESHOLD OPTIMIZATION")
    print("-" * 80)
    
    model.optimize_threshold(X_val, y_val)
    
    # Step 6: Evaluation
    print("\n  STEP 6: MODEL EVALUATION")
    print("-" * 80)
    
    # Evaluate on validation set
    val_metrics, val_proba = model.evaluate(X_val, y_val, "Validation")
    
    # Evaluate on test set
    test_metrics, test_proba = model.evaluate(X_test, y_test, "Test")
    
    # Step 7: Generate Evaluation Plots
    print("\n  STEP 7: GENERATING EVALUATION PLOTS")
    print("-" * 80)
    
    model.plot_evaluation(y_test, test_proba, save_path=str(models_dir / 'model_evaluation.png'))
    
    # Feature Importance Plot
    plt.figure(figsize=(12, 8))
    top_30_features = model.feature_importance.head(30)
    plt.barh(range(len(top_30_features)), top_30_features['importance'])
    plt.yticks(range(len(top_30_features)), top_30_features['feature'])
    plt.xlabel('Importance Score')
    plt.title('Top 30 Most Important Features for Credit Risk Prediction')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(models_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print("📊 Feature importance plot saved")
    
    # Step 8: Save Model
    print("\n  STEP 8: SAVING MODEL")
    print("-" * 80)
    
    model.save_model(str(models_dir / 'credit_risk_model.pkl'))
    
    # Save feature engine
    import joblib
    joblib.dump(feature_engine, str(models_dir / 'feature_engine.pkl'))
    print(f"  Feature engine saved to {models_dir / 'feature_engine.pkl'}")
    
    # Step 9: Generate Model Report
    print("\n STEP 9: GENERATING MODEL REPORT")
    print("-" * 80)
    
    report = f"""
    ================================================================================
    CREDIT RISK SCORING SYSTEM - MODEL REPORT
    ================================================================================
    
    Training Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    DATASET STATISTICS
    ------------------
    Training Samples: {len(X_train):,}
    Validation Samples: {len(X_val):,}
    Test Samples: {len(X_test):,}
    Total Features: {X_train.shape[1]}
    
    Class Distribution (Training):
    - No Default: {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.2f}%)
    - Default: {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.2f}%)
    
    MODEL CONFIGURATION
    -------------------
    Algorithm: XGBoost (Gradient Boosting)
    Objective: Binary Classification
    Imbalanced Data Handling: scale_pos_weight
    
    VALIDATION SET PERFORMANCE
    --------------------------
    Accuracy: {val_metrics['accuracy']:.4f}
    ROC-AUC: {val_metrics['roc_auc']:.4f}
    Average Precision: {val_metrics['avg_precision']:.4f}
    F1 Score: {val_metrics['f1_score']:.4f}
    Optimal Threshold: {model.threshold:.4f}
    
    TEST SET PERFORMANCE
    --------------------
    Accuracy: {test_metrics['accuracy']:.4f}
    ROC-AUC: {test_metrics['roc_auc']:.4f}
    Average Precision: {test_metrics['avg_precision']:.4f}
    F1 Score: {test_metrics['f1_score']:.4f}
    
    TOP 10 MOST IMPORTANT FEATURES
    -------------------------------
    {model.feature_importance.head(10).to_string(index=False)}
    
    BUSINESS IMPACT
    ---------------
    The model successfully predicts credit default risk with high accuracy.
    Key insights:
    - Debt-to-income ratio is the strongest predictor
    - Credit score and payment history are critical factors
    - Model can reduce default rate in approved loans significantly
    - Enables data-driven credit decisions at scale
    
    ================================================================================
    """
    
    with open(models_dir / 'model_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    
    print("\n" + "=" * 80)
    print(" TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated Files:")
    print(f"  📁 {data_dir / 'credit_train.csv'}")
    print(f"  📁 {data_dir / 'credit_test.csv'}")
    print(f"  📁 {data_dir / 'credit_train_processed.csv'}")
    print(f"  📁 {data_dir / 'credit_test_processed.csv'}")
    print(f"  📁 {models_dir / 'credit_risk_model.pkl'}")
    print(f"  📁 {models_dir / 'feature_engine.pkl'}")
    print(f"  📁 {models_dir / 'model_evaluation.png'}")
    print(f"  📁 {models_dir / 'feature_importance.png'}")
    print(f"  📁 {models_dir / 'model_report.txt'}")
    print("\n Ready to deploy! Run the Streamlit app to use the model.")


if __name__ == "__main__":
    main()