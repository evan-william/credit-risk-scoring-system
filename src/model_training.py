import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, f1_score, accuracy_score,
    roc_curve, average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class CreditRiskModel:
    def __init__(self):
        self.model = None
        self.threshold = 0.5
        self.feature_importance = None
        self.training_history = {}
        
    def train(self, X_train, y_train, X_val, y_val):
        print("    Training Gradient Boosting Model...")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Default rate (train): {y_train.mean()*100:.2f}%")
        print(f"   Default rate (val): {y_val.mean()*100:.2f}%")
        
        # Calculate sample weights for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"   Scale pos weight: {scale_pos_weight:.2f}")
        
        # Create sample weights
        sample_weights = np.where(y_train == 1, scale_pos_weight, 1.0)
        
        # Gradient Boosting parameters optimized for credit risk
        params = {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_samples_split': 5,
            'min_samples_leaf': 3,
            'subsample': 0.8,
            'max_features': 'sqrt',
            'random_state': 42,
            'verbose': 1,
            'validation_fraction': 0.1,
            'n_iter_no_change': 20,
            'tol': 1e-4
        }
        
        # Initialize model
        self.model = GradientBoostingClassifier(**params)
        
        # Train with sample weights for imbalanced data
        print("\n   Training progress...")
        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate on validation set
        val_pred_proba = self.model.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_pred_proba)
        
        print(f"\n   Validation ROC-AUC: {val_auc:.4f}")
        print("\n    Training completed!")
        
        return self.model
    
    def optimize_threshold(self, X_val, y_val):
        print("\n  Optimizing decision threshold...")
        
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)
        
        # Calculate F1 scores for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        
        # Find threshold that maximizes F1
        optimal_idx = np.argmax(f1_scores)
        self.threshold = thresholds[optimal_idx]
        
        print(f"   Optimal threshold: {self.threshold:.4f}")
        print(f"   Expected F1 score: {f1_scores[optimal_idx]:.4f}")
        print(f"   Expected Precision: {precisions[optimal_idx]:.4f}")
        print(f"   Expected Recall: {recalls[optimal_idx]:.4f}")
        
        return self.threshold
    
    def evaluate(self, X_test, y_test, dataset_name="Test"):
        print(f"\n Evaluating on {dataset_name} Set...")
        
        # Predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n {dataset_name} Set Metrics:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        print(f"   Average Precision: {avg_precision:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n Confusion Matrix:")
        print(f"   TN: {cm[0,0]:,} | FP: {cm[0,1]:,}")
        print(f"   FN: {cm[1,0]:,} | TP: {cm[1,1]:,}")
        
        # Classification Report
        print(f"\n Classification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['No Default', 'Default']))
        
        # Business Metrics
        self._print_business_metrics(y_test, y_pred, y_pred_proba)
        
        metrics = {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'threshold': self.threshold
        }
        
        return metrics, y_pred_proba
    
    def _print_business_metrics(self, y_true, y_pred, y_pred_proba):
        print(f"\n Business Metrics:")
        
        # Default detection rate (recall for default class)
        default_recall = confusion_matrix(y_true, y_pred)[1,1] / (y_true == 1).sum()
        print(f"   Default Detection Rate: {default_recall*100:.2f}%")
        
        # False approval rate (Type II error)
        false_approval_rate = confusion_matrix(y_true, y_pred)[1,0] / (y_true == 1).sum()
        print(f"   False Approval Rate: {false_approval_rate*100:.2f}%")
        
        # Precision for approved loans (1 - FDR for class 0)
        approved_precision = confusion_matrix(y_true, y_pred)[0,0] / (y_pred == 0).sum()
        print(f"   Approved Loan Quality: {approved_precision*100:.2f}%")
        
        # Expected default rate in approved loans
        approved_default_rate = confusion_matrix(y_true, y_pred)[1,0] / (y_pred == 0).sum()
        print(f"   Expected Default Rate (Approved): {approved_default_rate*100:.2f}%")
        
        # Rejection rate
        rejection_rate = (y_pred == 1).sum() / len(y_pred)
        print(f"   Rejection Rate: {rejection_rate*100:.2f}%")
    
    def plot_evaluation(self, y_test, y_pred_proba, save_path=None):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 0].set_xlim([0.0, 1.0])
        axes[0, 0].set_ylim([0.0, 1.05])
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend(loc="lower right")
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        axes[0, 1].plot(recall, precision, color='green', lw=2,
                       label=f'PR curve (AP = {avg_precision:.3f})')
        axes[0, 1].axhline(y=y_test.mean(), color='navy', linestyle='--', 
                          label=f'Baseline ({y_test.mean():.3f})')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend(loc="upper right")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Feature Importance (Top 20)
        top_features = self.feature_importance.head(20)
        axes[1, 0].barh(range(len(top_features)), top_features['importance'])
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features['feature'])
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Top 20 Feature Importances')
        axes[1, 0].invert_yaxis()
        
        # 4. Prediction Distribution
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                   xticklabels=['No Default', 'Default'],
                   yticklabels=['No Default', 'Default'])
        axes[1, 1].set_ylabel('True Label')
        axes[1, 1].set_xlabel('Predicted Label')
        axes[1, 1].set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Evaluation plots saved to {save_path}")
        
        return fig
    
    def save_model(self, filepath='../models/credit_risk_model.pkl'):
        model_data = {
            'model': self.model,
            'threshold': self.threshold,
            'feature_importance': self.feature_importance,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        joblib.dump(model_data, filepath)
        print(f" Model saved to {filepath}")
    
    def load_model(self, filepath='../models/credit_risk_model.pkl'):
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.threshold = model_data['threshold']
        self.feature_importance = model_data['feature_importance']
        print(f" Model loaded from {filepath}")
        return self


if __name__ == "__main__":
    print("Credit Risk Model Training Pipeline")
    print("=" * 60)