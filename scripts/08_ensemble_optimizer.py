"""
Ensemble Optimizer for High-Accuracy Predictions
Combines multiple models trained with different strategies
"""
import numpy as np
import pickle
import json
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

def create_base_models():
    """Create a diverse set of base models"""
    base_models = [
        ('gb_strong', GradientBoostingClassifier(n_estimators=300, max_depth=6, 
                                                 learning_rate=0.03, subsample=0.85,
                                                 random_state=42)),
        ('rf_strong', RandomForestClassifier(n_estimators=250, max_depth=16,
                                             min_samples_split=2, min_samples_leaf=1,
                                             random_state=42, n_jobs=-1)),
        ('hist_gb', HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, random_state=42)),
        ('svm_rbf', SVC(kernel='rbf', C=15, gamma='scale', probability=True, random_state=42)),
    ]
    return base_models

def create_ensemble(base_models):
    """Create a voting ensemble with optimal weights"""
    voting_clf = VotingClassifier(
        estimators=base_models,
        voting='soft',  # Use probability estimates
        weights=[2, 2, 1.5, 1]  # Weighted voting based on model strengths
    )
    return voting_clf

def train_and_evaluate_ensemble(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train ensemble and evaluate on all sets"""
    print("[v0] Creating ensemble with optimized base models...")
    
    base_models = create_base_models()
    ensemble = create_ensemble(base_models)
    
    print("[v0] Training ensemble...")
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    y_pred_train = ensemble.predict(X_train)
    y_pred_val = ensemble.predict(X_val)
    y_pred_test = ensemble.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    val_acc = accuracy_score(y_val, y_pred_val)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    train_f1 = f1_score(y_train, y_pred_train)
    val_f1 = f1_score(y_val, y_pred_val)
    test_f1 = f1_score(y_test, y_pred_test)
    
    print(f"[v0] Train Accuracy: {train_acc:.4f} | Val Accuracy: {val_acc:.4f} | Test Accuracy: {test_acc:.4f}")
    print(f"[v0] Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | Test F1: {test_f1:.4f}")
    
    # Get probability estimates for AUC
    if hasattr(ensemble, 'predict_proba'):
        y_proba_test = ensemble.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, y_proba_test)
        print(f"[v0] Test ROC-AUC: {test_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    
    # Classification report
    report = classification_report(y_test, y_pred_test, output_dict=True)
    
    results = {
        'train_accuracy': float(train_acc),
        'val_accuracy': float(val_acc),
        'test_accuracy': float(test_acc),
        'train_f1': float(train_f1),
        'val_f1': float(val_f1),
        'test_f1': float(test_f1),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    if hasattr(ensemble, 'predict_proba'):
        results['test_auc'] = float(test_auc)
    
    pickle.dump(ensemble, open('output/ensemble_model.pkl', 'wb'))
    with open('output/ensemble_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[v0] Ensemble training complete!")
    
    return ensemble, results

def run_ensemble_optimizer():
    """Execute ensemble optimization"""
    with open('output/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    ensemble, results = train_and_evaluate_ensemble(X_train, X_val, X_test, y_train, y_val, y_test)
    
    return ensemble, results

if __name__ == "__main__":
    run_ensemble_optimizer()
