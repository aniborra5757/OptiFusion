"""
Evaluation and Metrics Module
Generates accuracy scores, confusion matrices, and comprehensive metrics
"""
import numpy as np
import pickle
import json
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                           roc_auc_score, precision_score, recall_score, f1_score,
                           roc_curve, auc)
import csv

def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    y_pred_proba = None
    
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    except:
        # If model doesn't support predict_proba, use decision_function
        try:
            y_pred_proba = model.decision_function(X_test)
            # Normalize to [0, 1]
            y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
        except:
            y_pred_proba = y_pred.astype(float)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'true_negatives': int(cm[0, 0]),
        'false_positives': int(cm[0, 1]),
        'false_negatives': int(cm[1, 0]),
        'true_positives': int(cm[1, 1]),
        'specificity': float(cm[0, 0] / (cm[0, 0] + cm[0, 1])) if (cm[0, 0] + cm[0, 1]) > 0 else 0,
        'sensitivity': float(cm[1, 1] / (cm[1, 0] + cm[1, 1])) if (cm[1, 0] + cm[1, 1]) > 0 else 0,
        'classification_report': class_report
    }
    
    return metrics

def run_evaluation():
    """Run comprehensive evaluation"""
    print("[v0] Running comprehensive evaluation...")
    
    with open('output/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_test = data['X_test']
    y_test = data['y_test']
    
    all_results = {}
    
    # Evaluate Ensemble model
    try:
        with open('output/ensemble_model.pkl', 'rb') as f:
            ensemble_model = pickle.load(f)
        ensemble_metrics = evaluate_model(ensemble_model, X_test, y_test, "Ensemble Model (Optimized)")
        all_results['Ensemble_Model'] = ensemble_metrics
        print(f"[v0] Ensemble Model - Accuracy: {ensemble_metrics['accuracy']:.4f}, F1: {ensemble_metrics['f1_score']:.4f}")
    except Exception as e:
        print(f"[v0] Could not load ensemble model: {e}")
    
    # Evaluate RL model
    try:
        with open('output/rl_best_model.pkl', 'rb') as f:
            rl_model = pickle.load(f)
        rl_metrics = evaluate_model(rl_model, X_test, y_test, "Reinforcement Learning Model")
        all_results['RL_Model'] = rl_metrics
        print(f"[v0] RL Model - Accuracy: {rl_metrics['accuracy']:.4f}, F1: {rl_metrics['f1_score']:.4f}")
    except Exception as e:
        print(f"[v0] Could not load RL model: {e}")
    
    # Evaluate FL model
    try:
        with open('output/fl_global_model.pkl', 'rb') as f:
            fl_model = pickle.load(f)
        fl_metrics = evaluate_model(fl_model, X_test, y_test, "Federated Learning Global Model")
        all_results['FL_Model'] = fl_metrics
        print(f"[v0] FL Model - Accuracy: {fl_metrics['accuracy']:.4f}, F1: {fl_metrics['f1_score']:.4f}")
    except Exception as e:
        print(f"[v0] Could not load FL model: {e}")
    
    # Save all results
    with open('output/evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("[v0] Evaluation complete!")
    return all_results

if __name__ == "__main__":
    run_evaluation()
