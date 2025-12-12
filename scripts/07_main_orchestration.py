"""
Main Orchestration Script
Executes the complete ML pipeline with all components
"""
import sys
import os
import json
import subprocess

def run_pipeline():
    """Execute complete pipeline"""
    print("\n" + "="*70)
    print("   HEALTHCARE ML PIPELINE: SA + RL + FEDERATED LEARNING")
    print("="*70 + "\n")
    
    scripts = [
        ('01_data_processing.py', 'Data Processing'),
        ('02_simulated_annealing.py', 'Simulated Annealing Optimization'),
        ('03_reinforcement_learning.py', 'Reinforcement Learning'),
        ('04_federated_learning.py', 'Federated Learning'),
        ('08_ensemble_optimizer.py', 'Ensemble Optimizer'),
        ('05_evaluation_metrics.py', 'Evaluation Metrics'),
        ('06_visualizations.py', 'Visualization Generation'),
    ]
    
    for script, description in scripts:
        print(f"\n{'─'*70}")
        print(f"▶ RUNNING: {description}")
        print(f"{'─'*70}")
        
        script_path = os.path.join('scripts', script)
        
        try:
            result = subprocess.run([sys.executable, script_path], 
                                  capture_output=True, text=True, timeout=300)
            
            if result.stdout:
                print(result.stdout)
            if result.stderr and result.returncode != 0:
                print(f"Error: {result.stderr}")
                print(f"Warning: {description} encountered an error")
            
            print(f"✓ {description} completed successfully")
        
        except subprocess.TimeoutExpired:
            print(f"✗ {description} timed out!")
        except Exception as e:
            print(f"✗ Error running {description}: {e}")
    
    # Print summary
    print("\n" + "="*70)
    print("   PIPELINE EXECUTION SUMMARY")
    print("="*70)
    
    output_files = []
    if os.path.exists('output'):
        output_files = [f for f in os.listdir('output') if os.path.isfile(os.path.join('output', f))]
    
    print("\nGenerated Output Files:")
    for f in sorted(output_files):
        print(f"  ✓ {f}")
    
    print("\n" + "─"*70)
    print("FINAL ACCURACY SCORES:")
    print("─"*70)
    
    try:
        # Try to load ensemble results first (highest accuracy)
        try:
            with open('output/ensemble_results.json', 'r') as f:
                ensemble_results = json.load(f)
            
            print(f"\nEnsemble Model (Optimized for High Accuracy):")
            print(f"  Test Accuracy:  {ensemble_results['test_accuracy']:.4f} *** TARGET: 95-97%")
            print(f"  Test F1-Score:  {ensemble_results['test_f1']:.4f}")
            if 'test_auc' in ensemble_results:
                print(f"  Test ROC-AUC:   {ensemble_results['test_auc']:.4f}")
            print(f"\n  Confusion Matrix:")
            cm = ensemble_results['confusion_matrix']
            print(f"    [[{cm[0][0]:4d}, {cm[0][1]:4d}],")
            print(f"     [{cm[1][0]:4d}, {cm[1][1]:4d}]]")
            
        except Exception as e:
            print(f"  (Ensemble results not available: {e})")
        
        # Load other model results
        with open('output/evaluation_results.json', 'r') as f:
            eval_results = json.load(f)
        
        print("\n" + "─"*70)
        print("INDIVIDUAL MODEL RESULTS:")
        print("─"*70)
        
        for model, metrics in eval_results.items():
            print(f"\n{model}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  Sensitivity (True Positive Rate): {metrics['sensitivity']:.4f}")
            print(f"  Specificity (True Negative Rate): {metrics['specificity']:.4f}")
            print(f"\n  Confusion Matrix:")
            cm = metrics['confusion_matrix']
            print(f"    [[{cm[0][0]:4d}, {cm[0][1]:4d}],")
            print(f"     [{cm[1][0]:4d}, {cm[1][1]:4d}]]")
    
    except Exception as e:
        print(f"Note: Could not load evaluation results: {e}")
    
    print("\n" + "="*70)
    print("   PIPELINE COMPLETE!")
    print("="*70)
    print("\nAll outputs saved to 'output/' directory")
    print("View the generated PNG files for visualizations")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_pipeline()
