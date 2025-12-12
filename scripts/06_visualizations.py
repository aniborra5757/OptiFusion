"""
Visualization Module
Creates comprehensive visualizations and saves plots
"""
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_confusion_matrices(metrics):
    """Plot confusion matrices for all models"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Confusion Matrices - Healthcare Model Evaluation', fontsize=16, fontweight='bold')
    
    model_names = list(metrics.keys())
    for idx, (model_name, model_metrics) in enumerate(metrics.items()):
        if idx < 2:
            cm = np.array(model_metrics['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=False, annot_kws={'size': 14})
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
            axes[idx].set_xticklabels(['Negative', 'Positive'])
            axes[idx].set_yticklabels(['Negative', 'Positive'])
    
    plt.tight_layout()
    plt.savefig('output/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("[v0] Saved: confusion_matrices.png")
    plt.close()

def plot_accuracy_comparison(metrics):
    """Plot accuracy and F1-score comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Performance Comparison - Healthcare Systems', fontsize=16, fontweight='bold')
    
    model_names = list(metrics.keys())
    accuracies = [metrics[m]['accuracy'] for m in model_names]
    f1_scores = [metrics[m]['f1_score'] for m in model_names]
    
    # Accuracy
    colors = ['#FF6B6B', '#4ECDC4']
    bars1 = ax1.bar(model_names, accuracies, color=colors[:len(model_names)], alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # F1-Score
    bars2 = ax2.bar(model_names, f1_scores, color=colors[:len(model_names)], alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1-Score Comparison', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/accuracy_f1_comparison.png', dpi=300, bbox_inches='tight')
    print("[v0] Saved: accuracy_f1_comparison.png")
    plt.close()

def plot_metrics_radar(metrics):
    """Plot radar chart for multi-metric comparison"""
    from math import pi
    
    model_names = list(metrics.keys())
    metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    angles = [n / float(len(metric_keys)) * 2 * pi for n in range(len(metric_keys))]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    colors_list = ['#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3']
    
    for model_idx, model_name in enumerate(model_names):
        values = [metrics[model_name][key] for key in metric_keys]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, 
               color=colors_list[model_idx % len(colors_list)], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors_list[model_idx % len(colors_list)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_keys, fontsize=10, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title('Multi-Metric Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('output/metrics_radar.png', dpi=300, bbox_inches='tight')
    print("[v0] Saved: metrics_radar.png")
    plt.close()

def plot_sa_optimization(sa_results):
    """Plot simulated annealing convergence"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Simulated Annealing Optimization Progress', fontsize=16, fontweight='bold')
    
    # Score history
    ax1.plot(sa_results['score_history'], linewidth=2, color='#FF6B6B', alpha=0.8)
    ax1.fill_between(range(len(sa_results['score_history'])), 
                     sa_results['score_history'], alpha=0.3, color='#FF6B6B')
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Best Score', fontsize=12, fontweight='bold')
    ax1.set_title('Fitness Score Convergence', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Temperature schedule
    ax2.plot(sa_results['temperature_history'], linewidth=2, color='#4ECDC4', alpha=0.8)
    ax2.fill_between(range(len(sa_results['temperature_history'])), 
                     sa_results['temperature_history'], alpha=0.3, color='#4ECDC4')
    ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Temperature', fontsize=12, fontweight='bold')
    ax2.set_title('Cooling Schedule', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/sa_optimization.png', dpi=300, bbox_inches='tight')
    print("[v0] Saved: sa_optimization.png")
    plt.close()

def plot_rl_training(rl_results):
    """Plot RL training progress"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    episode_rewards = rl_results['episode_rewards']
    
    ax.plot(episode_rewards, marker='o', linewidth=2, markersize=6, 
           color='#FF6B6B', alpha=0.8, label='Episode Reward')
    
    # Add moving average
    window = 3
    moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    ax.plot(range(window-1, len(episode_rewards)), moving_avg, linewidth=3, 
           color='#4ECDC4', alpha=0.8, label=f'{window}-Episode Moving Average')
    
    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold')
    ax.set_title('Reinforcement Learning Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/rl_training.png', dpi=300, bbox_inches='tight')
    print("[v0] Saved: rl_training.png")
    plt.close()

def plot_fl_rounds(fl_results):
    """Plot federated learning round performance"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Federated Learning Training Progress', fontsize=16, fontweight='bold')
    
    round_metrics = fl_results['round_metrics']
    rounds = range(1, len(round_metrics) + 1)
    
    accuracies = [m['accuracy'] for m in round_metrics]
    f1_scores = [m['f1_score'] for m in round_metrics]
    
    # Accuracy
    ax1.plot(rounds, accuracies, marker='s', linewidth=2, markersize=8, 
            color='#FF6B6B', alpha=0.8)
    ax1.fill_between(rounds, accuracies, alpha=0.3, color='#FF6B6B')
    ax1.set_xlabel('Federated Round', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Global Model Accuracy Over Rounds', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # F1-Score
    ax2.plot(rounds, f1_scores, marker='s', linewidth=2, markersize=8, 
            color='#4ECDC4', alpha=0.8)
    ax2.fill_between(rounds, f1_scores, alpha=0.3, color='#4ECDC4')
    ax2.set_xlabel('Federated Round', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax2.set_title('Global Model F1-Score Over Rounds', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('output/fl_training.png', dpi=300, bbox_inches='tight')
    print("[v0] Saved: fl_training.png")
    plt.close()

def run_visualizations():
    """Generate all visualizations"""
    print("[v0] Generating visualizations...")
    
    # Load results
    with open('output/evaluation_results.json', 'r') as f:
        eval_results = json.load(f)
    
    with open('output/sa_results.json', 'r') as f:
        sa_results = json.load(f)
    
    with open('output/rl_results.json', 'r') as f:
        rl_results = json.load(f)
    
    with open('output/fl_results.json', 'r') as f:
        fl_results = json.load(f)
    
    # Generate plots
    plot_confusion_matrices(eval_results)
    plot_accuracy_comparison(eval_results)
    plot_metrics_radar(eval_results)
    plot_sa_optimization(sa_results)
    plot_rl_training(rl_results)
    plot_fl_rounds(fl_results)
    
    print("[v0] All visualizations saved to output/ directory!")

if __name__ == "__main__":
    run_visualizations()
