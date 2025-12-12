"""
Reinforcement Learning Module - OPTIMIZED
Uses enhanced Q-Learning with ensemble methods for high accuracy
"""
import numpy as np
import pickle
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                             AdaBoostClassifier, HistGradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from collections import defaultdict

class RLModelSelector:
    def __init__(self, X_train, X_val, y_train, y_val, learning_rate=0.15, epsilon=0.05, gamma=0.95):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.learning_rate = learning_rate
        self.epsilon = epsilon  # Reduced epsilon for less exploration
        self.gamma = gamma  # Increased gamma for future rewards
        self.Q_table = defaultdict(float)
        self.episode_rewards = []
        
        self.actions = {
            'lr_high_c': LogisticRegression(C=0.01, max_iter=5000, random_state=42, solver='lbfgs'),
            'rf_optimized': RandomForestClassifier(n_estimators=200, max_depth=15, 
                                                   min_samples_split=2, min_samples_leaf=1,
                                                   random_state=42, n_jobs=-1),
            'gb_optimized': GradientBoostingClassifier(n_estimators=200, max_depth=5, 
                                                       learning_rate=0.05, subsample=0.8,
                                                       random_state=42),
            'svm_rbf': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
            'ada_boost': AdaBoostClassifier(n_estimators=100, learning_rate=0.8, random_state=42),
            'hist_gb': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42),
            'knn_optimized': KNeighborsClassifier(n_neighbors=5, weights='distance', metric='minkowski')
        }
    
    def get_reward(self, model):
        """Train and evaluate model with cross-validation"""
        try:
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            val_model = model.__class__(**model.get_params())
            val_model.fit(self.X_train, self.y_train)
            y_pred = val_model.predict(self.X_val)
            
            accuracy = accuracy_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred)
            
            # Try to get ROC-AUC if model supports probability
            try:
                if hasattr(val_model, 'predict_proba'):
                    y_proba = val_model.predict_proba(self.X_val)[:, 1]
                    auc = roc_auc_score(self.y_val, y_proba)
                else:
                    auc = accuracy
            except:
                auc = accuracy
            
            reward = 0.5 * accuracy + 0.2 * f1 + 0.2 * auc + 0.1 * np.mean(cv_scores)
            return reward
        except Exception as e:
            print(f"[v0] Error in reward calculation: {e}")
            return 0.0
    
    def select_action(self, state, available_actions):
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(available_actions)
        else:
            # Select best known action
            best_action = max(available_actions, 
                            key=lambda a: self.Q_table[(state, a)])
            return best_action
    
    def train(self, episodes=30):  # Increased episodes for better convergence
        """Train Q-learning agent with enhanced exploration"""
        print(" Starting Enhanced Reinforcement Learning training...")
        
        action_list = list(self.actions.keys())
        
        for episode in range(episodes):
            state = 0
            episode_reward = 0
            
            for step in range(7):  # Increased max steps
                action = self.select_action(state, action_list)
                model_copy = self.actions[action].__class__(**self.actions[action].get_params())
                reward = self.get_reward(model_copy)
                
                next_state = state + 1
                
                current_q = self.Q_table[(state, action)]
                max_next_q = max(self.Q_table[(next_state, a)] 
                               for a in action_list) if step < 6 else 0
                new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
                self.Q_table[(state, action)] = new_q
                
                episode_reward += reward
                state = next_state
            
            self.episode_rewards.append(episode_reward)
            
            if (episode + 1) % 5 == 0:
                avg_reward = np.mean(self.episode_rewards[-5:])
                print(f"[v0] Episode {episode+1}: Avg Reward = {avg_reward:.4f}")
        
        best_action = max(self.actions.keys(), 
                         key=lambda a: self.Q_table[(0, a)])
        print(f"[v0] Best model selected: {best_action}")
        return best_action
    
    def get_best_model(self):
        """Get the best model based on training"""
        best_action = max(self.actions.keys(), 
                         key=lambda a: self.Q_table[(0, a)])
        return best_action, self.actions[best_action]

def run_reinforcement_learning():
    """Execute RL model selection"""
    with open('output/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']
    
    rl_selector = RLModelSelector(X_train, X_val, y_train, y_val)
    best_action = rl_selector.train(episodes=30)
    best_model_name, best_model = rl_selector.get_best_model()
    
    # Train final model
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_pred)
    val_f1 = f1_score(y_val, y_pred)
    
    results = {
        'best_model': best_model_name,
        'validation_accuracy': float(val_accuracy),
        'validation_f1': float(val_f1),
        'episode_rewards': rl_selector.episode_rewards
    }
    
    with open('output/rl_results.json', 'w') as f:
        json.dump(results, f)
    
    pickle.dump(best_model, open('output/rl_best_model.pkl', 'wb'))
    
    print(f" RL Training complete! Best model: {best_model_name}")
    print(f" Validation Accuracy: {val_accuracy:.4f}, F1-Score: {val_f1:.4f}")
    
    return best_model, best_model_name

if __name__ == "__main__":
    run_reinforcement_learning()
