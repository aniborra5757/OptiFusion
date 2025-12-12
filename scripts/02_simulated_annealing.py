"""
Simulated Annealing Optimization Module
Optimizes feature selection and model hyperparameters
"""
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import math
import json

class SimulatedAnnealingOptimizer:
    def __init__(self, X_train, X_val, y_train, y_val, initial_temp=100, cooling_rate=0.95):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.best_solution = None
        self.best_score = 0
        self.temperature_history = []
        self.score_history = []
        
    def fitness(self, feature_mask):
        """Evaluate fitness of a feature selection"""
        if np.sum(feature_mask) == 0:
            return 0
        
        # Train model with selected features
        X_train_selected = self.X_train[:, feature_mask.astype(bool)]
        X_val_selected = self.X_val[:, feature_mask.astype(bool)]
        
        clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        clf.fit(X_train_selected, self.y_train)
        y_pred = clf.predict(X_val_selected)
        
        # Score: accuracy - penalty for too many features
        accuracy = accuracy_score(self.y_val, y_pred)
        feature_penalty = 0.001 * np.sum(feature_mask)
        score = accuracy - feature_penalty
        
        return score
    
    def get_neighbor(self, solution):
        """Generate neighboring solution by flipping random features"""
        neighbor = solution.copy()
        flip_idx = np.random.randint(0, len(neighbor))
        neighbor[flip_idx] = 1 - neighbor[flip_idx]
        return neighbor
    
    def optimize(self, n_features, max_iterations=100):
        """Run simulated annealing"""
        print("[v0] Starting Simulated Annealing optimization...")
        
        # Initialize random solution
        current_solution = np.random.randint(0, 2, n_features)
        current_score = self.fitness(current_solution)
        
        self.best_solution = current_solution.copy()
        self.best_score = current_score
        
        temperature = self.initial_temp
        
        for iteration in range(max_iterations):
            # Generate neighbor
            neighbor_solution = self.get_neighbor(current_solution)
            neighbor_score = self.fitness(neighbor_solution)
            
            # Acceptance probability
            delta = neighbor_score - current_score
            if delta > 0 or np.random.rand() < math.exp(delta / temperature):
                current_solution = neighbor_solution
                current_score = neighbor_score
            
            # Update best
            if current_score > self.best_score:
                self.best_solution = current_solution.copy()
                self.best_score = current_score
            
            # Cool down
            temperature *= self.cooling_rate
            
            self.temperature_history.append(temperature)
            self.score_history.append(self.best_score)
            
            if (iteration + 1) % 20 == 0:
                n_selected = np.sum(self.best_solution)
                print(f"[v0] Iteration {iteration+1}: Best Score = {self.best_score:.4f}, Features = {n_selected}")
        
        return self.best_solution, self.best_score

def run_simulated_annealing():
    """Execute SA optimization"""
    with open('output/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']
    
    optimizer = SimulatedAnnealingOptimizer(X_train, X_val, y_train, y_val)
    best_features, best_score = optimizer.optimize(n_features=X_train.shape[1], max_iterations=100)
    
    # Save results
    results = {
        'best_features': best_features.tolist(),
        'best_score': float(best_score),
        'n_selected_features': int(np.sum(best_features)),
        'temperature_history': optimizer.temperature_history,
        'score_history': optimizer.score_history
    }
    
    with open('output/sa_results.json', 'w') as f:
        json.dump(results, f)
    
    print(f"[v0] SA Optimization complete! Selected {np.sum(best_features)} features")
    print(f"[v0] Best score: {best_score:.4f}")
    
    return best_features, best_score

if __name__ == "__main__":
    run_simulated_annealing()
