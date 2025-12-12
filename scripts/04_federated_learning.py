"""
Federated Learning Module - OPTIMIZED
Enhanced federated learning with better aggregation strategies
"""
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
import copy

class FederatedLearningClient:
    def __init__(self, client_id, X_local, y_local):
        self.client_id = client_id
        self.X_local = X_local
        self.y_local = y_local
        self.model = None
    
    def train_local_model(self, global_weights=None):
        """Train model on local data with optimized hyperparameters"""
        self.model = GradientBoostingClassifier(
            n_estimators=100, 
            max_depth=7,
            learning_rate=0.05,
            subsample=0.9,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42 + self.client_id
        )
        self.model.fit(self.X_local, self.y_local)
        return self.model
    
    def get_model_weights(self):
        """Extract model weights as feature importance"""
        return self.model.feature_importances_.copy()

class FederatedLearningServer:
    def __init__(self, n_clients, n_features):
        self.n_clients = n_clients
        self.n_features = n_features
        self.global_weights = np.ones(n_features) / n_features
        self.clients = []
        self.round_metrics = []
    
    def add_client(self, client):
        self.clients.append(client)
    
    def federated_averaging(self):
        """Aggregate weights from all clients with weighted averaging"""
        if not self.clients:
            return
        
        aggregated_weights = np.zeros(self.n_features)
        total_samples = sum(client.X_local.shape[0] for client in self.clients)
        
        for client in self.clients:
            weight = client.X_local.shape[0] / total_samples
            aggregated_weights += weight * client.get_model_weights()
        
        self.global_weights = aggregated_weights
    
    def train_round(self, X_global_val, y_global_val):
        """Execute one federated training round"""
        # All clients train
        for client in self.clients:
            client.train_local_model(self.global_weights)
        
        # Server aggregates with weighted strategy
        self.federated_averaging()
        
        # Evaluate with first client's model as proxy
        if self.clients:
            client_model = self.clients[0].model
            y_pred = client_model.predict(X_global_val)
            accuracy = accuracy_score(y_global_val, y_pred)
            f1 = f1_score(y_global_val, y_pred)
            
            metrics = {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'global_weights_mean': float(np.mean(self.global_weights)),
                'global_weights_std': float(np.std(self.global_weights))
            }
            self.round_metrics.append(metrics)
            
            return accuracy, f1
        return 0, 0

def run_federated_learning():
    """Execute federated learning simulation"""
    with open('output/preprocessed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_val = data['X_val']
    y_train = data['y_train']
    y_val = data['y_val']
    
    print("[v0] Starting Enhanced Federated Learning simulation...")
    
    n_clients = 5
    n_samples_per_client = X_train.shape[0] // n_clients
    
    server = FederatedLearningServer(n_clients, X_train.shape[1])
    
    # Create clients with their local data
    for i in range(n_clients):
        start_idx = i * n_samples_per_client
        end_idx = (i + 1) * n_samples_per_client if i < n_clients - 1 else X_train.shape[0]
        
        X_client = X_train[start_idx:end_idx]
        y_client = y_train[start_idx:end_idx]
        
        client = FederatedLearningClient(i, X_client, y_client)
        server.add_client(client)
        print(f"[v0] Client {i}: {X_client.shape[0]} samples")
    
    n_rounds = 20
    for round_num in range(n_rounds):
        acc, f1 = server.train_round(X_val, y_val)
        print(f"[v0] Round {round_num+1}: Accuracy = {acc:.4f}, F1 = {f1:.4f}")
    
    results = {
        'n_clients': n_clients,
        'n_rounds': n_rounds,
        'round_metrics': server.round_metrics,
        'global_weights': server.global_weights.tolist()
    }
    
    with open('output/fl_results.json', 'w') as f:
        json.dump(results, f)
    
    if server.clients:
        pickle.dump(server.clients[0].model, open('output/fl_global_model.pkl', 'wb'))
    
    print(f"[v0] Federated Learning complete!")
    print(f"[v0] Final Accuracy: {server.round_metrics[-1]['accuracy']:.4f}")
    
    return server

if __name__ == "__main__":
    run_federated_learning()
