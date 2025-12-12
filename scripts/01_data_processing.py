"""
Healthcare Data Processing Module - UCI Heart Disease Dataset
Loads and preprocesses Kaggle UCI Heart Disease dataset with advanced feature engineering
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import os

def load_healthcare_data():
    """Load UCI Heart Disease dataset"""
    print("[v0] Loading UCI Heart Disease dataset...")
    
    # UCI Heart Disease Dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    column_names = [
        'Age', 'Sex', 'ChestPain', 'RestingBP', 'Cholesterol', 
        'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
        'Oldpeak', 'ST_Slope', 'NumMajorVessels', 'Thalassemia', 'HeartDisease'
    ]
    
    try:
        df = pd.read_csv(url, names=column_names, na_values='?')
        print(f" Dataset loaded successfully from UCI repository!")
    except:
        print(" Could not download. Creating sample dataset for demo...")
        np.random.seed(42)
        n_samples = 918
        df = pd.DataFrame({
            'Age': np.random.uniform(29, 77, n_samples),
            'Sex': np.random.randint(0, 2, n_samples),
            'ChestPain': np.random.randint(1, 5, n_samples),
            'RestingBP': np.random.uniform(94, 200, n_samples),
            'Cholesterol': np.random.uniform(126, 564, n_samples),
            'FastingBS': np.random.randint(0, 2, n_samples),
            'RestingECG': np.random.randint(0, 3, n_samples),
            'MaxHR': np.random.uniform(71, 202, n_samples),
            'ExerciseAngina': np.random.randint(0, 2, n_samples),
            'Oldpeak': np.random.uniform(0, 6.2, n_samples),
            'ST_Slope': np.random.randint(1, 4, n_samples),
            'NumMajorVessels': np.random.randint(0, 4, n_samples),
            'Thalassemia': np.random.randint(3, 8, n_samples),
            'HeartDisease': np.random.randint(0, 2, n_samples)
        })
    
    # Handle missing values
    df = df.dropna()
    
    # Convert target to binary (0 = no disease, 1 = disease present)
    df['HeartDisease'] = (df['HeartDisease'] > 0).astype(int)
    
    X = df.drop('HeartDisease', axis=1).values
    y = df['HeartDisease'].values
    
    feature_names = df.drop('HeartDisease', axis=1).columns.tolist()
    
    print(f"\n Dataset shape: {df.shape}")
    print(f"\n Features ({len(feature_names)}):")
    for i, fname in enumerate(feature_names, 1):
        print(f"     {i}. {fname}")
    
    print(f"\n Total samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f" Class distribution: No Disease={np.sum(y==0)}, Disease={np.sum(y==1)}")
    print(f" Disease Prevalence: {np.mean(y):.2%}")
    
    return X, y, feature_names, df

def preprocess_data(X, y, test_size=0.15, val_size=0.1):
    """Split and scale data with advanced feature engineering"""
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Further split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size/(1-test_size), 
        random_state=42, stratify=y_train
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Polynomial feature engineering
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train_scaled)
    X_val_poly = poly_features.transform(X_val_scaled)
    X_test_poly = poly_features.transform(X_test_scaled)
    
    # Feature selection using SelectKBest
    selector = SelectKBest(f_classif, k=min(30, X_train_poly.shape[1]))
    X_train_selected = selector.fit_transform(X_train_poly, y_train)
    X_val_selected = selector.transform(X_val_poly)
    X_test_selected = selector.transform(X_test_poly)
    
    print(f"\n Data split:")
    print(f"     Train set: {X_train_selected.shape}")
    print(f"     Val set: {X_val_selected.shape}")
    print(f"     Test set: {X_test_selected.shape}")
    print(f" Features after engineering & selection: {X_train_selected.shape[1]}")
    
    os.makedirs('output', exist_ok=True)
    
    # Save preprocessed data
    data = {
        'X_train': X_train_selected,
        'X_val': X_val_selected,
        'X_test': X_test_selected,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'scaler': scaler,
        'poly_features': poly_features,
        'selector': selector
    }
    
    with open('output/preprocessed_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    
    print(f" Data preprocessing complete with advanced feature engineering\n")
    
    return data

if __name__ == "__main__":
    X, y, features, df = load_healthcare_data()
    data = preprocess_data(X, y)
    print(" Data processing complete!")
