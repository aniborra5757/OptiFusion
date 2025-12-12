# UCI Heart Disease Prediction Dataset - Information Guide

## Dataset Overview
The **UCI Heart Disease Dataset** is sourced from the UCI Machine Learning Repository and Kaggle. It contains medical records of **918 patients** with **13 clinical features** used to predict heart disease presence.

**Dataset Source:**
- **UCI Machine Learning Repository**: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
- **Kaggle Dataset**: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- **License**: Public Domain
- **Original Institution**: Cleveland Clinic Foundation

---

## Dataset Features (13 Total)

### 1. **Age** 
   - **Type**: Continuous (Integer)
   - **Range**: 29-77 years
   - **Description**: Age of the patient in years
   - **Relevance**: Strong predictor of heart disease risk

### 2. **Sex** 
   - **Type**: Binary (0/1)
   - **Values**: 0=Female, 1=Male
   - **Description**: Biological sex of patient
   - **Relevance**: Gender differences in disease presentation

### 3. **ChestPain**
   - **Type**: Categorical (1-4)
   - **Values**: 
     - 1 = Typical Angina
     - 2 = Atypical Angina
     - 3 = Non-anginal Pain
     - 4 = Asymptomatic
   - **Description**: Type of chest pain experienced
   - **Relevance**: Symptom type strongly indicates disease likelihood

### 4. **RestingBP (Resting Blood Pressure)**
   - **Type**: Continuous (Float)
   - **Range**: 94-200 mmHg
   - **Description**: Resting blood pressure on hospital admission
   - **Relevance**: Key cardiovascular health indicator; hypertension increases risk

### 5. **Cholesterol**
   - **Type**: Continuous (Float)
   - **Range**: 126-564 mg/dL
   - **Description**: Serum cholesterol level
   - **Relevance**: High cholesterol is a major risk factor for heart disease

### 6. **FastingBS (Fasting Blood Sugar)**
   - **Type**: Binary (0/1)
   - **Values**: 0=Blood sugar ≤120 mg/dL, 1=Blood sugar >120 mg/dL
   - **Description**: Whether fasting blood sugar is elevated
   - **Relevance**: Diabetes/metabolic disorder indicator

### 7. **RestingECG (Resting Electrocardiogram)**
   - **Type**: Categorical (0-2)
   - **Values**: 
     - 0 = Normal
     - 1 = ST-T wave abnormality
     - 2 = Left ventricular hypertrophy
   - **Description**: Results of resting electrocardiographic test
   - **Relevance**: Direct cardiac electrical activity measurement

### 8. **MaxHR (Maximum Heart Rate)**
   - **Type**: Continuous (Float)
   - **Range**: 71-202 bpm
   - **Description**: Maximum heart rate achieved during exercise stress test
   - **Relevance**: Heart rate response to stress; low values indicate cardiac dysfunction

### 9. **ExerciseAngina (Exercise-Induced Angina)**
   - **Type**: Binary (0/1)
   - **Values**: 0=No, 1=Yes
   - **Description**: Whether angina occurs during exercise
   - **Relevance**: Symptom indicator of insufficient blood flow to heart

### 10. **Oldpeak**
   - **Type**: Continuous (Float)
   - **Range**: 0-6.2 mm
   - **Description**: ST depression induced by exercise relative to rest (ST depression)
   - **Relevance**: ECG finding related to ischemia during stress

### 11. **ST_Slope (Slope of ST Segment)**
   - **Type**: Categorical (1-3)
   - **Values**: 
     - 1 = Upsloping
     - 2 = Flat
     - 3 = Downsloping
   - **Description**: Slope of the ST segment during exercise ECG
   - **Relevance**: Different slopes indicate different disease severity

### 12. **NumMajorVessels (Number of Major Vessels)**
   - **Type**: Categorical (0-3)
   - **Range**: 0, 1, 2, or 3
   - **Description**: Number of major vessels (0-3) colored by fluoroscopy
   - **Relevance**: Indicates coronary artery obstruction; higher = more blockage

### 13. **HeartDisease** (Target Variable)
   - **Type**: Binary (0/1)
   - **Values**: 0=No disease, 1=Disease present
   - **Description**: Diagnosis of heart disease (presence/absence)
   - **Class Distribution**: 
     - 0 (No Disease): ~498 patients (approximately 55%)
     - 1 (Disease): ~420 patients (approximately 45%)
   - **Balance**: Relatively balanced dataset

---

## Data Characteristics

| Characteristic | Value |
|---|---|
| **Number of Samples** | 918 |
| **Number of Features** | 13 |
| **Feature Types** | 4 Categorical + 5 Binary + 4 Continuous |
| **Missing Values** | Some records excluded; clean dataset |
| **Target Variable** | HeartDisease (Binary Classification) |
| **Class Balance** | Balanced (approximately 55% vs 45%) |
| **Dataset Type** | Multivariate |
| **Application** | Binary Classification |

---

## Clinical Significance

### Diagnostic Importance:
The dataset combines multiple diagnostic tests to predict coronary artery disease:
- **Symptoms**: Chest pain type, exercise-induced angina
- **Demographics**: Age, sex
- **Vital Signs**: Resting blood pressure, maximum heart rate
- **Laboratory Tests**: Cholesterol, fasting blood sugar
- **Diagnostic Tests**: ECG results, stress test findings, coronary angiography

### Key Predictors of Heart Disease:
1. **Chest Pain Type** - Primary symptom predictor
2. **Exercise-Induced Angina** - Strong symptom indicator
3. **ST Depression (Oldpeak)** - ECG abnormality measure
4. **Number of Major Vessels** - Coronary obstruction indicator
5. **Age** - Demographic risk factor
6. **Cholesterol & Blood Pressure** - Cardiovascular stress factors

### Clinical Context:
- Coronary artery disease affects 18.2 million Americans (>18 years)
- Leading cause of death in the USA
- Early prediction enables preventive interventions
- This dataset helps identify high-risk patients

---

## Model Performance Expectations

Using advanced ML techniques with this dataset:
- **Target Accuracy**: 95-97%
- **Primary Challenge**: Feature diversity (categorical, continuous, binary)
- **Best Algorithms**: Gradient Boosting, Random Forest, SVM with RBF kernel
- **Optimization Techniques**:
  - Simulated Annealing for hyperparameter tuning
  - Reinforcement Learning for model selection
  - Federated Learning for multi-client training
  - Polynomial feature engineering (captures interaction effects)
  - SelectKBest feature selection (reduces noise, selects top 30)

---

## How to Download the Dataset

### Option 1: Automatic Download (Our Pipeline)
The ML pipeline automatically downloads from UCI Machine Learning Repository.

### Option 2: Manual Download from Kaggle
\`\`\`bash
kaggle datasets download -d johnsmith88/heart-disease-dataset
unzip heart-disease-dataset.zip
\`\`\`

### Option 3: Direct UCI Link
Visit: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/

---

## References

- **Dataset**: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
- **Kaggle**: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- **Original Study**: Angiographic disease status (0 = normal, 1-4 = levels of stenosis)

---

## Data Preprocessing in Pipeline

1. **Missing Value Handling**: Rows with missing values (marked as '?') are removed
2. **Target Conversion**: Multi-class vessel counts converted to binary classification (0 vs 1+)
3. **Feature Scaling**: StandardScaler normalizes all features
4. **Polynomial Features**: Degree-2 polynomial features capture feature interactions
5. **Feature Selection**: SelectKBest selects top 30 features for model optimization
6. **Train-Val-Test Split**: 75.25% train, 10% validation, 14.75% test (stratified)
