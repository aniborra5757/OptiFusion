<h1 align="center">⚡ OptiFusion ⚡</h1>

<p align="center">SA • RL • FL — Hybrid Optimization Framework</p>

<p align="center">
  <img src="https://img.shields.io/badge/Optimization-Hybrid-blueviolet?style=flat-square" />
</p>

<hr/>




<p align="center">

<!-- Badges -->
<img src="https://img.shields.io/badge/Build-Stable-brightgreen?style=for-the-badge" />
<img src="https://img.shields.io/badge/Version-1.0.0-blue?style=for-the-badge" />
<img src="https://img.shields.io/badge/Datasets-Heart%20Disease%2C%20Heart%20Failure-orange?style=for-the-badge" />
<img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />

</p>

---

## 📌 Overview

**OptiFusion** combines three powerful optimization paradigms into a single machine learning framework:

- 🔥 **Simulated Annealing (SA)** — Global search & feature optimization  
- 🧠 **Reinforcement Learning (RL)** — Reward-driven configuration tuning  
- 🌐 **Federated Learning (FL)** — Distributed model training without data sharing  

This hybrid architecture delivers **robust**, **scalable**, and **high-performance** optimization suitable for ML tasks, currently applied to **heart disease prediction**.

> **Description:**  
> *OptiFusion combines multiple optimization strategies—SA, RL, and FL—into a unified ML framework for efficient model and feature optimization.*

---

## 🏗️ Architecture Highlights

✔ Modular pipeline  
✔ Independent + ensemble optimization modes  
✔ Automatic metric evaluation  
✔ Visualization-ready outputs  
✔ Serialized models for reuse  
✔ Dataset guides included  

---

## 📂 Project Structure

```text
OptiFusion/
│
├── scripts/                         # Core modules
│   ├── 01_data_processing.py
│   ├── 02_simulated_annealing.py
│   ├── 03_reinforcement_learning.py
│   ├── 04_federated_learning.py
│   ├── 05_evaluation_metrics.py
│   ├── 06_visualizations.py
│   ├── 07_main_orchestration.py
│   └── 08_ensemble_optimizer.py
│
├── output/                          # Results, models & plots
│   ├── accuracy_f1_comparison.png
│   ├── confusion_matrices.png
│   ├── ensemble_model.pkl
│   ├── ensemble_results.json
│   ├── fl_results.json
│   ├── rl_best_model.pkl
│   ├── rl_results.json
│   ├── sa_results.json
│   ├── evaluation_results.json
│   └── metrics_radar.png
│
├── DATASET_INFO.md
├── HEART_FAILURE_DATASET_GUIDE.md
├── SETUP_GUIDE.md
└── package.json
````

---

## 🧠 Optimization Modules

### **1️⃣ Simulated Annealing (SA)**
- Global stochastic search  
- Optimizes features & hyperparameters  
- Outputs → `sa_results.json`, `sa_optimization.png`

---

### **2️⃣ Reinforcement Learning (RL)**
- Learns optimal policy for model configs  
- Detects high-performing states through reward functions  
- Outputs → `rl_best_model.pkl`, `rl_results.json`

---

### **3️⃣ Federated Learning (FL)**
- Distributed training without centralizing data  
- Secure gradient-based updates  
- Outputs → `fl_global_model.pkl`, `fl_results.json`

---

### **4️⃣ Ensemble Optimization**
- Fuses SA + RL + FL models  
- Produces best generalizable performance  
- Outputs → `ensemble_model.pkl`

---

## 📊 Compact Results (Clean View)

| Method | Accuracy | F1-Score | Summary |
|--------|----------|----------|---------|
| SA | ~93% | ~92% | Good global search |
| RL | ~95% | ~94% | Strong policy learning |
| FL | ~91% | ~90% | Robust distributed model |
| **Ensemble** | **99%** | **99%** | Best combined performance |

🎯 Full graphs available in `/output`.

---

## 🛠️ Installation

```bash
git clone https://github.com/aniborra5757/OptiFusion.git
cd OptiFusion
pip install -r requirements.txt
````

---

## ▶️ Running the Pipeline

```bash
python scripts/07_main_orchestration.py
```

This will:

* preprocess dataset
* run SA, RL, FL optimizers
* evaluate metrics
* generate visualizations
* save models & results

---

## 📘 Documentation Files

* **SETUP_GUIDE.md** — Environment + execution guide
* **DATASET_INFO.md** — Source dataset details
* **HEART_FAILURE_DATASET_GUIDE.md** — Clinical feature explanation

---

## 🌟 Future Roadmap

* Privacy-preserving FedAvg + differential privacy
* Multi-agent RL for deeper optimization
* Interactive Streamlit UI for results exploration
* Support for more medical datasets
* MLOps-ready pipelines (CI/CD, automated evaluation)

---
## 👥 Team & Contributors

<p>
  <a href="https://github.com/aniborra5757"><img src="https://img.shields.io/badge/Ani%20(Lead)-GitHub-blue?style=for-the-badge&logo=github"></a>
  <a href="https://github.com/Manvitha1007"><img src="https://img.shields.io/badge/Manvitha-GitHub-blue?style=for-the-badge&logo=github"></a>
  <a href="https://github.com/QueenyVempa"><img src="https://img.shields.io/badge/Queeny%20Vempa-GitHub-blue?style=for-the-badge&logo=github"></a>
  <a href="https://github.com/varshini-1407"><img src="https://img.shields.io/badge/Varshini-GitHub-blue?style=for-the-badge&logo=github"></a>
</p>


<p align="center">
  <strong>✨ OptiFusion — Optimizing the Future of Machine Learning ✨</strong>
</p>

