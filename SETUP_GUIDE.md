# Healthcare ML Pipeline - VS Code Setup Guide

## Prerequisites
- Python 3.8+ installed on your system
- VS Code installed
- Git (optional, for cloning)

## Step-by-Step Instructions

### Step 1: Download and Open Project in VS Code

#### Option A: Using GitHub (Recommended)
1. Click the GitHub button in v0 (top right)
2. Sign in to GitHub and create a new repository
3. Copy the repository link
4. Open VS Code
5. Press `Ctrl + K` then `Ctrl + O` (Mac: `Cmd + K` then `Cmd + O`)
6. Paste the repo link and clone it
7. Open the cloned folder in VS Code

#### Option B: Download ZIP
1. Click the three dots menu (top right of v0)
2. Select "Download ZIP"
3. Extract the ZIP file to a folder
4. Open VS Code
5. Go to File → Open Folder
6. Select the extracted project folder

---

### Step 2: Set Up Python Environment

1. **Open Terminal in VS Code**
   - Press `Ctrl + ~` (Mac: `Cmd + ~`)
   - Or go to Terminal → New Terminal

2. **Create Virtual Environment** (Recommended)
   \`\`\`bash
   python -m venv venv
   \`\`\`

3. **Activate Virtual Environment**
   - **Windows:**
     \`\`\`bash
     venv\Scripts\activate
     \`\`\`
   - **Mac/Linux:**
     \`\`\`bash
     source venv/bin/activate
     \`\`\`
   
   You should see `(venv)` at the start of your terminal line.

4. **Install Required Dependencies**
   \`\`\`bash
   pip install numpy pandas scikit-learn matplotlib seaborn scipy joblib
   \`\`\`

---

### Step 3: Select Python Interpreter in VS Code

1. Press `Ctrl + Shift + P` (Mac: `Cmd + Shift + P`)
2. Type "Python: Select Interpreter"
3. Choose the interpreter from your `venv` folder
   - Should show something like `./venv/bin/python`
4. Click on it

---

### Step 4: Run the Complete Pipeline

1. **Navigate to Scripts Folder**
   - In the file explorer (left sidebar), expand the `scripts` folder

2. **Run Main Orchestration Script**
   - Right-click on `scripts/07_main_orchestration.py`
   - Select "Run Python File in Terminal"
   
   OR manually in terminal:
   \`\`\`bash
   python scripts/07_main_orchestration.py
   \`\`\`

3. **Wait for Execution**
   - The pipeline will run through all 7 steps
   - You'll see progress messages in the terminal
   - Total runtime: ~2-3 minutes

---

### Step 5: View Results and Outputs

#### View Console Output
- All accuracy scores, metrics, and logs appear in the VS Code terminal
- Look for:
  - `[v0] Simulated Annealing Results`
  - `[v0] Reinforcement Learning Metrics`
  - `[v0] Federated Learning Accuracy`
  - `[v0] Ensemble Optimizer Results` (should show 95-97% accuracy)

#### Access Generated Files
1. In VS Code file explorer, navigate to the `output/` folder
2. You'll find:
   - `confusion_matrix_sa.png`
   - `confusion_matrix_rl.png`
   - `confusion_matrix_federated.png`
   - `confusion_matrix_ensemble.png`
   - `accuracy_comparison.png`
   - `model_radar_chart.png`
   - `sa_convergence_plot.png`
   - `federated_learning_convergence.png`
   - `ensemble_feature_importance.png`
   - `roc_curves_comparison.png`

#### View Visualizations
1. Click on any `.png` file in the explorer
2. VS Code will open it in a built-in image viewer
3. Or double-click to open in your default image viewer

---

### Step 6: Run Individual Scripts (Optional)

If you want to run specific components:

\`\`\`bash
# Data Processing
python scripts/01_data_processing.py

# Simulated Annealing
python scripts/02_simulated_annealing.py

# Reinforcement Learning
python scripts/03_reinforcement_learning.py

# Federated Learning
python scripts/04_federated_learning.py

# Ensemble Optimizer
python scripts/08_ensemble_optimizer.py

# Visualizations
python scripts/06_visualizations.py

# Evaluation Metrics
python scripts/05_evaluation_metrics.py
\`\`\`

---

### Step 7: Debugging & Development

#### Enable Debug Mode
1. Click on the Run icon (left sidebar, looks like a play button)
2. Click "Create a launch.json file"
3. Select "Python"
4. Run with debugger: Press `F5`

#### Add Breakpoints
1. Click on the line number where you want to pause
2. A red dot appears
3. Run with debugger to pause at that line

#### View Variables
- During debug, variables appear in the Debug Console
- Hover over variables in code to see their values

---

### Troubleshooting

#### Issue: "Python command not found"
- **Solution:** Make sure Python is installed and added to PATH
- Test by running: `python --version`

#### Issue: "ModuleNotFoundError"
- **Solution:** Ensure virtual environment is activated and dependencies installed
- Rerun: `pip install numpy pandas scikit-learn matplotlib seaborn scipy joblib`

#### Issue: "Permission denied" (Mac/Linux)
- **Solution:** Make scripts executable
  \`\`\`bash
  chmod +x scripts/*.py
  \`\`\`

#### Issue: "No output folder created"
- **Solution:** Check that you're running from the project root directory
- Make sure you have write permissions in the project folder

---

### Expected Output Example

\`\`\`
[v0] ===== HEALTHCARE ML PIPELINE =====
[v0] Step 1: Data Processing... ✓
[v0] Dataset loaded: 569 samples, 30 features
[v0] Step 2: Simulated Annealing... ✓
[v0] Accuracy: 92.3%
[v0] Step 3: Reinforcement Learning... ✓
[v0] Accuracy: 94.7%
[v0] Step 4: Federated Learning (5 clients)... ✓
[v0] Global Model Accuracy: 93.1%
[v0] Step 5: Ensemble Optimizer... ✓
[v0] Ensemble Accuracy: 96.5% ✨
[v0] Step 6: Generating Visualizations... ✓
[v0] All plots saved to output/
[v0] ===== PIPELINE COMPLETE =====
\`\`\`

---

### Next Steps

1. **Modify Hyperparameters**
   - Open individual script files
   - Change parameters like:
     - `num_rl_episodes` in `03_reinforcement_learning.py`
     - `num_federated_rounds` in `04_federated_learning.py`
   - Rerun to see how accuracy changes

2. **Use Different Datasets**
   - Replace `load_breast_cancer()` in `01_data_processing.py`
   - Use any sklearn dataset or load CSV files

3. **Export Results**
   - Results are automatically saved in `output/`
   - Share PNG visualizations and accuracy metrics

---

### Quick Reference Commands

\`\`\`bash
# Activate environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn scipy joblib

# Run entire pipeline
python scripts/07_main_orchestration.py

# Run specific script
python scripts/02_simulated_annealing.py

# Deactivate environment
deactivate
\`\`\`

---

For any issues or questions, check the debug output in the terminal for detailed error messages.
