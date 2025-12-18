# Reinforcement Learning Tree (RLT) – Comprehensive Machine Learning Pipeline

A complete, production-ready framework for data exploration, preparation, model training, and deployment of advanced machine learning models based on Reinforcement Learning Trees (RLT).

## Overview

This project provides an **end-to-end machine learning pipeline** designed for:
- **Data Understanding**: Automated exploratory data analysis, statistical profiling, and advanced visualizations
- **Data Preparation**: Intelligent missing value imputation, categorical encoding, and feature scaling
- **Model Development**: Implementation and training of RLT algorithms with comparison against benchmark models (Random Forest, Extra Trees)
- **Hyperparameter Optimization**: Adaptive parameter tuning based on dataset characteristics
- **Model Deployment**: Interactive web application via Streamlit for inference and predictions
- **Research & Benchmarking**: Comprehensive evaluation framework for academic publication

## Project Architecture

```
rlt-project/
│
├── 📊 Data Analysis & Exploration
│   ├── understanding_data.py          
│   └── plots/                         
│
├── 🔧 Data Preparation & Preprocessing
│   ├── preparation_data.py            
│   └── data/                          
│
├── 🤖 Machine Learning Core
│   ├── rlt_module.py                  
│   ├── Enhanced_rlt.py               
│   ├── run_rlt.py               
│             
│
├── 🧪 Experimental & Testing
│   ├── run_rlt.py                      
│   └── run_simulation_testing.py
└    ── run_real_datasets.py       
│
├── 🌐 Web Application & Deployment
│   └── rlt_streamlit_app/
│       └── app.py                      
│
├── 📚 Documentation
│   ├── README.md                      
│   └── requirements.txt              
│
└── 💾 Generated Outputs
    ├── rlt_models/                    
    ├── dso2/                           
    │   ├── models/                   
    │   └── results/                   
    ├── understanding.csv              
    └── plots/                        
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- 2GB RAM minimum (4GB recommended for large datasets)
- Unix/Linux, macOS, or Windows environment

### Quick Start Installation

```bash
# 1. Clone or download the project
git clone <repository-url>
cd rlt-project

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate              # Linux/macOS
# or
venv\Scripts\activate                 # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import rlt_module; print('✅ Installation successful!')"
```

### Conda Installation (Alternative)

```bash
conda create -n rlt-env python=3.8
conda activate rlt-env
pip install -r requirements.txt
```

---

## Module Documentation

### 1. understanding_data.py – Exploratory Data Analysis

Automated analysis module for dataset characterization, feature profiling, and visual analytics.

**Key Capabilities:**
- 🔍 Automatic target column detection using heuristic rules
- 📊 Distribution analysis (histograms, density plots)
- 📦 Outlier detection (boxplots, statistical methods)
- 🔥 Correlation analysis and heatmap visualization
- 📉 Principal Component Analysis (PCA) with 2D/3D projections
- 📋 Comprehensive CSV report generation
- 🎯 Automatic task classification (classification vs. regression)

**Supported Datasets:**
Boston Housing, Parkinson's Disease, Sonar, Wine Quality (Red/White), Ozone, Concrete Compressive Strength, Breast Cancer Wisconsin, Auto MPG, Bank Marketing, Paddy

**Usage Examples:**

```python
from understanding_data import inspect_datasets, run_pca, detect_target_column, generate_visualizations

# Automated analysis of all datasets
inspect_datasets()

# Perform PCA analysis on specific dataset
components, explained_variance = run_pca(
    df, 
    target_col="quality", 
    dataset_name="Wine_Quality"
)

# Detect target column automatically
target = detect_target_column(df)

# Generate visualizations
generate_visualizations("Dataset_Name", X, y)
```

**Generated Outputs:**
- `understanding.csv` – Summary statistics for all datasets
- `plots/[Dataset_Name]/` – Organized visualizations per dataset
- `plots/[Dataset_Name]/pca/` – PCA analysis and scree plots

---

### 2. preparation_data.py – Data Preprocessing

Advanced data cleaning and imputation module utilizing machine learning-based missing value estimation.

**Core Functions:**

- 🧹 Structural validation (removal of empty columns/rows)
- 🤖 RF-based imputation for numerical features
- 🏷️ Mode-based imputation for categorical features
- 📊 Comprehensive preprocessing report
- 📈 Data quality assessment

**Usage:**

```python
from preparation_data import impute_missing_with_rf, clean_dataset

# Advanced RF-based imputation
df_imputed = impute_missing_with_rf(df)

# Basic cleaning with reporting
df_clean, cleaning_report = clean_dataset(df)
print(cleaning_report)
```

**Method Description:**

The imputation strategy employs RandomForest regressors/classifiers:
- For each column with missing values, the algorithm trains a model using complete data from other features
- Missing values are predicted using the trained model
- Both numerical and categorical features are handled appropriately

---

### 3. rlt_module.py – Core RLT Implementation

Base implementation of Reinforcement Learning Trees with support for both classification and regression tasks.

**Class: ReinforcementLearningTree (Classification)**

```python
from rlt_module import ReinforcementLearningTree

model = ReinforcementLearningTree(
    n_estimators=50,              # Number of trees in ensemble
    max_depth=8,                  # Maximum tree depth
    min_samples_split=10,         # Minimum samples for node split
    muting_threshold=0.15,        # Feature importance threshold
    embedded_model_depth=2,       # Depth of embedded Extra Trees
    linear_combination=2          # Exponent for importance weighting
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
```

**Class: ReinforcementLearningRegressor (Regression)**

```python
from rlt_module import ReinforcementLearningRegressor

model = ReinforcementLearningRegressor(
    n_estimators=50,
    max_depth=8,
    min_samples_split=10
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Algorithm Overview:**

1. **Tree Construction**: Bootstrap aggregation with feature selection via embedded Extra Trees models
2. **Feature Selection**: Dynamic candidate feature selection based on importance scores (muting threshold)
3. **Split Criterion**: Gini impurity (classification) or variance reduction (regression)
4. **Ensemble Aggregation**: Majority voting (classification) or mean prediction (regression)

---

### 4. Enhanced_rlt.py – Adaptive ML Pipeline

Production-grade pipeline providing automatic preprocessing, hyperparameter optimization, and inference.

**Core Capabilities:**
- 🎯 Automatic task detection (classification/regression)
- 🧠 Adaptive hyperparameter tuning based on dataset characteristics
- 🔄 End-to-end preprocessing (scaling, encoding, imputation)
- 💾 Model serialization and deserialization
- 📊 Multi-metric evaluation framework

**Adaptive Hyperparameter Tuning:**

```python
def adaptive_rlt_params(n_samples, n_features):
    """
    Dynamically adjust hyperparameters based on dataset size and dimensionality
    
    Args:
        n_samples: Number of training instances
        n_features: Number of input features
    
    Returns:
        Dictionary of tuned hyperparameters
    """
    return dict(
        n_estimators=min(100, max(20, n_samples // 20)),
        max_depth=min(12, max(4, int(np.log2(n_features + 1) * 2))),
        min_samples_split=max(2, n_samples // 100),
        muting_threshold=0.1 if n_samples > 1000 else 0.2,
        embedded_model_depth=2 if n_features < 20 else 3,
        linear_combination=min(3, n_features // 5)
    )
```

**Usage:**

```python
from Enhanced_rlt import train_and_save, load_and_predict, adaptive_rlt_params

# Train with adaptive parameters
metrics = train_and_save(
    df=df,
    target_col="target",
    model_path="./models/rlt_model.pkl",
    use_adaptive=True
)

print(f"Metrics: {metrics}")

# Load and perform inference
predictions = load_and_predict("./models/rlt_model.pkl", new_data)
```

**Evaluation Metrics:**

Classification:
- Accuracy, Precision, Recall, F1-Score (weighted)

Regression:
- RMSE, MAE, R² Score

---

### 5. rlt_training.py – Multi-Strategy Training Framework

Comprehensive training module for evaluating multiple RLT configurations and strategies.

**Implemented Strategies:**
- Standard RLT (baseline)
- RLT with 50% feature muting
- RLT with 80% feature muting
- RLT with linear combination (k=2, k=5)

**Usage:**

```bash
python rlt_training.py
```

**Output Structure:**
- Trained models saved in `rlt/` directory
- Performance metrics and comparisons
- Training time statistics

---

### 6. rlt_paper_results.py – Research Benchmarking Suite

Comprehensive research evaluation framework for academic publication, generating comparative results across multiple synthetic scenarios.

**Evaluation Scenarios:**

| Scenario | Type | Description |
|----------|------|-------------|
| Scenario 1 | Classification | Gaussian CDF-based classification with probability thresholding |
| Scenario 2 | Regression | Non-linear regression with interaction effects |
| Scenario 3 | Regression | Checkerboard pattern with correlated features |
| Scenario 4 | Regression | Linear model with varied coefficient magnitudes |

**Tested Dimensions:** p ∈ {200, 500, 1000} features

**Benchmark Methods:**
- Random Forest (n_estimators=500)
- Extremely Randomized Trees (n_estimators=500)
- RLT (naive, moderate, and aggressive configurations)

**Usage:**

```bash
python rlt_paper_results.py
```

**Generated Outputs:**
- `dso2/models/` – 1,440 trained model checkpoints
- `dso2/results/` – 3 CSV tables with comprehensive results
- Mean ± SD performance metrics across 10 repetitions

---

### 7. run_rlt.py – Standard Training Pipeline

Simplified training script for quick model development and prototyping.

**Usage:**

```bash
python run_rlt.py
```

---

### 8. rlt_streamlit_app/app.py – Interactive Web Application

Production-grade web interface for model training, evaluation, and inference using Streamlit framework.

**Application Features:**
- 📤 CSV dataset upload with automatic validation
- 🎯 Automatic target column selection
- 🚀 One-click model training
- 📊 Real-time metric visualization
- 🎯 Interactive manual prediction interface
- 📈 Probability distribution visualization
- 💾 Model persistence and loading

**Launch Instructions:**

```bash
cd rlt_streamlit_app
streamlit run app.py
```

Access the application at: `http://localhost:8501`

**Application Workflow:**

1. **Data Upload**: Select and upload CSV file
2. **Target Selection**: Choose prediction target
3. **Model Training**: Initiate training with adaptive parameters
4. **Prediction**: Input feature values for inference
5. **Results**: View predictions and confidence metrics

---

## Comprehensive Usage Examples

### Example 1: End-to-End Machine Learning Pipeline

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from understanding_data import detect_target_column
from preparation_data import impute_missing_with_rf
from Enhanced_rlt import train_and_save, load_and_predict

# Step 1: Data Loading
df = pd.read_csv("data/breast_cancer.csv")

# Step 2: Data Preprocessing
df_clean = impute_missing_with_rf(df)

# Step 3: Model Training
metrics = train_and_save(
    df=df_clean,
    target_col="diagnosis",
    model_path="./models/cancer_classifier.pkl",
    test_size=0.2,
    use_adaptive=True
)

print(f"Training Metrics:\n{metrics}")

# Step 4: Inference on New Data
new_samples = df_clean.drop(columns=["diagnosis"]).sample(10)
predictions = load_and_predict("./models/cancer_classifier.pkl", new_samples)

print(f"Predictions:\n{predictions}")
```

### Example 2: Comparative Model Evaluation

```python
from rlt_module import ReinforcementLearningTree, ReinforcementLearningRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import numpy as np

# Classification Task
rlt_clf = ReinforcementLearningTree(n_estimators=50, max_depth=8)
rf_clf = RandomForestClassifier(n_estimators=50, random_state=42)

rlt_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)

y_pred_rlt = rlt_clf.predict(X_test)
y_pred_rf = rf_clf.predict(X_test)

print(f"RLT Classification Accuracy: {accuracy_score(y_test, y_pred_rlt):.4f}")
print(f"RF Classification Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

# Regression Task
rlt_reg = ReinforcementLearningRegressor(n_estimators=50, max_depth=8)
rf_reg = RandomForestRegressor(n_estimators=50, random_state=42)

rlt_reg.fit(X_train, y_train)
rf_reg.fit(X_train, y_train)

y_pred_rlt = rlt_reg.predict(X_test)
y_pred_rf = rf_reg.predict(X_test)

print(f"\nRLT Regression R²: {r2_score(y_test, y_pred_rlt):.4f}")
print(f"RF Regression R²: {r2_score(y_test, y_pred_rf):.4f}")
print(f"\nRLT RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rlt)):.4f}")
print(f"RF RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.4f}")
```

### Example 3: Advanced Exploratory Data Analysis

```python
from understanding_data import (
    inspect_datasets,
    run_pca,
    generate_visualizations,
    display_dataset_plots,
    clean_dataset
)
import pandas as pd

# Comprehensive dataset analysis
inspect_datasets()

# Load specific dataset
df = pd.read_csv("data/breast_cancer.csv")

# Clean and visualize
df_clean, report = clean_dataset(df)
print(f"Cleaning Report:\n{report}")

# Generate visualizations
generate_visualizations("Breast_Cancer", df_clean.drop("diagnosis", axis=1), df_clean["diagnosis"])

# Perform dimensionality reduction
components, explained_variance = run_pca(
    df_clean,
    target_col="diagnosis",
    dataset_name="Breast_Cancer"
)

# Display results
display_dataset_plots("Breast_Cancer")
```

---

## Evaluation Metrics

### Classification Tasks
- **Accuracy**: Overall fraction of correct predictions
- **Precision (weighted)**: Precision averaged by class support
- **Recall (weighted)**: Recall averaged by class support
- **F1-Score (weighted)**: Harmonic mean of precision and recall

### Regression Tasks
- **RMSE**: Root mean squared error measuring average prediction deviation
- **MAE**: Mean absolute error for robust error measurement
- **R² Score**: Coefficient of determination (0-1 scale)

---

## Technical Specifications

### System Requirements
- **Minimum**: Python 3.8, 2GB RAM
- **Recommended**: Python 3.10+, 4GB RAM, SSD storage
- **Processing**: Multi-core CPU recommended for parallel training

### Computational Complexity
- **Training**: O(n·m·log(m)·d) where n=samples, m=features, d=depth
- **Inference**: O(log(d)) per sample
- **Memory**: O(n·m) for data + O(d·m) for model storage

---

## Visualization Capabilities

Generated visualizations per dataset:

- **Distribution Analysis**: Histograms and density plots for feature characterization
- **Outlier Detection**: Boxplots identifying extreme values
- **Feature Relationships**: Correlation heatmaps and scatter plots
- **Dimensionality Reduction**: PCA projections (2D/3D) with variance explained
- **Scree Plot**: Cumulative explained variance for component selection

---

## Configuration & Customization

### Adding New Datasets

Modify `DATASET_CATALOG` in `understanding_data.py`:

```python
DATASET_CATALOG = {
    "Dataset_Name": {"file": "dataset_file.csv"},
    "Excel_Dataset": {"file": "dataset_file.xlsx"},
}
```

### Custom Hyperparameter Configuration

```python
# Option 1: Fixed Parameters
model = ReinforcementLearningTree(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    muting_threshold=0.2
)

# Option 2: Adaptive Parameters
from Enhanced_rlt import adaptive_rlt_params
params = adaptive_rlt_params(n_samples=1000, n_features=50)
model = ReinforcementLearningTree(**params)
```

---

## Important Considerations

- ✅ Data files must be in CSV or Excel format
- ✅ Directories (`plots/`, `models/`, `data/`) are created automatically
- ✅ RF imputation requires minimum 5 samples per column
- ✅ Streamlit deployment requires `streamlit>=1.0`
- ✅ For datasets >100K rows, use parallel processing and consider distributed training
- ✅ Categorical features are automatically encoded using LabelEncoder
- ✅ Numerical features are standardized using StandardScaler

---

## Troubleshooting Guide

| Issue | Solution |
|-------|----------|
| `ImportError: rlt_module` | Verify all `.py` files are in the same directory |
| Out of Memory | Reduce `n_estimators`, use smaller dataset, or increase system RAM |
| Streamlit module not found | Navigate to `rlt_streamlit_app/` and execute from that directory |
| NaN in predictions | Apply `impute_missing_with_rf()` before model training |
| Slow training | Reduce max_depth, features, or n_estimators; enable parallel processing |
| Inconsistent results | Set random seeds: `np.random.seed(42)` in preprocessing |

---

## Dependencies

See `requirements.txt` for comprehensive dependency list and versions.

**Core Stack:**
- **Data Processing**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn, Pillow
- **Web Framework**: streamlit, altair
- **Utilities**: IPython, openpyxl, joblib

---

## Research & Academic Applications

This framework is designed for:
- Comparative ML algorithm evaluation
- Hyperparameter tuning studies
- Feature importance analysis
- Model benchmarking across scenarios
- Publication-ready result generation

For research applications, utilize:
- `rlt_paper_results.py` for benchmark comparisons
- Scenario-based evaluation in synthetic data
- Statistical significance testing
- Performance table generation

---

## References & Documentation

- **scikit-learn Documentation**: https://scikit-learn.org/stable/documentation.html
- **Streamlit Documentation**: https://docs.streamlit.io/
- **Pandas User Guide**: https://pandas.pydata.org/docs/user_guide/
- **NumPy Documentation**: https://numpy.org/doc/
- **Matplotlib Tutorial**: https://matplotlib.org/stable/tutorials/

---

## Citation & Attribution

For academic use, please cite this work as:

```bibtex
@software{rlt_pipeline_2025,
  title={Reinforcement Learning Tree: Comprehensive Machine Learning Pipeline},
  author={Project Contributors},
  year={2025},
  url={https://github.com/Nexus_v1.1/rlt-project},
  version={2.0}
}
```

---
 

---

**Document Version**: 2.0 (Production Release)  
**Last Updated**: December 2025  
**Status**: Production Ready ✅  
**License**: Academic   
