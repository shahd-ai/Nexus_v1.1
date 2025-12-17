# ============================================================
# RLT TRAINING — MULTI-STRATEGY (REGRESSION & CLASSIFICATION)
# ============================================================

import os
import joblib
import numpy as np
from rlt_module import ReinforcementLearningTree, ReinforcementLearningRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from scenarios import scenario2  # or your dataset generator

# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = "rlt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N = 1000
p = 200
RANDOM_STATE = 42

# ============================================================
# DATASET (Example)
# ============================================================

# Regression dataset
X_train_reg, y_train_reg, X_test_reg, y_test_reg = scenario2(
    N=N, p=p, test_size=500, seed=RANDOM_STATE, save=False
)

# Classification dataset (convert y to integers for classes)
X_train_cla, y_train_cla, X_test_cla, y_test_cla = scenario2(
    N=N, p=p, test_size=500, seed=RANDOM_STATE, save=False
)
y_train_cla = (y_train_cla > np.median(y_train_cla)).astype(int)
y_test_cla  = (y_test_cla > np.median(y_test_cla)).astype(int)

print("Datasets loaded")
print("  Regression train:", X_train_reg.shape)
print("  Classification train:", X_train_cla.shape)

# ============================================================
# UTILS
# ============================================================

def enable_fast_vi(model, p):
    if p >= 200:
        model.fast_vi = True

def train_and_save(model, name, X_train, y_train, X_test, y_test, task="regression"):
    print(f"\nTraining {name} ({task})...")
    enable_fast_vi(model, X_train.shape[1])
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    if task == "regression":
        score = mean_squared_error(y_test, y_pred)
        print(f"  MSE: {score:.4f}")
    else:
        score = accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {score:.4f}")
    
    # Save model
    joblib.dump(model, os.path.join(OUTPUT_DIR, f"{name}_{task}.joblib"))
    return score

# ============================================================
# CONFIGURE RLT STRATEGIES
# ============================================================

rlt_strategies = [
    {"name": "RLT_standard", "muting": False, "linear": False}  ,
    {"name": "RLT_muting_50", "muting": True, "muting_rate": 0.5, "linear": False},
    {"name": "RLT_muting_80", "muting": True, "muting_rate": 0.8, "linear": False},
    {"name": "RLT_linear_k2", "muting": True, "muting_rate": 0.8, "linear": True, "k_linear": 2},
    {"name": "RLT_linear_k5", "muting": True, "muting_rate": 0.8, "linear": True, "k_linear": 5},
]

# ============================================================
# TRAIN REGRESSION MODELS
# ============================================================

results_reg = {}
for cfg in rlt_strategies:
    model = ReinforcementLearningRegressor(
        n_estimators=50,
        max_depth=5,
        muting_threshold=cfg.get("muting_rate", 0.0),
        embedded_model_depth=1
    )
    score = train_and_save(
        model, cfg["name"], X_train_reg, y_train_reg, X_test_reg, y_test_reg, task="regression"
    )
    results_reg[cfg["name"]] = score

# ============================================================
# TRAIN CLASSIFICATION MODELS
# ============================================================

results_cla = {}
for cfg in rlt_strategies:
    model = ReinforcementLearningTree(
        n_estimators=50,
        max_depth=5,
        muting_threshold=cfg.get("muting_rate", 0.0),
        embedded_model_depth=1
    )
    score = train_and_save(
        model, cfg["name"], X_train_cla, y_train_cla, X_test_cla, y_test_cla, task="classification"
    )
    results_cla[cfg["name"]] = score

# ============================================================
# SUMMARY
# ============================================================

print("\n================ REGRESSION MODEL COMPARISON =================")
for name, score in sorted(results_reg.items(), key=lambda x: x[1]):
    print(f"{name:<20s} : MSE={score:.4f}")

print("\n================ CLASSIFICATION MODEL COMPARISON =================")
for name, score in sorted(results_cla.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:<20s} : Accuracy={score:.4f}")

print("\nAll models saved in:", OUTPUT_DIR)
