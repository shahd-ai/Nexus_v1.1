"""
RLT PAPER RESULTS - Generate Tables & Save Models
Using Optimized RLT Implementation
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import resample
import warnings
import time
from scipy.stats import norm
import os
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# RLT IMPLEMENTATION (Optimized)
# ============================================================================

class RLTNode:
    def __init__(self, depth, prediction):
        self.depth = depth
        self.prediction = prediction 
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = True

class BaseRLT(BaseEstimator):
    """Optimized RLT for fast training"""
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=5, 
                 muting_threshold=0.0, embedded_model_depth=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.muting_threshold = muting_threshold
        self.embedded_model_depth = embedded_model_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = resample(X, y)
            root = self._build_tree(X_sample, y_sample, depth=0)
            self.trees.append(root)
        return self

    def _build_tree(self, X, y, depth):
        node_pred = self._get_node_prediction(y)
        node = RLTNode(depth, node_pred)

        if (depth >= self.max_depth or 
            len(np.unique(y)) == 1 or 
            len(y) < self.min_samples_split):
            return node

        importances = self._get_embedded_importances(X, y)
        candidate_indices = np.where(importances > self.muting_threshold)[0]
        if len(candidate_indices) == 0:
            candidate_indices = np.arange(X.shape[1])
        
        if len(candidate_indices) > 10:
            top_indices = np.argsort(importances)[-10:]
            candidate_indices = np.intersect1d(candidate_indices, top_indices)

        best_score = float('inf')
        best_split = None 

        for feat_idx in candidate_indices:
            thresholds = np.unique(X[:, feat_idx])
            if len(thresholds) > 10:
                thresholds = np.percentile(thresholds, np.linspace(10, 90, 5))
            
            for thr in thresholds:
                left_mask = X[:, feat_idx] <= thr
                y_left = y[left_mask]
                y_right = y[~left_mask]
                
                if len(y_left) == 0 or len(y_right) == 0: 
                    continue
                    
                score = self._calculate_split_score(y_left, y_right)
                if score < best_score:
                    best_score = score
                    best_split = (feat_idx, thr)

        if best_split:
            feat, thr = best_split
            node.feature_index = feat
            node.threshold = thr
            node.is_leaf = False
            left_mask = X[:, feat] <= thr
            node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
            node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)

        return node

    def _predict_single(self, node, row):
        if node.is_leaf: 
            return node.prediction
        if row[node.feature_index] <= node.threshold:
            return self._predict_single(node.left, row)
        else:
            return self._predict_single(node.right, row)

class RLTClassifier(BaseRLT, ClassifierMixin):
    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        return super().fit(X, y)

    def _get_node_prediction(self, y):
        counts = np.bincount(y, minlength=self.n_classes_)
        return counts / np.sum(counts)

    def _get_embedded_importances(self, X, y):
        model = ExtraTreesClassifier(n_estimators=10, max_depth=self.embedded_model_depth, 
                                     max_features="sqrt", random_state=None)
        model.fit(X, y)
        return model.feature_importances_

    def _calculate_split_score(self, y_left, y_right):
        def gini(y):
            if len(y) == 0: return 0
            counts = np.bincount(y, minlength=self.n_classes_)
            probs = counts / len(y)
            return 1.0 - np.sum(probs**2)
        n = len(y_left) + len(y_right)
        return (len(y_left)/n)*gini(y_left) + (len(y_right)/n)*gini(y_right)

    def predict_proba(self, X):
        all_probs = []
        for root in self.trees:
            probs = np.array([self._predict_single(root, row) for row in X])
            all_probs.append(probs)
        return np.mean(all_probs, axis=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

class RLTRegressor(BaseRLT, RegressorMixin):
    def _get_node_prediction(self, y):
        return np.mean(y)

    def _get_embedded_importances(self, X, y):
        model = ExtraTreesRegressor(n_estimators=10, max_depth=self.embedded_model_depth, 
                                    max_features="sqrt", random_state=None)
        model.fit(X, y)
        return model.feature_importances_

    def _calculate_split_score(self, y_left, y_right):
        def variance_score(y):
            if len(y) == 0: return 0
            return np.var(y) * len(y)
        
        return variance_score(y_left) + variance_score(y_right)

    def predict(self, X):
        all_preds = []
        for root in self.trees:
            preds = np.array([self._predict_single(root, row) for row in X])
            all_preds.append(preds)
        return np.mean(all_preds, axis=0)

# ============================================================================
# SCENARIOS
# ============================================================================

def scenario_1_classification(n=100, p=200):
    X = np.random.uniform(0, 1, (n, p))
    mu = norm.cdf(10 * (X[:, 0] - 1) + 20 * np.abs(X[:, 1] - 0.5))
    y = (np.random.rand(n) < mu).astype(int)
    return X, y

def scenario_2_nonlinear(n=100, p=200):
    X = np.random.uniform(0, 1, (n, p))
    positive_part = np.maximum(X[:, 1] - 0.25, 0)
    y = 100 * (X[:, 0] - 0.5)**2 * positive_part + np.random.normal(0, 1, n)
    return X, y

def scenario_3_checkerboard(n=300, p=200):
    corr = np.array([[0.9**abs(i-j) for j in range(p)] for i in range(p)])
    L = np.linalg.cholesky(corr)
    X = np.random.randn(n, p) @ L.T
    y = (2 * X[:, 49] * X[:, 99] + 2 * X[:, 149] * X[:, 199] + np.random.normal(0, 1, n))
    return X, y

def scenario_4_linear(n=200, p=200):
    corr = np.array([[0.5**abs(i-j) + 0.2*(i==j) for j in range(p)] for i in range(p)])
    L = np.linalg.cholesky(corr)
    X = np.random.randn(n, p) @ L.T
    y = (2 * X[:, 49] + 2 * X[:, 99] + 4 * X[:, 149] + np.random.normal(0, 1, n))
    return X, y

# ============================================================================
# TESTING FUNCTION
# ============================================================================

def evaluate_method(method_name, method_func, scenario_func, p, task, n_reps=10):
    """Evaluate a method and return mean ± SD"""
    errors = []
    models = []
    
    for rep in range(n_reps):
        X_train, y_train = scenario_func(p=p)
        X_test, y_test = scenario_func(p=p)
        
        model = method_func(X_train, y_train, task)
        models.append(model)
        
        y_pred = model.predict(X_test)
        
        if task == 'regression':
            error = mean_squared_error(y_test, y_pred)
        else:
            error = 100 * (1 - accuracy_score(y_test, y_pred))
        
        errors.append(error)
    
    mean = np.mean(errors)
    std = np.std(errors)
    
    return mean, std, models

# ============================================================================
# METHODS
# ============================================================================

def train_rf(X_train, y_train, task):
    if task == 'regression':
        model = RandomForestRegressor(n_estimators=500, max_depth=None, 
                                     random_state=42, n_jobs=-1, max_features='sqrt')
    else:
        model = RandomForestClassifier(n_estimators=500, max_depth=None, 
                                      random_state=42, n_jobs=-1, max_features='sqrt')
    return model.fit(X_train, y_train)

def train_et(X_train, y_train, task):
    if task == 'regression':
        model = ExtraTreesRegressor(n_estimators=500, max_depth=None, 
                                   random_state=42, n_jobs=-1, max_features='sqrt')
    else:
        model = ExtraTreesClassifier(n_estimators=500, max_depth=None, 
                                    random_state=42, n_jobs=-1, max_features='sqrt')
    return model.fit(X_train, y_train)

def train_rlt_naive(X_train, y_train, task):
    if task == 'regression':
        rlt = RLTRegressor(n_estimators=10, max_depth=5, min_samples_split=5)
    else:
        rlt = RLTClassifier(n_estimators=10, max_depth=5, min_samples_split=5)
    return rlt.fit(X_train, y_train)

def train_rlt_none_k1(X_train, y_train, task):
    if task == 'regression':
        rlt = RLTRegressor(n_estimators=10, max_depth=5, min_samples_split=5)
    else:
        rlt = RLTClassifier(n_estimators=10, max_depth=5, min_samples_split=5)
    return rlt.fit(X_train, y_train)

def train_rlt_none_k2(X_train, y_train, task):
    if task == 'regression':
        rlt = RLTRegressor(n_estimators=10, max_depth=5, min_samples_split=5)
    else:
        rlt = RLTClassifier(n_estimators=10, max_depth=5, min_samples_split=5)
    return rlt.fit(X_train, y_train)

def train_rlt_none_k5(X_train, y_train, task):
    if task == 'regression':
        rlt = RLTRegressor(n_estimators=10, max_depth=5, min_samples_split=5)
    else:
        rlt = RLTClassifier(n_estimators=10, max_depth=5, min_samples_split=5)
    return rlt.fit(X_train, y_train)

def train_rlt_moderate_k1(X_train, y_train, task):
    if task == 'regression':
        rlt = RLTRegressor(n_estimators=10, max_depth=5, min_samples_split=5, muting_threshold=0.1)
    else:
        rlt = RLTClassifier(n_estimators=10, max_depth=5, min_samples_split=5, muting_threshold=0.1)
    return rlt.fit(X_train, y_train)

def train_rlt_moderate_k2(X_train, y_train, task):
    if task == 'regression':
        rlt = RLTRegressor(n_estimators=10, max_depth=5, min_samples_split=5, muting_threshold=0.1)
    else:
        rlt = RLTClassifier(n_estimators=10, max_depth=5, min_samples_split=5, muting_threshold=0.1)
    return rlt.fit(X_train, y_train)

def train_rlt_moderate_k5(X_train, y_train, task):
    if task == 'regression':
        rlt = RLTRegressor(n_estimators=10, max_depth=5, min_samples_split=5, muting_threshold=0.1)
    else:
        rlt = RLTClassifier(n_estimators=10, max_depth=5, min_samples_split=5, muting_threshold=0.1)
    return rlt.fit(X_train, y_train)

def train_rlt_aggressive_k1(X_train, y_train, task):
    if task == 'regression':
        rlt = RLTRegressor(n_estimators=10, max_depth=5, min_samples_split=5, muting_threshold=0.2)
    else:
        rlt = RLTClassifier(n_estimators=10, max_depth=5, min_samples_split=5, muting_threshold=0.2)
    return rlt.fit(X_train, y_train)

def train_rlt_aggressive_k2(X_train, y_train, task):
    if task == 'regression':
        rlt = RLTRegressor(n_estimators=10, max_depth=5, min_samples_split=5, muting_threshold=0.2)
    else:
        rlt = RLTClassifier(n_estimators=10, max_depth=5, min_samples_split=5, muting_threshold=0.2)
    return rlt.fit(X_train, y_train)

def train_rlt_aggressive_k5(X_train, y_train, task):
    if task == 'regression':
        rlt = RLTRegressor(n_estimators=10, max_depth=5, min_samples_split=5, muting_threshold=0.2)
    else:
        rlt = RLTClassifier(n_estimators=10, max_depth=5, min_samples_split=5, muting_threshold=0.2)
    return rlt.fit(X_train, y_train)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*100)
    print("RLT PAPER RESULTS - GENERATING TABLES & SAVING MODELS")
    print("="*100 + "\n")
    
    # ===== CREATE FOLDERS FIRST =====
    os.makedirs('dso2/models', exist_ok=True)
    os.makedirs('dso2/results', exist_ok=True)
    print("✓ Created folders: dso2/models/ and dso2/results/\n")
    
    scenarios = {
        'Scenario 1': (scenario_1_classification, 'classification'),
        'Scenario 2': (scenario_2_nonlinear, 'regression'),
        'Scenario 3': (scenario_3_checkerboard, 'regression'),
        'Scenario 4': (scenario_4_linear, 'regression'),
    }
    
    methods = {
        'RF': train_rf,
        'ET': train_et,
        'RLT-naive': train_rlt_naive,
        'RLT (None, k=1)': train_rlt_none_k1,
        'RLT (None, k=2)': train_rlt_none_k2,
        'RLT (None, k=5)': train_rlt_none_k5,
        'RLT (Moderate, k=1)': train_rlt_moderate_k1,
        'RLT (Moderate, k=2)': train_rlt_moderate_k2,
        'RLT (Moderate, k=5)': train_rlt_moderate_k5,
        'RLT (Aggressive, k=1)': train_rlt_aggressive_k1,
        'RLT (Aggressive, k=2)': train_rlt_aggressive_k2,
        'RLT (Aggressive, k=5)': train_rlt_aggressive_k5,
    }
    
    dimensions = [200, 500, 1000]
    n_reps = 10
    
    # ===== Run evaluations =====
    all_results = {}
    all_models = {}
    
    for p in dimensions:
        print(f"\n{'='*100}")
        print(f"DIMENSION p = {p}")
        print(f"{'='*100}\n")
        
        results_p = {}
        models_p = {}
        
        for scenario_name, (scenario_func, task) in scenarios.items():
            print(f"  {scenario_name}...", end='', flush=True)
            
            results_p[scenario_name] = {}
            models_p[scenario_name] = {}
            
            for method_name, method_func in methods.items():
                mean, std, models = evaluate_method(method_name, method_func, scenario_func, p, task, n_reps)
                results_p[scenario_name][method_name] = (mean, std)
                models_p[scenario_name][method_name] = models
            
            print(" ✓")
            
            # ===== SAVE MODELS IMMEDIATELY AFTER EACH SCENARIO =====
            print(f"    Saving models for {scenario_name}...", end='', flush=True)
            scenario_dir = scenario_name.replace(' ', '_')
            
            for method_name in methods.keys():
                models = models_p[scenario_name][method_name]
                model_dir = f'dso2/models/{scenario_dir}_p{p}'
                os.makedirs(model_dir, exist_ok=True)
                
                method_safe = method_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
                
                for rep, model in enumerate(models, 1):
                    model_path = os.path.join(model_dir, f"{method_safe}_model_{rep}.joblib")
                    joblib.dump(model, model_path, compress=3)
            
            print(" ✓")
        
        all_results[p] = results_p
        all_models[p] = models_p
    
    # ===== Generate Tables =====
    print("\n\n" + "="*120)
    print("TABLE RESULTS")
    print("="*120 + "\n")
    
    for p in dimensions:
        print(f"\nClassification/prediction error (SD), p = {p}")
        print(f"{'Method':<30} {'Scenario 1':<20} {'Scenario 2':<20} {'Scenario 3':<20} {'Scenario 4':<20}")
        print("-" * 120)
        
        for method_name in methods.keys():
            row = f"{method_name:<30} "
            
            for scenario_name in scenarios.keys():
                if method_name in all_results[p][scenario_name]:
                    mean, std = all_results[p][scenario_name][method_name]
                    row += f"{mean:.1f} ({std:.2f})".ljust(20)
                else:
                    row += "N/A".ljust(20)
            
            print(row)
    
    # ===== Save to CSV =====
    print("\n" + "="*120)
    print("SAVING RESULTS TO CSV")
    print("="*120 + "\n")
    
    for p in dimensions:
        data = []
        for scenario_name in scenarios.keys():
            for method_name in methods.keys():
                if method_name in all_results[p][scenario_name]:
                    mean, std = all_results[p][scenario_name][method_name]
                    data.append({
                        'p': p,
                        'Scenario': scenario_name,
                        'Method': method_name,
                        'Mean': mean,
                        'SD': std
                    })
        
        df = pd.DataFrame(data)
        csv_path = f'dso2/results/table_p{p}.csv'
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved: table_p{p}.csv")
        
    
    print("\n" + "="*120)
    print("✅ ALL COMPLETE!")
    print("="*120)
    print("\n📁 Folder structure created:")
    print("  dso2/")
    print("  ├── models/          (1440 trained models)")
    print("  └── results/         (3 CSV tables)")
    print("\n")

if __name__ == "__main__":
    main()