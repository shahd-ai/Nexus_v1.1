# rlt_module.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Literal, Optional
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib  # for saving/loading models

# Create model folder if it doesn't exist
os.makedirs("model", exist_ok=True)

# ===============================
# REINFORCEMENT LEARNING TREE
# ===============================
class ReinforcementLearningTree:
    """
    Reinforcement Learning Tree with configurable strategies
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        task: Literal['regression', 'classification'] = 'regression',
        embedded_model: Literal['rf', 'et'] = 'rf',
        use_variable_muting: bool = False,
        muting_rate: float = 0.5,
        use_linear_combination: bool = False,
        k_linear: int = 1,
        alpha: float = 0.25,
        p_protected: Optional[int] = None,
        min_samples_split: int = 10,
        max_depth: Optional[int] = None,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.task = task
        self.embedded_model = embedded_model
        self.use_variable_muting = use_variable_muting
        self.muting_rate = muting_rate
        self.use_linear_combination = use_linear_combination
        self.k_linear = k_linear
        self.alpha = alpha
        self.p_protected = p_protected
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees_ = []
        self.feature_importances_ = None
        self.n_features_ = None

    # --- embedded model selection ---
    def _get_embedded_model(self, n_features: int):
        n_trees = min(100, max(50, n_features // 5))
        if self.task == 'regression':
            if self.embedded_model == 'rf':
                return RandomForestRegressor(
                    n_estimators=n_trees, max_features='sqrt', bootstrap=True,
                    oob_score=True, random_state=self.random_state, n_jobs=-1
                )
            else:
                return ExtraTreesRegressor(
                    n_estimators=n_trees, max_features='sqrt', bootstrap=True,
                    oob_score=True, random_state=self.random_state, n_jobs=-1
                )
        else:
            if self.embedded_model == 'rf':
                return RandomForestClassifier(
                    n_estimators=n_trees, max_features='sqrt', bootstrap=True,
                    oob_score=True, random_state=self.random_state, n_jobs=-1
                )
            else:
                return ExtraTreesClassifier(
                    n_estimators=n_trees, max_features='sqrt', bootstrap=True,
                    oob_score=True, random_state=self.random_state, n_jobs=-1
                )

    # --- variable importance ---
    def _calculate_variable_importance(self, X, y, active_features):
        if len(active_features) == 0:
            return np.array([])
        X_active = X[:, active_features]
        n_samples = X_active.shape[0]
        boot_idx = np.random.choice(n_samples, n_samples, replace=True)
        oob_idx = np.array([i for i in range(n_samples) if i not in boot_idx])
        if len(oob_idx) < 5:
            return np.zeros(len(active_features))
        embedded = self._get_embedded_model(len(active_features))
        embedded.fit(X_active[boot_idx], y[boot_idx])
        y_pred = embedded.predict(X_active[oob_idx])
        baseline_error = mean_squared_error(y[oob_idx], y_pred) if self.task == 'regression' else 1 - accuracy_score(y[oob_idx], y_pred)
        importances = np.zeros(len(active_features))
        for i, feat_idx in enumerate(active_features):
            X_perm = X_active[oob_idx].copy()
            np.random.shuffle(X_perm[:, i])
            y_pred_perm = embedded.predict(X_perm)
            perm_error = mean_squared_error(y[oob_idx], y_pred_perm) if self.task == 'regression' else 1 - accuracy_score(y[oob_idx], y_pred_perm)
            importances[i] = (perm_error / baseline_error) - 1 if baseline_error > 0 else 0
        return np.maximum(importances, 0)

    # --- select splitting variable ---
    def _select_splitting_variables(self, X, y, active_features, protected_features):
        if len(active_features) == 0:
            return [], []
        vi = self._calculate_variable_importance(X, y, active_features)
        if np.sum(vi) == 0:
            idx = np.random.choice(len(active_features))
            return [active_features[idx]], [1.0]
        sorted_idx = np.argsort(vi)[::-1]
        if not self.use_linear_combination or self.k_linear == 1:
            return [active_features[sorted_idx[0]]], [1.0]
        max_vi = vi[sorted_idx[0]]
        threshold = self.alpha * max_vi
        selected, coefs = [], []
        for idx in sorted_idx[:self.k_linear]:
            if vi[idx] >= threshold and vi[idx] > 0:
                feat_idx = active_features[idx]
                selected.append(feat_idx)
                corr = np.corrcoef(X[:, feat_idx], y)[0, 1]
                sign = np.sign(corr) if not np.isnan(corr) else 1.0
                coefs.append(sign * np.sqrt(vi[idx]))
        if len(selected) == 0:
            return [active_features[sorted_idx[0]]], [1.0]
        return selected, coefs

    # --- tree building ---
    def _build_tree(self, X, y, active_features, protected_features, depth=0):
        n_samples, n_features = X.shape
        if n_samples < self.min_samples_split or len(active_features) == 0 or (self.max_depth and depth >= self.max_depth):
            return {'type': 'leaf', 'value': np.mean(y)}
        split_features, coefficients = self._select_splitting_variables(X, y, active_features, protected_features)
        if len(split_features) == 0:
            return {'type': 'leaf', 'value': np.mean(y)}
        split_values = X[:, split_features[0]] if len(split_features) == 1 else sum(coef * X[:, feat] for feat, coef in zip(split_features, coefficients))
        low_val, high_val = np.quantile(split_values, 0.25), np.quantile(split_values, 0.75)
        if low_val == high_val:
            return {'type': 'leaf', 'value': np.mean(y)}
        split_point = np.random.uniform(low_val, high_val)
        left_mask = split_values <= split_point
        if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0:
            return {'type': 'leaf', 'value': np.mean(y)}
        new_protected = set(protected_features).union(set(split_features))
        left_child = self._build_tree(X[left_mask], y[left_mask], active_features, new_protected, depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], active_features, new_protected, depth + 1)
        return {'type': 'node', 'features': split_features, 'coefficients': coefficients, 'split_point': split_point, 'left': left_child, 'right': right_child}

    # --- predict tree ---
    def _predict_tree(self, tree, X):
        if tree['type'] == 'leaf':
            return np.full(X.shape[0], tree['value'])
        split_values = X[:, tree['features'][0]] if len(tree['features']) == 1 else sum(coef * X[:, feat] for feat, coef in zip(tree['features'], tree['coefficients']))

        left_mask = split_values <= tree['split_point']
        preds = np.zeros(X.shape[0])
        preds[left_mask] = self._predict_tree(tree['left'], X[left_mask])
        preds[~left_mask] = self._predict_tree(tree['right'], X[~left_mask])
        return preds

    # --- fit ---
    def fit(self, X, y, save_name: Optional[str] = None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        self.n_features_ = X.shape[1]
        if self.p_protected is None:
            self.p_protected = max(1, int(np.log(self.n_features_)))
        self.trees_ = []
        all_importances = np.zeros(self.n_features_)
        for i in range(self.n_estimators):
            boot_idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
            tree = self._build_tree(X[boot_idx], y[boot_idx], list(range(self.n_features_)), set())
            self.trees_.append(tree)
            all_importances[:self.n_features_] += self._calculate_variable_importance(X[boot_idx], y[boot_idx], list(range(self.n_features_)))
        self.feature_importances_ = all_importances / self.n_estimators
        if save_name:
            joblib.dump(self, f"model/{save_name}.pkl")
            print(f"Model saved as model/{save_name}.pkl")
        return self

    # --- predict ---
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        preds = np.zeros((X.shape[0], len(self.trees_)))
        for i, tree in enumerate(self.trees_):
            preds[:, i] = self._predict_tree(tree, X)
        return np.mean(preds, axis=1) if self.task == 'regression' else (np.mean(preds, axis=1) > 0.5).astype(int)
