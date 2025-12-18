import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, log_loss
import warnings
warnings.filterwarnings('ignore')

class ReinforcementLearningTree:
    """
    Reinforcement Learning Tree with configurable strategies
    PRODUCTION READY - Works for BOTH regression AND classification tasks
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        task: str = 'regression',
        embedded_model: str = 'et',
        use_variable_muting: bool = False,
        muting_rate: float = 0.5,
        use_linear_combination: bool = False,
        k_linear: int = 1,
        alpha: float = 0.25,
        p_protected: int = None,
        min_samples_split: int = 10,
        max_depth: int = None,
        random_state: int = 42
    ):
        assert task in ['regression', 'classification'], "task must be 'regression' or 'classification'"
        assert embedded_model in ['rf', 'et'], "embedded_model must be 'rf' or 'et'"
        
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
        np.random.seed(random_state)
        
    def _get_embedded_model(self, n_features: int):
        """Create embedded model for variable importance calculation"""
        n_trees = 30  # REDUCED for speed
        
        if self.task == 'regression':
            if self.embedded_model == 'rf':
                return RandomForestRegressor(
                    n_estimators=n_trees,
                    max_features='sqrt',
                    bootstrap=True,
                    oob_score=True,
                    random_state=self.random_state,
                    n_jobs=-1,
                    max_depth=10
                )
            else:  # ET
                return ExtraTreesRegressor(
                    n_estimators=n_trees,
                    max_features='sqrt',
                    bootstrap=False,
                    random_state=self.random_state,
                    n_jobs=-1,
                    max_depth=10
                )
        else:  # CLASSIFICATION
            if self.embedded_model == 'rf':
                return RandomForestClassifier(
                    n_estimators=n_trees,
                    max_features='sqrt',
                    bootstrap=True,
                    oob_score=True,
                    random_state=self.random_state,
                    n_jobs=-1,
                    max_depth=10
                )
            else:  # ET
                return ExtraTreesClassifier(
                    n_estimators=n_trees,
                    max_features='sqrt',
                    bootstrap=False,
                    random_state=self.random_state,
                    n_jobs=-1,
                    max_depth=10
                )
    
    def _calculate_variable_importance(self, X, y, active_features):
        """
        Calculate variable importance using permutation importance
        ROBUST - handles all edge cases
        """
        if len(active_features) == 0:
            return np.array([])
        
        X_active = X[:, active_features]
        n = X_active.shape[0]
        
        # Check minimum samples
        if n < self.min_samples_split:
            return np.zeros(len(active_features))
        
        # Check if classification has at least 2 classes
        if self.task == 'classification':
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                return np.zeros(len(active_features))
        
        try:
            embedded = self._get_embedded_model(len(active_features))
            
            # ===== RF: OOB Permutation Importance =====
            if self.embedded_model == "rf":
                boot_idx = np.random.choice(n, n, replace=True)
                oob_idx = np.setdiff1d(np.arange(n), boot_idx)
                
                if len(oob_idx) < 10:
                    return np.zeros(len(active_features))
                
                embedded.fit(X_active[boot_idx], y[boot_idx])
                eval_X = X_active[oob_idx]
                eval_y = y[oob_idx]
            
            # ===== ET: Hold-out Validation =====
            else:
                # FIX: Check if stratify is possible
                if self.task == 'classification':
                    unique, counts = np.unique(y, return_counts=True)
                    # Can only stratify if all classes have >= 2 samples in test set
                    min_samples_class = np.min(counts)
                    can_stratify = min_samples_class >= 3  # Need at least 3 for 25% split
                    
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X_active, y, test_size=0.25, random_state=42,
                        stratify=y if can_stratify else None
                    )
                else:
                    X_tr, X_val, y_tr, y_val = train_test_split(
                        X_active, y, test_size=0.25, random_state=42
                    )
                
                embedded.fit(X_tr, y_tr)
                eval_X = X_val
                eval_y = y_val
            
            # ===== Baseline Error Calculation =====
            if self.task == 'regression':
                baseline = mean_squared_error(eval_y, embedded.predict(eval_X))
            else:  # CLASSIFICATION
                unique_eval = np.unique(eval_y)
                if len(unique_eval) < 2:
                    return np.zeros(len(active_features))
                
                y_pred_proba = embedded.predict_proba(eval_X)
                if y_pred_proba.shape[1] == 2:
                    baseline = log_loss(eval_y, y_pred_proba[:, 1])
                else:
                    baseline = log_loss(eval_y, y_pred_proba)
            
            baseline = max(baseline, 1e-8)
            
            # ===== Permutation Importance =====
            importances = np.zeros(len(active_features))
            max_perm = min(len(active_features), 15)  # Reduced from 20
            perm_indices = np.random.choice(
                len(active_features), size=max_perm, replace=False
            )
            
            for i in perm_indices:
                try:
                    X_perm = eval_X.copy()
                    np.random.shuffle(X_perm[:, i])
                    
                    if self.task == 'regression':
                        perm_error = mean_squared_error(eval_y, embedded.predict(X_perm))
                    else:  # CLASSIFICATION
                        unique_eval = np.unique(eval_y)
                        if len(unique_eval) < 2:
                            continue
                        
                        y_pred_proba = embedded.predict_proba(X_perm)
                        if y_pred_proba.shape[1] == 2:
                            perm_error = log_loss(eval_y, y_pred_proba[:, 1])
                        else:
                            perm_error = log_loss(eval_y, y_pred_proba)
                    
                    importances[i] = max((perm_error / baseline) - 1, 0)
                except:
                    importances[i] = 0
            
            return importances
        
        except Exception as e:
            return np.zeros(len(active_features))
    
    def _select_splitting_variables(self, X, y, active_features, protected_features):
        """
        Select splitting variable(s) using reinforcement learning approach
        """
        if len(active_features) == 0:
            return [], []
        
        try:
            vi = self._calculate_variable_importance(X, y, active_features)
            
            if np.sum(vi) == 0:
                selected_idx = np.random.choice(len(active_features))
                return [active_features[selected_idx]], [1.0]
            
            sorted_indices = np.argsort(vi)[::-1]
            
            if not self.use_linear_combination or self.k_linear == 1:
                best_idx = sorted_indices[0]
                return [active_features[best_idx]], [1.0]
            
            # ===== Linear Combination Split =====
            max_vi = vi[sorted_indices[0]]
            threshold = self.alpha * max_vi
            
            selected_features = []
            coefficients = []
            
            for idx in sorted_indices[:self.k_linear]:
                if vi[idx] >= threshold and vi[idx] > 0:
                    feat_idx = active_features[idx]
                    selected_features.append(feat_idx)
                    
                    # Safe correlation calculation
                    try:
                        y_numeric = y.astype(float)
                        
                        if len(np.unique(y_numeric)) > 1:
                            corr = np.corrcoef(X[:, feat_idx], y_numeric)[0, 1]
                            sign = np.sign(corr) if not np.isnan(corr) else 1.0
                        else:
                            sign = 1.0
                    except:
                        sign = 1.0
                    
                    coef = sign * np.sqrt(vi[idx])
                    coefficients.append(coef)
            
            if len(selected_features) == 0:
                best_idx = sorted_indices[0]
                return [active_features[best_idx]], [1.0]
            
            return selected_features, coefficients
        
        except Exception as e:
            if len(active_features) > 0:
                return [active_features[0]], [1.0]
            return [], []
    
    def _update_muted_protected(self, vi_dict, active_features, protected_features, p_total):
        """
        Update muted and protected feature sets
        """
        if not self.use_variable_muting:
            return active_features, protected_features
        
        n_to_mute = max(1, int(self.muting_rate * len(active_features)))
        
        if n_to_mute == 0 or len(active_features) <= (self.p_protected or 1):
            return active_features, protected_features
        
        # Get importances for non-protected features
        importances = []
        for f in active_features:
            if f not in protected_features:
                importance = vi_dict.get(f, 0)
                importances.append((f, importance))
        
        if len(importances) == 0:
            return active_features, protected_features
        
        # Sort by importance and mute lowest
        importances.sort(key=lambda x: x[1])
        to_mute = [f for f, _ in importances[:n_to_mute]]
        
        new_active = [f for f in active_features if f not in to_mute]
        
        # Ensure we have at least some features
        if len(new_active) == 0:
            new_active = list(active_features)
        
        return new_active, protected_features
    
    def _build_tree(self, X, y, active_features, protected_features, depth=0):
        """
        Recursively build a single RLT tree
        ROBUST - handles edge cases
        """
        n_samples, n_features = X.shape
        
        # Stopping criteria
        if (n_samples < self.min_samples_split or 
            len(active_features) == 0 or
            (self.max_depth is not None and depth >= self.max_depth)):
            
            if self.task == 'regression':
                leaf_value = np.mean(y)
            else:
                leaf_value = np.mean(y.astype(float))
            
            return {'type': 'leaf', 'value': leaf_value}
        
        # Select splitting variables
        split_features, coefficients = self._select_splitting_variables(
            X, y, active_features, protected_features
        )
        
        if len(split_features) == 0:
            if self.task == 'regression':
                leaf_value = np.mean(y)
            else:
                leaf_value = np.mean(y.astype(float))
            return {'type': 'leaf', 'value': leaf_value}
        
        # Create split values
        if len(split_features) == 1:
            split_values = X[:, split_features[0]]
        else:
            split_values = np.zeros(n_samples)
            for feat, coef in zip(split_features, coefficients):
                split_values += coef * X[:, feat]
        
        # Find split point
        q_low, q_high = 0.25, 0.75
        try:
            low_val = np.quantile(split_values, q_low)
            high_val = np.quantile(split_values, q_high)
        except:
            if self.task == 'regression':
                leaf_value = np.mean(y)
            else:
                leaf_value = np.mean(y.astype(float))
            return {'type': 'leaf', 'value': leaf_value}
        
        if low_val == high_val:
            if self.task == 'regression':
                leaf_value = np.mean(y)
            else:
                leaf_value = np.mean(y.astype(float))
            return {'type': 'leaf', 'value': leaf_value}
        
        split_point = np.random.uniform(low_val, high_val)
        
        # Split data
        left_mask = split_values <= split_point
        right_mask = ~left_mask
        
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            if self.task == 'regression':
                leaf_value = np.mean(y)
            else:
                leaf_value = np.mean(y.astype(float))
            return {'type': 'leaf', 'value': leaf_value}
        
        # Update protected features
        new_protected = set(protected_features).union(set(split_features))
        
        # VI for muting
        vi_dict = {f: 1.0 for f in split_features}
        
        # Update active features for children
        left_active, left_protected = self._update_muted_protected(
            vi_dict, active_features, new_protected, n_features
        )
        right_active, right_protected = list(left_active), set(left_protected)
        
        # Recursively build children
        left_child = self._build_tree(
            X[left_mask], y[left_mask], left_active, left_protected, depth + 1
        )
        right_child = self._build_tree(
            X[right_mask], y[right_mask], right_active, right_protected, depth + 1
        )
        
        return {
            'type': 'node',
            'features': split_features,
            'coefficients': coefficients,
            'split_point': split_point,
            'left': left_child,
            'right': right_child
        }
    
    def _predict_tree(self, tree, X):
        """
        Predict using a single tree
        """
        if tree['type'] == 'leaf':
            return np.full(X.shape[0], tree['value'])
        
        # Calculate split values
        if len(tree['features']) == 1:
            split_values = X[:, tree['features'][0]]
        else:
            split_values = np.zeros(X.shape[0])
            for feat, coef in zip(tree['features'], tree['coefficients']):
                split_values += coef * X[:, feat]
        
        predictions = np.zeros(X.shape[0])
        left_mask = split_values <= tree['split_point']
        
        if np.any(left_mask):
            predictions[left_mask] = self._predict_tree(tree['left'], X[left_mask])
        if np.any(~left_mask):
            predictions[~left_mask] = self._predict_tree(tree['right'], X[~left_mask])
        
        return predictions
    
    def fit(self, X, y, verbose=False):
        """
        Fit the RLT ensemble
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        self.n_features_ = X.shape[1]
        
        if self.p_protected is None:
            self.p_protected = max(1, int(np.log(self.n_features_)))
        
        if verbose:
            print(f"\nTraining RLT ({self.task.upper()}):")
            print(f"  - Samples: {X.shape[0]}, Features: {X.shape[1]}")
            print(f"  - Embedded Model: {self.embedded_model.upper()}")
            print(f"  - Variable Muting: {self.use_variable_muting} (rate={self.muting_rate})")
            print(f"  - Linear Combination: {self.use_linear_combination} (k={self.k_linear})")
            print(f"  - Trees: {self.n_estimators}")
        
        self.trees_ = []
        all_importances = np.zeros(self.n_features_)
        
        for i in range(self.n_estimators):
            if verbose and (i + 1) % 20 == 0:
                print(f"  Tree {i+1}/{self.n_estimators}")
            
            # Bootstrap sample
            boot_idx = np.random.choice(X.shape[0], size=X.shape[0], replace=True)
            X_boot = X[boot_idx]
            y_boot = y[boot_idx]
            
            # Build tree
            active_features = list(range(self.n_features_))
            protected_features = set()
            tree = self._build_tree(X_boot, y_boot, active_features, protected_features)
            self.trees_.append(tree)
            
            # Track importances
            if tree['type'] == 'node':
                for f in tree['features']:
                    all_importances[f] += 1
        
        self.feature_importances_ = all_importances / self.n_estimators
        
        if verbose:
            print("Training completed!\n")
        
        return self
    
    def predict(self, X):
        """
        Make predictions
        - Regression: returns real values
        - Classification: returns binary predictions (0/1)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = np.zeros((X.shape[0], len(self.trees_)))
        
        for i, tree in enumerate(self.trees_):
            predictions[:, i] = self._predict_tree(tree, X)
        
        # Average predictions
        mean_pred = np.mean(predictions, axis=1)
        
        if self.task == 'regression':
            return mean_pred
        else:  # CLASSIFICATION
            return (mean_pred >= 0.5).astype(int)
    
    def predict_proba(self, X):
        """
        Get probability predictions (classification only)
        """
        if self.task != 'classification':
            raise ValueError("predict_proba is only available for classification tasks")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = np.zeros((X.shape[0], len(self.trees_)))
        
        for i, tree in enumerate(self.trees_):
            predictions[:, i] = self._predict_tree(tree, X)
        
        # Average probabilities
        proba_class1 = np.mean(predictions, axis=1)
        
        # Return (prob_class_0, prob_class_1)
        return np.column_stack([1 - proba_class1, proba_class1])