import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.utils import resample

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
    """Shared logic for RLT Classifier and Regressor"""
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=5, 
                 muting_threshold=0.0, embedded_model_depth=1, linear_combination=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.muting_threshold = muting_threshold
        self.embedded_model_depth = embedded_model_depth
        self.linear_combination = linear_combination
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = resample(X, y)
            root = self._build_tree(X_sample, y_sample, depth=0)
            self.trees.append(root)
        return self

    def predict(self, X):
        raise NotImplementedError

    def _build_tree(self, X, y, depth):
        node_pred = self._get_node_prediction(y)
        node = RLTNode(depth, node_pred)

        if (depth >= self.max_depth or 
            len(np.unique(y)) == 1 or 
            len(y) < self.min_samples_split):
            return node

        try:
            importances = self._get_embedded_importances(X, y)
            importances = importances ** self.linear_combination
        except:
            return node

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

# --- CLASSIFIER ---
class ReinforcementLearningTree(BaseRLT, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        return super().fit(X, y)

    def _get_node_prediction(self, y):
        counts = np.bincount(y, minlength=self.n_classes_)
        return counts / np.sum(counts)

    def _get_embedded_importances(self, X, y):
        if X.shape[0] < 5:
            return np.ones(X.shape[1]) / X.shape[1]
        model = ExtraTreesClassifier(n_estimators=10, max_depth=self.embedded_model_depth, 
                                     max_features="sqrt", random_state=None)
        model.fit(X, y)
        return model.feature_importances_

    def _calculate_split_score(self, y_left, y_right):
        def gini(y):
            if len(y) == 0:
                return 0
            counts = np.bincount(y, minlength=self.n_classes_)
            probs = counts / len(y)
            return 1.0 - np.sum(probs**2)
        n = len(y_left) + len(y_right)
        return (len(y_left)/n)*gini(y_left) + (len(y_right)/n)*gini(y_right)

    def predict_proba(self, X):
        all_probs = []
        for root in self.trees:
            probs_list = []
            for row in X:
                pred = self._predict_single(root, row)
                if isinstance(pred, np.ndarray):
                    probs_list.append(pred)
                else:
                    # Cas feuille: créer un vecteur de probabilités
                    p = np.zeros(self.n_classes_)
                    p[int(pred)] = 1.0
                    probs_list.append(p)
            probs = np.array(probs_list)
            if probs.shape[1] != self.n_classes_:
                probs = np.ones((len(X), self.n_classes_)) / self.n_classes_
            all_probs.append(probs)
        return np.mean(all_probs, axis=0)

    def predict(self, X):
        try:
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]
        except:
            return np.zeros(len(X), dtype=int)

# --- REGRESSOR ---
class ReinforcementLearningRegressor(BaseRLT, RegressorMixin):
    def _get_node_prediction(self, y):
        return np.mean(y)

    def _get_embedded_importances(self, X, y):
        if X.shape[0] < 5:
            return np.ones(X.shape[1]) / X.shape[1]
        model = ExtraTreesRegressor(n_estimators=10, max_depth=self.embedded_model_depth, 
                                    max_features="sqrt", random_state=None)
        model.fit(X, y)
        return model.feature_importances_

    def _calculate_split_score(self, y_left, y_right):
        def variance_score(y):
            if len(y) == 0:
                return 0
            return np.var(y) * len(y)
        return variance_score(y_left) + variance_score(y_right)

    def predict(self, X):
        try:
            all_preds = []
            for root in self.trees:
                preds = []
                for row in X:
                    pred = self._predict_single(root, row)
                    if isinstance(pred, (int, float, np.number)):
                        preds.append(float(pred))
                    else:
                        preds.append(np.mean(pred) if len(pred) > 0 else 0.0)
                all_preds.append(np.array(preds))
            return np.mean(all_preds, axis=0)
        except Exception as e:
            return np.zeros(len(X))
