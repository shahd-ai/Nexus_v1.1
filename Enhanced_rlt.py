"""
Business-Agnostic RLT Pipeline
==============================
• Works on ANY tabular dataset
• Auto-detects classification vs regression
• Generic preprocessing
• Train / Save / Load / Predict
• Adaptive hyperparameter tuning
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import warnings
warnings.filterwarnings("ignore")

from rlt_module import ReinforcementLearningTree, ReinforcementLearningRegressor
from preparation_data import impute_missing_with_rf

# =====================================================
# ADAPTIVE HYPERPARAMETERS
# =====================================================

def adaptive_rlt_params(n_samples, n_features):
    """
    Adaptively tune RLT hyperparameters based on dataset size and complexity.
    
    Args:
        n_samples: Number of training samples
        n_features: Number of features
    
    Returns:
        Dictionary of adaptive hyperparameters
    """
    return dict(
        n_estimators=min(100, max(20, n_samples // 20)),
        max_depth=min(12, max(4, int(np.log2(n_features + 1) * 2))),
        min_samples_split=max(2, n_samples // 100),
        muting_threshold=0.1 if n_samples > 1000 else 0.2,
        embedded_model_depth=2 if n_features < 20 else 3,
        linear_combination=min(3, n_features // 5)
    )

# =====================================================
# RLT MODELS
# =====================================================

class EnhancedRLTClassifier(ReinforcementLearningTree):
    def __init__(self, **params):
        """Initialize classifier with optional custom parameters"""
        defaults = dict(
            n_estimators=50,
            max_depth=8,
            min_samples_split=10,
            muting_threshold=0.15,
            embedded_model_depth=2,
            linear_combination=2
        )
        defaults.update(params)
        super().__init__(**defaults)

class EnhancedRLTRegressor(ReinforcementLearningRegressor):
    def __init__(self, **params):
        """Initialize regressor with optional custom parameters"""
        defaults = dict(
            n_estimators=50,
            max_depth=8,
            min_samples_split=10,
            muting_threshold=0.15,
            embedded_model_depth=2,
            linear_combination=2
        )
        defaults.update(params)
        super().__init__(**defaults)

# =====================================================
# UTILS
# =====================================================

def detect_task_type(y):
    """Auto-detect classification or regression"""
    y = np.asarray(y)
    unique = np.unique(y)
    if y.dtype == object:
        return "classification"
    if len(unique) <= 20 and np.all(unique == unique.astype(int)):
        return "classification"
    return "regression"

def preprocess_data(df, target_col):
    """Generic preprocessing with smart NaN imputation"""
    # Handle missing values using Random Forest imputation
    if df.isna().sum().sum() > 0:
        print("\n🔧 Imputing missing values using Random Forest...")
        df = impute_missing_with_rf(df)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    label_encoder = None
    if y.dtype == "object":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    X = X.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_array = y.values if isinstance(y, pd.Series) else np.array(y)
    return X_scaled, y_array, scaler, label_encoder, X.columns.tolist()

def evaluate_model(task, y_true, y_pred):
    """Generic evaluation for both classification and regression"""
    if task == "classification":
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    else:
        return {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }

# =====================================================
# TRAIN & SAVE
# =====================================================

def train_and_save(
    df,
    target_col,
    model_path="./rlt_models/rlt_model.pkl",
    test_size=0.2,
    use_adaptive=True
):
    """Train and save RLT model with adaptive or fixed parameters."""
    print("\n" + "=" * 80)
    print("TRAINING BUSINESS-FREE RLT MODEL")
    print("=" * 80)

    X, y, scaler, label_encoder, feature_names = preprocess_data(df, target_col)
    task = detect_task_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if use_adaptive:
        adaptive_params = adaptive_rlt_params(X_train.shape[0], X_train.shape[1])
        print(f"\n📊 Using adaptive hyperparameters:")
        for k, v in adaptive_params.items():
            print(f"  {k}: {v}")
        model = EnhancedRLTClassifier(**adaptive_params) if task == "classification" else EnhancedRLTRegressor(**adaptive_params)
    else:
        model = EnhancedRLTClassifier() if task == "classification" else EnhancedRLTRegressor()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate_model(task, y_test, y_pred)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump({
            "model": model,
            "scaler": scaler,
            "label_encoder": label_encoder,
            "task": task,
            "features": feature_names
        }, f)

    print(f"\n✓ Model trained as: {task.upper()}")
    print("✓ Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n💾 Model saved to: {model_path}")
    return metrics

# =====================================================
# LOAD & PREDICT
# =====================================================

def load_and_predict(model_path, df):
    """Load model and make predictions"""
    print("\n" + "=" * 80)
    print("LOADING MODEL & MAKING PREDICTIONS")
    print("=" * 80)

    with open(model_path, "rb") as f:
        obj = pickle.load(f)

    model = obj["model"]
    scaler = obj["scaler"]
    label_encoder = obj["label_encoder"]
    task = obj["task"]
    features = obj["features"]

    X = df[features]
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)

    if task == "classification":
        probs = model.predict_proba(X_scaled) if hasattr(model, "predict_proba") else None
        if label_encoder:
            preds = label_encoder.inverse_transform(preds)
        return preds, probs

    return preds

# =====================================================
# EXAMPLE USAGE
# =====================================================

def main():
    """Replace this with ANY CSV / DataFrame for testing"""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df["target"] = y

    train_and_save(
        df=df,
        target_col="target",
        model_path="./rlt_models/rlt_classification.pkl",
        use_adaptive=True
    )

    new_df = df.drop(columns=["target"]).sample(5)
    preds, probs = load_and_predict("./rlt_models/rlt_classification.pkl", new_df)

    print("\nSample predictions:")
    print(preds)
    if probs is not None:
        print("\nPrediction probabilities:")
        print(probs)

if __name__ == "__main__":
    main()
 