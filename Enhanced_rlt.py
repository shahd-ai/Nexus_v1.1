"""
Business-Agnostic RLT Pipeline
==============================
• Works on ANY tabular dataset
• Auto-detects classification vs regression
• Generic preprocessing
• Train / Save / Load / Predict
"""

import joblib
import numpy as np
import pandas as pd
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

# =====================================================
# RLT MODELS
# =====================================================

class EnhancedRLTClassifier(ReinforcementLearningTree):
    def __init__(self):
        super().__init__(
            n_estimators=50,
            max_depth=8,
            min_samples_split=10,
            muting_threshold=0.15,
            embedded_model_depth=2,
            linear_combination=2
        )

class EnhancedRLTRegressor(ReinforcementLearningRegressor):
    def __init__(self):
        super().__init__(
            n_estimators=50,
            max_depth=8,
            min_samples_split=10,
            muting_threshold=0.15,
            embedded_model_depth=2,
            linear_combination=2
        )

# =====================================================
# UTILS
# =====================================================

def detect_task_type(y):
    """Auto-detect classification or regression"""
    if y.dtype == "object":
        return "classification"
    unique_ratio = len(np.unique(y)) / len(y)
    if unique_ratio < 0.1 or y.dtype == int:
        return "classification"
    return "regression"

def preprocess_data(df, target_col):
    """Generic preprocessing"""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Encode target if categorical
    label_encoder = None
    if y.dtype == "object":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Keep numeric features only
    X = X.select_dtypes(include=[np.number])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_array = y.values if isinstance(y, pd.Series) else np.array(y)

    return X_scaled, y_array, scaler, label_encoder, X.columns.tolist()

def evaluate_model(task, y_true, y_pred):
    """Generic evaluation"""
    if task == "classification":
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
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
    model_path="./rlt_models/rlt_model.joblib",
    test_size=0.2
):
    print("\n" + "=" * 80)
    print("TRAINING BUSINESS-FREE RLT MODEL")
    print("=" * 80)

    X, y, scaler, label_encoder, feature_names = preprocess_data(df, target_col)
    task = detect_task_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model = EnhancedRLTClassifier() if task == "classification" else EnhancedRLTRegressor()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = evaluate_model(task, y_test, y_pred)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        joblib.dump({
            "model": model,
            "scaler": scaler,
            "label_encoder": label_encoder,
            "task": task,
            "features": feature_names
        }, f)

    print(f"✓ Model trained as: {task.upper()}")
    print("✓ Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n💾 Model saved to: {model_path}")
    return metrics

# =====================================================
# LOAD & PREDICT
# =====================================================

def load_and_predict(model_path, df):
    print("\n" + "=" * 80)
    print("LOADING MODEL & MAKING PREDICTIONS")
    print("=" * 80)

    with open(model_path, "rb") as f:
        obj = joblib.load(f)

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

def evaluate_saved_rlt_model(
    df,
    target_col,
    model_path,
    test_size=0.2,
    random_state=42,
    n_samples_preview=20
):
    """
    Evaluate a saved RLT model on a recreated train/test split
    (RAW features → model handles scaling internally)

    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset including target
    target_col : str
        Name of target column
    model_path : str
        Path to saved RLT model (.joblib)
    test_size : float
        Test split ratio
    random_state : int
        Random seed
    n_samples_preview : int
        Number of prediction samples to display
    """

    # === Preprocess to get feature list & encoders ===
    X_scaled, y, scaler, label_encoder, features = preprocess_data(df, target_col)
    X_raw = df[features]

    # === Recreate train/test split on RAW data ===
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state
    )

    # === Load model & predict ===
    preds, probs = load_and_predict(model_path, X_test)

    # === Restore original labels if needed ===
    if label_encoder is not None:
        try:
            y_test_disp = label_encoder.inverse_transform(y_test)
        except Exception:
            y_test_disp = y_test
    else:
        y_test_disp = y_test

    # === Preview predictions ===
    print("\nSample predictions vs actual:")
    for i, (p, a) in enumerate(zip(preds[:n_samples_preview], y_test_disp[:n_samples_preview])):
        print(f"{i+1:2d}. pred={p} | actual={a}")

    # === Evaluation ===
    print("\nAccuracy:", accuracy_score(y_test_disp, preds))

    print("\nClassification report:")
    print(classification_report(y_test_disp, preds, zero_division=0))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test_disp, preds))

    return {
        "accuracy": accuracy_score(y_test_disp, preds),
        "confusion_matrix": confusion_matrix(y_test_disp, preds),
        "report": classification_report(y_test_disp, preds, zero_division=0, output_dict=True),
        "probs": probs
    }
def main():
    """
    Replace this with ANY CSV / DataFrame for testing
    """

    # Example: classification dataset
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

    # Train
    train_and_save(
        df=df,
        target_col="target",
        model_path="./rlt_models/rlt_enhanced.joblib"
    )

    # Predict on new samples
    new_df = df.drop(columns=["target"]).sample(5)
    preds, probs = load_and_predict("./rlt_models/rlt_enhanced.joblib", new_df)

    print("\nSample predictions:")
    print(preds)
    if probs is not None:
        print("\nPrediction probabilities:")
        print(probs)

if __name__ == "__main__":
    main()
