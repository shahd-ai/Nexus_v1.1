from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
import pandas as pd
import numpy as np

from rlt_module import ReinforcementLearningTree


def _prepare_data(df, target_col):
    y = df[target_col].values

    # classification: transformer texte → labels
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)

    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True).values

    # détecter automatiquement la nature du problème
    task = "regression" if np.issubdtype(y.dtype, np.number) and len(np.unique(y)) > 10 else "classification"

    return X, y, task


def _evaluate_rlt(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if model.task == "regression":
        score = mean_squared_error(y_test, y_pred)
    else:
        score = accuracy_score(y_test, y_pred)

    return model, score


# ==========================================================
# RLT STANDARD
# ==========================================================
def train_rlt_standard(df, target_col, n_estimators=50):
    X, y, task = _prepare_data(df, target_col)

    model = ReinforcementLearningTree(
        n_estimators=n_estimators,
        task=task,
        embedded_model="rf",
        use_variable_muting=False,
        use_linear_combination=False,
        k_linear=1,
        random_state=42,
    )

    return _evaluate_rlt(model, X, y)


# ==========================================================
# RLT MUTING
# ==========================================================
def train_rlt_muting(df, target_col, n_estimators=50, muting_rate=0.5):
    X, y, task = _prepare_data(df, target_col)

    model = ReinforcementLearningTree(
        n_estimators=n_estimators,
        task=task,
        embedded_model="rf",
        use_variable_muting=True,
        muting_rate=muting_rate,
        use_linear_combination=False,
        k_linear=1,
        random_state=42,
    )

    return _evaluate_rlt(model, X, y)


# ==========================================================
# RLT LINEAR
# ==========================================================
def train_rlt_linear(df, target_col, n_estimators=50, k_linear=2):
    X, y, task = _prepare_data(df, target_col)

    model = ReinforcementLearningTree(
        n_estimators=n_estimators,
        task=task,
        embedded_model="et",
        use_variable_muting=False,
        use_linear_combination=True,
        k_linear=k_linear,
        random_state=42,
    )

    return _evaluate_rlt(model, X, y)


# ==========================================================
# RLT FULL (Paper)
# ==========================================================
def train_rlt_full(df, target_col, n_estimators=50, k_linear=2, muting_rate=0.5):
    X, y, task = _prepare_data(df, target_col)

    model = ReinforcementLearningTree(
        n_estimators=n_estimators,
        task=task,
        embedded_model="et",
        use_variable_muting=True,
        muting_rate=muting_rate,
        use_linear_combination=True,
        k_linear=k_linear,
        random_state=42,
    )

    return _evaluate_rlt(model, X, y)
