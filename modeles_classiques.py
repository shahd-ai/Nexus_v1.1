import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

# -------------------------
# 1️⃣ Lasso Regression
# -------------------------
def train_lasso(df, target_col, alpha=1.0):
    y = df[target_col]
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    X_scaled = StandardScaler().fit_transform(X)
    
    model = Lasso(alpha=alpha, max_iter=10000)
    cv_scores = cross_val_score(model, X_scaled, y, cv=10, scoring='neg_mean_squared_error')
    model.fit(X_scaled, y)
    
    return model, -cv_scores.mean()

# -------------------------
# 2️⃣ Gradient Boosting
# -------------------------
def train_boosting(df, target_col, n_estimators=1000, learning_rate=0.01, max_depth=3, min_samples_leaf=2):
    y = df[target_col]
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    X_scaled = StandardScaler().fit_transform(X)
    
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    cv_scores = cross_val_score(model, X_scaled, y, cv=10, scoring='neg_mean_squared_error')
    model.fit(X_scaled, y)
    
    return model, -cv_scores.mean()

# -------------------------
# 3️⃣ Random Forest
# -------------------------
def train_rf(df, target_col, n_estimators=1000, mtry=None, min_samples_leaf=2, bootstrap=1.0):
    y = df[target_col]
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    X_scaled = StandardScaler().fit_transform(X)
    
    p = X.shape[1]
    if mtry is None:
        mtry = int(np.sqrt(p))
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=mtry,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap < 1.0,  # True if bootstrap ratio <1
        random_state=42
    )
    cv_scores = cross_val_score(model, X_scaled, y, cv=10, scoring='neg_mean_squared_error')
    model.fit(X_scaled, y)
    
    return model, -cv_scores.mean()

# -------------------------
# 4️⃣ Extra Trees
# -------------------------
def train_et(df, target_col, n_estimators=500, mtry=None, min_samples_leaf=2, num_random_cuts=1):
    y = df[target_col]
    X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
    X_scaled = StandardScaler().fit_transform(X)
    
    p = X.shape[1]
    if mtry is None:
        mtry = int(np.sqrt(p))
    
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_features=mtry,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    cv_scores = cross_val_score(model, X_scaled, y, cv=10, scoring='neg_mean_squared_error')
    model.fit(X_scaled, y)
    
    return model, -cv_scores.mean()

# -------------------------
# 5️⃣ Placeholder for BART
# -------------------------
def train_bart(df, target_col):
    return "BART requires bartpy or R BART package"

# -------------------------
# 6️⃣ RLT-naive
# -------------------------
def train_rlt_naive(df, target_col):
    return "Custom RLT-naive implementation required"

# -------------------------
# 7️⃣ RLT
# -------------------------
def train_rlt(df, target_col):
    return "Custom RLT with muting implementation required"
