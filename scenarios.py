import os
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error



OUTPUT_DIR = "Scenarios"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_dataset(scenario_name, p, X_train, y_train, X_test, y_test):
    train_path = os.path.join(OUTPUT_DIR, f"{scenario_name}_p{p}_train.npz")
    test_path  = os.path.join(OUTPUT_DIR, f"{scenario_name}_p{p}_test.npz")

    np.savez(train_path, X=X_train, y=y_train)
    np.savez(test_path,  X=X_test,  y=y_test)

    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")


#############################################
#   SCENARIO 1
#############################################

def scenario1(N, p, test_size=1000, seed=None, save=True):
    if seed is not None:
        np.random.seed(seed)

    def generate(n):
        X = np.random.uniform(0, 1, size=(n, p))
        mu = norm.cdf(10 * (X[:, 0] - 1) + 20 * np.abs(X[:, 1] - 0.5))
        y = np.random.binomial(1, mu)
        return X, y

    X_train, y_train = generate(N)
    X_test, y_test = generate(test_size)

    if save:
        save_dataset("Scenario1", p, X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test


#############################################
#   SCENARIO 2
#############################################

def scenario2(N, p, test_size=1000, seed=None, save=True):
    if seed is not None:
        np.random.seed(seed)

    def generate(n):
        X = np.random.uniform(0, 1, size=(n, p))
        y = 100 * (X[:, 0] - 0.5)**2 * np.maximum(X[:, 1] - 0.25, 0) \
            + np.random.normal(0, 1, n)
        return X, y

    X_train, y_train = generate(N)
    X_test, y_test = generate(test_size)

    if save:
        save_dataset("Scenario2", p, X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test


#############################################
#   SCENARIO 3
#############################################

def scenario3(N, p, test_size=1000, seed=None, save=True):
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(p)
    Sigma = 0.9 ** np.abs(idx.reshape(-1, 1) - idx.reshape(1, -1))

    def generate(n):
        X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
        y = (
            2 * X[:, 49] * X[:, 99] +
            2 * X[:, 149] * X[:, 199] +
            np.random.normal(0, 1, n)
        )
        return X, y

    X_train, y_train = generate(N)
    X_test, y_test = generate(test_size)

    if save:
        save_dataset("Scenario3", p, X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test


#############################################
#   SCENARIO 4
#############################################

def scenario4(N, p, test_size=1000, seed=None, save=True):
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(p)
    Sigma = 0.5 ** np.abs(idx.reshape(-1, 1) - idx.reshape(1, -1)) \
            + 0.2 * np.eye(p)

    def generate(n):
        X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
        y = (
            2 * X[:, 49] +
            2 * X[:, 99] +
            4 * X[:, 149] +
            np.random.normal(0, 1, n)
        )
        return X, y

    X_train, y_train = generate(N)
    X_test, y_test = generate(test_size)

    if save:
        save_dataset("Scenario4", p, X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test

def evaluate_scenario2(scen_func, scen_name, models_dict, N=5000, p=200):
    print(f"\n\n============================")
    print(f"   Evaluating {scen_name}")
    print("============================\n")

    # Load data
    X_train, y_train, X_test, y_test = scen_func(
        N, p, test_size=1000, seed=42, save=False
    )
    print(f"Data loaded: {X_train.shape}")

    results = {}

    # Evaluate all RLT models
    for name, model in models_dict.items():
        try:
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            results[name] = mse
        except Exception as e:
            results[name] = np.nan
            print(f"❌ Error in {name}: {e}")

    df = pd.DataFrame.from_dict(results, orient="index", columns=["MSE"])
    df.sort_values("MSE", inplace=True)

    print("\nMSE Results:")

    # Compute Gain vs baseline RLT_standard
    if "RLT_standard" in df.index:
        df["Gain_vs_Standard"] = df.loc["RLT_standard","MSE"] - df["MSE"]

    # Plot MSE
    plt.figure(figsize=(10,6))
    colors = ["green" if m == df["MSE"].min() else "blue" for m in df["MSE"]]
    plt.bar(df.index, df["MSE"], color=colors)
    plt.title(f"MSE Comparison - {scen_name}")
    plt.xticks(rotation=45)
    plt.show()

    # Feature importance
    def plot_importance(model, title):
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "importance_"):
            imp = model.importance_
        else:
            print(f"No feature importance for {title}")
            return
        
        plt.figure(figsize=(14,4))
        plt.bar(np.arange(len(imp)), imp)
        plt.title(f"Feature Importance — {title} ({scen_name})")
        plt.xlabel("Feature index")
        plt.ylabel("Importance")
        plt.show()

    for name, model in models_dict.items():
        plot_importance(model, name)

    return df
def evaluate_scenario(scen_func, scen_name, models_dict, N=5000, p=10):
    print(f"\n=== ⬛ {scen_name}: running with N={N}, p={p} ===")

    # Load scenario data
    X_train, y_train, X_test, y_test = scen_func(
        N, p, test_size=1000, seed=42, save=False
    )
    print(f"Data loaded: {X_train.shape}")

    results = {}

    # Evaluate all RLT models
    for name, model in models_dict.items():
        try:
            y_pred = model.predict(X_test)
            mse = np.mean((y_test - y_pred)**2)
            results[name] = mse
            print(f"✔️ {name}: MSE = {mse:.4f}")
        except Exception as e:
            print(f"❌ Error evaluating {name}: {e}")

    return results
