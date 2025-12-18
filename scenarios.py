import os
import numpy as np
from scipy.stats import norm


OUTPUT_DIR = "Scenarios"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_dataset(scenario_name, p, X_train, y_train, X_test, y_test):
    train_path = os.path.join(OUTPUT_DIR, f"{scenario_name}_p{p}_train.npz")
    test_path = os.path.join(OUTPUT_DIR, f"{scenario_name}_p{p}_test.npz")

    np.savez(train_path, X=X_train, y=y_train)
    np.savez(test_path, X=X_test, y=y_test)

    print(f"Saved: {train_path}")
    print(f"Saved: {test_path}")


# ---------------------------------------------------------------------
# Scenario 1 : Classification with independent covariates
# Paper: N=100, X ~ Unif[0,1]^p
# mu = Φ(10*(X1 - 1) + 20*|X2 - 0.5|)
# Y ~ Bernoulli(mu)
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Scenario 2 : Non-linear model, independent covariates
# Paper: N=100
# Y = 100*(X1 - 0.5)^2 * (X2 - 0.25)+  + ε , ε ~ N(0,1)
# ---------------------------------------------------------------------

def scenario2(N, p, test_size=1000, seed=None, save=True):
    if seed is not None:
        np.random.seed(seed)

    def generate(n):
        X = np.random.uniform(0, 1, size=(n, p))
        positive = np.maximum(X[:, 1] - 0.25, 0)
        y = 100 * (X[:, 0] - 0.5)**2 * positive + np.random.normal(0, 1, n)
        return X, y

    X_train, y_train = generate(N)
    X_test, y_test = generate(test_size)

    if save:
        save_dataset("Scenario2", p, X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------
# Scenario 3 : Strong correlation, checkerboard-like model
# Paper: N = 300, X ~ N(0, Σ) with Σ[i,j] = 0.9^|i-j|
# Y = 2*X50*X100 + 2*X150*X200 + ε, ε ~ N(0,1)
# ---------------------------------------------------------------------

def scenario3(N, p, test_size=1000, seed=None, save=True):
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(p)
    Sigma = 0.9 ** np.abs(idx[:, None] - idx[None, :])

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


# ---------------------------------------------------------------------
# Scenario 4 : Linear model
# Paper: N = 200
# X ~ N(0, Σ) with Σ[i,j] = 0.5^|i-j| + 0.2*I(i=j)
# Y = 2*X50 + 2*X100 + 4*X150 + ε
# ---------------------------------------------------------------------

def scenario4(N, p, test_size=1000, seed=None, save=True):
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(p)
    Sigma = 0.5 ** np.abs(idx[:, None] - idx[None, :]) + 0.2 * np.eye(p)

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
