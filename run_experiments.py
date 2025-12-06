import numpy as np
import pandas as pd

from scenarios import scenario1, scenario2, scenario3, scenario4
import modeles_classiques as mc


def evaluate_models_on_scenario(X_train, y_train, X_test, y_test):
    df_train = pd.DataFrame(X_train)
    df_train["target"] = y_train

    df_test = pd.DataFrame(X_test)
    df_test["target"] = y_test

    results = {}

    # Lasso
    model, mse = mc.train_lasso(df_train, "target", alpha=1.0)
    y_pred = model.predict(df_test.drop(columns=["target"]))
    results["Lasso"] = np.mean((df_test["target"] - y_pred) ** 2)

    # Boosting
    model, mse = mc.train_boosting(df_train, "target")
    y_pred = model.predict(df_test.drop(columns=["target"]))
    results["Boosting"] = np.mean((df_test["target"] - y_pred) ** 2)

    # BART
    model, mse = mc.train_bart(df_train, "target")
    y_pred = model["model"].predict(df_test.drop(columns=["target"]))
    results["BART"] = np.mean((df_test["target"] - y_pred) ** 2)

    # Random Forest
    model, mse = mc.train_rf(df_train, "target")
    y_pred = model.predict(df_test.drop(columns=["target"]))
    results["RF"] = np.mean((df_test["target"] - y_pred) ** 2)

    # RF sqrt(p)
    model, mse = mc.train_rf_sqrtp(df_train, "target")
    y_pred = model.predict(df_test.drop(columns=["target"]))
    results["RF_sqrtp"] = np.mean((df_test["target"] - y_pred) ** 2)

    # RF log(p)
    model, mse = mc.train_rf_logp(df_train, "target")
    y_pred = model.predict(df_test.drop(columns=["target"]))
    results["RF_logp"] = np.mean((df_test["target"] - y_pred) ** 2)

    # Extra Trees
    model, mse = mc.train_et(df_train, "target")
    y_pred = model.predict(df_test.drop(columns=["target"]))
    results["ET"] = np.mean((df_test["target"] - y_pred) ** 2)

    return results


def run_all_experiments(N=2000, p=200):
    scenario_functions = {
        "Scenario 1": scenario1,
        "Scenario 2": scenario2,
        "Scenario 3": scenario3,
        "Scenario 4": scenario4
    }

    all_results = {}

    for name, func in scenario_functions.items():
        print(f"\n=== Running {name} ===")
        X_train, y_train, X_test, y_test = func(N=N, p=p, save=False)
        all_results[name] = evaluate_models_on_scenario(X_train, y_train, X_test, y_test)

    return pd.DataFrame(all_results)


if __name__ == "__main__":
    df = run_all_experiments()
    print(df)
