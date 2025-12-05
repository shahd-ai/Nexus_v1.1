# run_rlt_comparison.py
from rlt_module import ReinforcementLearningTree
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
import os

# ===== 0. Créer le dossier pour sauvegarder les modèles =====
if not os.path.exists("model"):
    os.makedirs("model")

# ===== 1. Préparer le dataset =====
X, y = make_regression(n_samples=300, n_features=50, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ===== 2. Créer différents modèles RLT =====
models = {
    "RLT_standard_ET": ReinforcementLearningTree(
        n_estimators=50, use_variable_muting=False, use_linear_combination=False, k_linear=1, embedded_model='et'
    ),
    "RLT_muting_moderate_ET": ReinforcementLearningTree(
        n_estimators=50, use_variable_muting=True, muting_rate=0.5, use_linear_combination=False, k_linear=1, embedded_model='et'
    ),
    "RLT_muting_aggressive_ET": ReinforcementLearningTree(
        n_estimators=50, use_variable_muting=True, muting_rate=0.8, use_linear_combination=False, k_linear=1, embedded_model='et'
    ),
    "RLT_linear_k2_ET": ReinforcementLearningTree(
        n_estimators=50, use_variable_muting=True, muting_rate=0.8, use_linear_combination=True, k_linear=2, embedded_model='et'
    ),
    "RLT_linear_k5_ET": ReinforcementLearningTree(
        n_estimators=50, use_variable_muting=True, muting_rate=0.8, use_linear_combination=True, k_linear=5, embedded_model='et'
    ),
    # Version RF embedded
    "RLT_linear_k2_RF": ReinforcementLearningTree(
        n_estimators=50, use_variable_muting=True, muting_rate=0.8, use_linear_combination=True, k_linear=2, embedded_model='rf'
    ),
    "RLT_linear_k5_RF": ReinforcementLearningTree(
        n_estimators=50, use_variable_muting=True, muting_rate=0.8, use_linear_combination=True, k_linear=5, embedded_model='rf'
    ),
}

# ===== 3. Entraîner et sauvegarder chaque modèle =====
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    # Sauvegarde
    joblib.dump(model, f"model/{name}.pkl")
    print(f"{name} saved to model/{name}.pkl")

# ===== 4. Évaluer les modèles =====
results = {}
for name, model in models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results[name] = mse
    print(f"{name} MSE: {mse:.4f}")

# ===== 5. Visualiser les résultats =====
plt.figure(figsize=(12,5))
plt.bar(results.keys(), results.values(), color='steelblue', alpha=0.7)
plt.ylabel("Mean Squared Error")
plt.title("RLT Model Comparison (ET vs RF embedded)")
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# ===== 6. Comparer l'importance des features =====
plt.figure(figsize=(20,5))
for idx, (name, model) in enumerate(models.items()):
    plt.subplot(1, len(models), idx+1)
    plt.bar(range(len(model.feature_importances_)), model.feature_importances_, color='orange')
    plt.title(name)
    plt.xlabel("Feature index")
    plt.ylabel("Importance")
plt.tight_layout()
plt.show()
