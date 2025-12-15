from rlt_module import ReinforcementLearningTree
from sklearn.metrics import mean_squared_error
import joblib
import os
from scenarios import scenario1, scenario2, scenario3, scenario4 ,evaluate_scenario

os.makedirs("model", exist_ok=True)


N = 5000
p = 200
X_train, y_train, X_test, y_test = scenario2(
    N, p, test_size=1000, seed=42, save=False
)

print("Scenario 2 loaded:", X_test.shape)

model = ReinforcementLearningTree(
    n_estimators=100,
    task='regression',
    embedded_model='rf',
    use_variable_muting=True,
    muting_rate=0.8,
    use_linear_combination=False,
    k_linear=1,
    random_state=42
)

print("\nTraining RLT_enhanced_aggressive_RF...")
model.fit(X_train, y_train)
print("Training complete.")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"RLT_enhanced_aggressive_RF MSE: {mse:.4f}")

save_path = "model/RLT_enhanced_aggressive_RF_scenario2.joblib"
joblib.dump(model, save_path)
print(f"Model saved to: {save_path}")
