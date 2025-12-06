import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from rlt_modular import *

print("\n" + "="*70)
print("TRAINING RLT MODELS - INDIVIDUAL SAVES")
print("="*70)

X, y = create_test_scenarios()['scenario_2'](n=200, p=200)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

print(f"\nDataset: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
print(f"Features: {X_train.shape[1]}\n")

# Define configurations
configs = [
    {
        'name': 'RLT_standard', 
        'params': {
            'n_estimators': 50, 
            'embedded_model': 'et', 
            'task': 'regression'
        }
    },
    {
        'name': 'RLT_muting_50', 
        'params': {
            'n_estimators': 50, 
            'embedded_model': 'et', 
            'task': 'regression',
            'use_variable_muting': True, 
            'muting_rate': 0.5
        }
    },
    {
        'name': 'RLT_muting_80', 
        'params': {
            'n_estimators': 50, 
            'embedded_model': 'et', 
            'task': 'regression',
            'use_variable_muting': True, 
            'muting_rate': 0.8
        }
    },
    {
        'name': 'RLT_linear_k2',
        'params': {
            'n_estimators': 50, 
            'embedded_model': 'et', 
            'task': 'regression',
            'use_variable_muting': True, 
            'muting_rate': 0.8,
            'use_linear_combination': True, 
            'k_linear': 2
        }
    },
    {
        'name': 'RLT_linear_k5',
        'params': {
            'n_estimators': 50, 
            'embedded_model': 'et', 
            'task': 'regression',
            'use_variable_muting': True, 
            'muting_rate': 0.8,
            'use_linear_combination': True, 
            'k_linear': 5
        }
    },
    {
        'name': 'RLT_RF_embedded',
        'params': {
            'n_estimators': 50, 
            'embedded_model': 'rf', 
            'task': 'regression',
            'use_variable_muting': True, 
            'muting_rate': 0.8,
            'use_linear_combination': True, 
            'k_linear': 2
        }
    }
]

# Train each model
total = len(configs)
for i, config in enumerate(configs, 1):
    model_name = config['name']
    print(f"\n[{i}/{total}] Training {model_name}...")
    print("-" * 50)
    
    # Create and train model
    rlt = ReinforcementLearningTree(**config['params'])
    rlt.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rlt.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"\n  Results for {model_name}:")
    print(f"    MSE: {mse:.4f}")
    
    # Save model with test data
    model_data = {
        'model': rlt,
        'mse': mse,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'params': config['params']
    }
    
    save_model(model_data, model_name)

print("\n" + "="*70)
print("ALL MODELS TRAINED AND SAVED")
print("="*70)

# Show all saved models
saved = list_saved_models()
print("\nSaved models:")
for model_name in saved:
    print(f"  - {model_name}")

print("\n" + "="*70 + "\n")