import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error
import time
import warnings
import sys
from threading import Lock

warnings.filterwarnings('ignore')

# ============ THREAD-SAFE PRINTING ============
print_lock = Lock()

def safe_print(*args, **kwargs):
    """Thread-safe print function"""
    with print_lock:
        print(*args, **kwargs, flush=True)

# ============ IMPORT RLT WITH ERROR HANDLING ============
try:
    from rlt_module import ReinforcementLearningTree, ReinforcementLearningRegressor
    RLT_AVAILABLE = True
except ImportError:
    safe_print("❌ RLT module not available!")
    RLT_AVAILABLE = False
    exit(1)

# ============ WRAPPER POUR RLT AVEC GESTION D'ERREURS ============
class SafeRLTWrapper:
    """Wrapper qui gère les erreurs de prédiction RLT"""
    
    def __init__(self, model):
        self.model = model
        self.n_classes_ = None
        
    def fit(self, X, y):
        self.model.fit(X, y)
        if hasattr(self.model, 'n_classes_'):
            self.n_classes_ = self.model.n_classes_
        else:
            self.n_classes_ = len(np.unique(y))
        return self
    
    def predict(self, X):
        try:
            return self.model.predict(X)
        except Exception as e:
            safe_print(f"\n      ⚠️  RLT predict error: {str(e)[:50]}... Using fallback")
            if hasattr(self.model, 'classes_'):
                return np.full(len(X), self.model.classes_[0])
            return np.zeros(len(X))
    
    def predict_proba(self, X):
        try:
            probs = self.model.predict_proba(X)
            if probs.shape[0] != len(X):
                raise ValueError(f"Shape mismatch: {probs.shape[0]} != {len(X)}")
            return probs
        except Exception as e:
            safe_print(f"\n      ⚠️  RLT predict_proba error: {str(e)[:50]}... Using fallback")
            return np.ones((len(X), self.n_classes_)) / self.n_classes_

# ============ NOISE FUNCTIONS ============
def add_gaussian_noise(X, noise_level=0.1):
    """Ajoute du bruit gaussien aux données"""
    X_noisy = X.copy()
    noise = np.random.normal(0, noise_level * np.std(X_noisy, axis=0), X_noisy.shape)
    X_noisy += noise
    return X_noisy

# ============ DETECT TARGET ============
def detect_target_column(df):
    """Détecte automatiquement la colonne cible"""
    target_keywords = ["quality", "target", "label", "Class", "y", "output", "diagnosis", "status"]
    for col in df.columns:
        if any(key in col.lower() for key in target_keywords):
            return col
    if df.iloc[:, -1].nunique() < 20:
        return df.columns[-1]
    return df.columns[-1]

# ============ LOAD DATASETS ============
def load_all_datasets():
    """Charge tous les datasets disponibles"""
    datasets = {}
    dataset_configs = [
        {"name": "Boston Housing", "file": "Boston_housing.csv", "skip_first_col": False},
        {"name": "Parkinsons", "file": "parkinsons.csv", "skip_first_col": False},
        {"name": "Sonar", "file": "sonar.csv", "skip_first_col": False},
        {"name": "White Wine", "file": "winequality-white.csv", "skip_first_col": False},
        {"name": "Red Wine", "file": "winequality-red.csv", "skip_first_col": False},
        {"name": "Breast Cancer", "file": "breast_cancer.csv", "skip_first_col": False},
      
    ]
    
    for config in dataset_configs:
        try:
            df = pd.read_csv(config['file'])
            target_col = detect_target_column(df)
            
            if config['skip_first_col'] and len(df.columns) > 2:
                df = df.iloc[:, 1:]
                if target_col == df.columns[0]:
                    target_col = detect_target_column(df)
            
            df_numeric = df.select_dtypes(include=[np.number])
            
            if target_col not in df_numeric.columns:
                if target_col in df.columns:
                    if df[target_col].dtype == 'object':
                        from sklearn.preprocessing import LabelEncoder
                        le = LabelEncoder()
                        df_numeric[target_col] = le.fit_transform(df[target_col])
            
            if target_col in df_numeric.columns:
                X = df_numeric.drop(columns=[target_col]).values
                y = df_numeric[target_col].values
                
                mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                X = X[mask]
                y = y[mask]
                
                if len(X) > 0:
                    n_unique = len(np.unique(y))
                    if n_unique <= 10 and n_unique < len(y) * 0.05:
                        task_type = 'classification'
                    else:
                        task_type = 'regression'
                    
                    datasets[config['name']] = {
                        'X': X,
                        'y': y,
                        'type': task_type
                    }
                    safe_print(f"✅ Loaded: {config['name']:<25} | Shape: {X.shape} | Type: {task_type}")
                    
        except Exception as e:
            safe_print(f"⚠️  Skipped: {config['name']:<25} | Error: {str(e)[:50]}")
    
    return datasets

# ============ TRAIN AND EVALUATE WITH NOISE ============
def train_and_evaluate_with_noise(model, X_train, y_train, X_test, y_test, task_type, noise_type='gaussian', noise_level=0.1):
    """Entraîne et évalue un modèle avec et sans bruit"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    start_time = time.time()
    try:
        model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
    except Exception as e:
        safe_print(f"❌ Training failed: {str(e)[:50]}")
        return None, None, None, None
    
    # Test on CLEAN
    try:
        y_pred_clean = model.predict(X_test_scaled)
    except Exception as e:
        safe_print(f"❌ Prediction failed: {str(e)[:50]}")
        return None, None, training_time, None
    
    # Add noise
    if noise_level > 0:
        X_test_noisy = add_gaussian_noise(X_test_scaled, noise_level)
    else:
        X_test_noisy = X_test_scaled
    
    # Test on NOISY
    try:
        y_pred_noisy = model.predict(X_test_noisy)
    except Exception as e:
        safe_print(f"❌ Noisy prediction failed: {str(e)[:50]}")
        y_pred_noisy = y_pred_clean
    
    # Calculate scores
    if task_type == 'classification':
        score_clean = accuracy_score(y_test, y_pred_clean)
        score_noisy = accuracy_score(y_test, y_pred_noisy)
        metric_name = 'Accuracy'
    else:
        score_clean = np.sqrt(mean_squared_error(y_test, y_pred_clean))
        score_noisy = np.sqrt(mean_squared_error(y_test, y_pred_noisy))
        metric_name = 'RMSE'
    
    return score_clean, score_noisy, training_time, metric_name

# ============ MAIN COMPARISON ============
def run_comparison_with_noise(noise_levels=[0.0, 0.1, 0.2, 0.3], noise_type='gaussian'):
    """Fonction principale de comparaison"""
    # ============ CREATE OUTPUT DIRECTORY ============
    import os
    output_dir = 'dso2_part2_fixed'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/results', exist_ok=True)
    os.makedirs(f'{output_dir}/models', exist_ok=True)
    os.makedirs(f'{output_dir}/datasets_noisy', exist_ok=True)
    
    safe_print("\n" + "="*100)
    safe_print(f"🚀 4-WAY COMPARISON - FIXED VERSION (Thread-Safe)")
    safe_print(f"📁 Output directory: {output_dir}/")
    safe_print("="*100 + "\n")
    
    datasets = load_all_datasets()
    
    if not datasets:
        safe_print("\n❌ No datasets found!")
        return
    
    safe_print(f"\n✅ Loaded {len(datasets)} datasets")
    safe_print(f"🔊 Noise levels: {noise_levels}")
    safe_print("="*100)
    
    results = []
    
    for dataset_name, data in datasets.items():
        safe_print(f"\n{'='*100}")
        safe_print(f"🔍 Dataset: {dataset_name}")
        safe_print(f"{'='*100}")
        
        X = data['X']
        y = data['y']
        task_type = data['type']
        
        safe_print(f"Task: {task_type.upper()} | Samples: {len(X)} | Features: {X.shape[1]}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # ============ SAVE DATASETS (once per dataset) ============
        df_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
        df_train['target'] = y_train
        df_train.to_csv(f'{output_dir}/datasets_noisy/{dataset_name.replace(" ", "_")}_train.csv', index=False)
        
        df_test_clean = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
        df_test_clean['target'] = y_test
        df_test_clean.to_csv(f'{output_dir}/datasets_noisy/{dataset_name.replace(" ", "_")}_test_clean.csv', index=False)
        
        for noise_level in noise_levels:
            noise_label = f"{int(noise_level*100)}%"
            safe_print(f"\n  📊 Noise Level: {noise_label}")
            
            # Save noisy test data
            if noise_level > 0:
                scaler_temp = StandardScaler()
                X_test_scaled_temp = scaler_temp.fit_transform(X_test)
                X_test_noisy_temp = add_gaussian_noise(X_test_scaled_temp, noise_level)
                df_test_noisy = pd.DataFrame(X_test_noisy_temp, columns=[f'feature_{i}' for i in range(X_test.shape[1])])
                df_test_noisy['target'] = y_test
                df_test_noisy.to_csv(f'{output_dir}/datasets_noisy/{dataset_name.replace(" ", "_")}_test_noisy_{noise_label}.csv', index=False)
            
            # ============ 1. RANDOM FOREST ============
            safe_print(f"    [1/4] Random Forest...", end=" ")
            sys.stdout.flush()
            
            if task_type == 'classification':
                rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
            else:
                rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
            
            rf_clean, rf_noisy, rf_time, metric_name = train_and_evaluate_with_noise(
                rf_model, X_train, y_train, X_test, y_test, task_type, noise_type, noise_level
            )
            
            if rf_clean is not None:
                safe_print(f"✅ Clean: {rf_clean:.4f} | Noisy: {rf_noisy:.4f} | Time: {rf_time:.2f}s")
                
                import joblib
                model_filename = f'{output_dir}/models/{dataset_name.replace(" ", "_")}_RF_noise{noise_label}.joblib'
                joblib.dump(rf_model, model_filename)
                
                results.append({
                    'Dataset': dataset_name,
                    'Algorithm': 'Random Forest (Baseline)',
                    'Noise_Level': noise_label,
                    'Score_Clean': rf_clean,
                    'Score_Noisy': rf_noisy,
                    'Score_Degradation': rf_clean - rf_noisy if task_type == 'classification' else rf_noisy - rf_clean,
                    'Metric': metric_name,
                    'Avg_Time_Sec': rf_time,
                    'Type': task_type
                })
            
            # ============ 2. EXTRA TREES ============
            safe_print(f"    [2/4] Extra Trees...", end=" ")
            sys.stdout.flush()
            
            if task_type == 'classification':
                et_model = ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
            else:
                et_model = ExtraTreesRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
            
            et_clean, et_noisy, et_time, metric_name = train_and_evaluate_with_noise(
                et_model, X_train, y_train, X_test, y_test, task_type, noise_type, noise_level
            )
            
            if et_clean is not None:
                safe_print(f"✅ Clean: {et_clean:.4f} | Noisy: {et_noisy:.4f} | Time: {et_time:.2f}s")
                
                import joblib
                model_filename = f'{output_dir}/models/{dataset_name.replace(" ", "_")}_ET_noise{noise_label}.joblib'
                joblib.dump(et_model, model_filename)
                
                results.append({
                    'Dataset': dataset_name,
                    'Algorithm': 'Extra Trees (ET)',
                    'Noise_Level': noise_label,
                    'Score_Clean': et_clean,
                    'Score_Noisy': et_noisy,
                    'Score_Degradation': et_clean - et_noisy if task_type == 'classification' else et_noisy - et_clean,
                    'Metric': metric_name,
                    'Avg_Time_Sec': et_time,
                    'Type': task_type
                })
            
            # ============ 3. RLT AGGRESSIVE ============
            safe_print(f"    [3/4] RLT Aggressive...", end=" ")
            sys.stdout.flush()
            
            if task_type == 'classification':
                rlt_aggressive = SafeRLTWrapper(ReinforcementLearningTree(
                    n_estimators=100, max_depth=10, muting_threshold=0.8, linear_combination=1
                ))
            else:
                rlt_aggressive = ReinforcementLearningRegressor(
                    n_estimators=100, max_depth=10, muting_threshold=0.8, linear_combination=1
                )
            
            rlt_agg_clean, rlt_agg_noisy, rlt_agg_time, metric_name = train_and_evaluate_with_noise(
                rlt_aggressive, X_train, y_train, X_test, y_test, task_type, noise_type, noise_level
            )
            
            if rlt_agg_clean is not None:
                safe_print(f"✅ Clean: {rlt_agg_clean:.4f} | Noisy: {rlt_agg_noisy:.4f} | Time: {rlt_agg_time:.2f}s")
                
                import joblib
                model_filename = f'{output_dir}/models/{dataset_name.replace(" ", "_")}_RLT_Agg_noise{noise_label}.joblib'
                joblib.dump(rlt_aggressive, model_filename)
                
                results.append({
                    'Dataset': dataset_name,
                    'Algorithm': 'RLT Aggressive (muting=80%)',
                    'Noise_Level': noise_label,
                    'Score_Clean': rlt_agg_clean,
                    'Score_Noisy': rlt_agg_noisy,
                    'Score_Degradation': rlt_agg_clean - rlt_agg_noisy if task_type == 'classification' else rlt_agg_noisy - rlt_agg_clean,
                    'Metric': metric_name,
                    'Avg_Time_Sec': rlt_agg_time,
                    'Type': task_type
                })
            
            # ============ 4. RLT LINEAR ============
            safe_print(f"    [4/4] RLT Linear...", end=" ")
            sys.stdout.flush()
            
            if task_type == 'classification':
                rlt_linear = SafeRLTWrapper(ReinforcementLearningTree(
                    n_estimators=100, max_depth=10, muting_threshold=0.0, linear_combination=2
                ))
            else:
                rlt_linear = ReinforcementLearningRegressor(
                    n_estimators=100, max_depth=10, muting_threshold=0.0, linear_combination=2
                )
            
            rlt_lin_clean, rlt_lin_noisy, rlt_lin_time, metric_name = train_and_evaluate_with_noise(
                rlt_linear, X_train, y_train, X_test, y_test, task_type, noise_type, noise_level
            )
            
            if rlt_lin_clean is not None:
                safe_print(f"✅ Clean: {rlt_lin_clean:.4f} | Noisy: {rlt_lin_noisy:.4f} | Time: {rlt_lin_time:.2f}s")
                
                import joblib
                model_filename = f'{output_dir}/models/{dataset_name.replace(" ", "_")}_RLT_Lin_noise{noise_label}.joblib'
                joblib.dump(rlt_linear, model_filename)
                
                results.append({
                    'Dataset': dataset_name,
                    'Algorithm': 'RLT Linear (k=2)',
                    'Noise_Level': noise_label,
                    'Score_Clean': rlt_lin_clean,
                    'Score_Noisy': rlt_lin_noisy,
                    'Score_Degradation': rlt_lin_clean - rlt_lin_noisy if task_type == 'classification' else rlt_lin_noisy - rlt_lin_clean,
                    'Metric': metric_name,
                    'Avg_Time_Sec': rlt_lin_time,
                    'Type': task_type
                })
    
    # ============ SAVE RESULTS ============
    safe_print(f"\n\n{'='*100}")
    safe_print("💾 SAVING RESULTS")
    safe_print(f"{'='*100}\n")
    
    results_df = pd.DataFrame(results)
    
    # Save complete detailed results
    output_file_detailed = f'{output_dir}/results/4algorithms_detailed_results.csv'
    results_df.to_csv(output_file_detailed, index=False)
    safe_print(f"✅ Detailed results saved to: {output_file_detailed}")
    
    # Create simplified format
    simplified = []
    for dataset in results_df['Dataset'].unique():
        for algo in results_df['Algorithm'].unique():
            subset = results_df[(results_df['Dataset'] == dataset) & (results_df['Algorithm'] == algo)]
            
            noisy_scores = subset[subset['Noise_Level'] != '0%']['Score_Noisy']
            if len(noisy_scores) > 0:
                avg_score = noisy_scores.mean()
                avg_time = subset['Avg_Time_Sec'].mean()
                
                simplified.append({
                    'Dataset': dataset,
                    'Algorithm': algo,
                    'Score': avg_score,
                    'Metric': subset['Metric'].iloc[0],
                    'Avg_Time_Sec': avg_time,
                    'Type': subset['Type'].iloc[0]
                })
    
    simplified_df = pd.DataFrame(simplified)
    output_file_simplified = f'{output_dir}/results/4algorithms_simplified_results.csv'
    simplified_df.to_csv(output_file_simplified, index=False)
    safe_print(f"✅ Simplified results saved to: {output_file_simplified}")
    
    # Save summary statistics
    summary_stats = []
    for algo in results_df['Algorithm'].unique():
        algo_data = results_df[results_df['Algorithm'] == algo]
        summary_stats.append({
            'Algorithm': algo,
            'Avg_Degradation': algo_data['Score_Degradation'].mean(),
            'Avg_Training_Time': algo_data['Avg_Time_Sec'].mean(),
            'Num_Experiments': len(algo_data)
        })
    
    summary_df = pd.DataFrame(summary_stats)
    output_file_summary = f'{output_dir}/results/algorithms_summary_statistics.csv'
    summary_df.to_csv(output_file_summary, index=False)
    safe_print(f"✅ Summary statistics saved to: {output_file_summary}")
    
    safe_print(f"\n{'='*100}")
    safe_print("📁 OUTPUT DIRECTORY STRUCTURE")
    safe_print(f"{'='*100}")
    safe_print(f"{output_dir}/")
    safe_print(f"  ├── results/")
    safe_print(f"  │   ├── 4algorithms_detailed_results.csv")
    safe_print(f"  │   ├── 4algorithms_simplified_results.csv")
    safe_print(f"  │   └── algorithms_summary_statistics.csv")
    safe_print(f"  ├── models/")
    safe_print(f"  │   └── [All trained models .joblib files]")
    safe_print(f"  └── datasets_noisy/")
    safe_print(f"      └── [Train, test clean, and test noisy datasets]")
    safe_print(f"{'='*100}\n")
    
    safe_print("📊 DETAILED RESULTS")
    safe_print("="*100)
    safe_print(results_df.to_string(index=False))
    
    safe_print(f"\n\n📊 SIMPLIFIED RESULTS")
    safe_print("="*100)
    safe_print(simplified_df.to_string(index=False))
    
    safe_print(f"\n\n📈 SUMMARY STATISTICS")
    safe_print("="*100)
    safe_print(summary_df.to_string(index=False))
    safe_print(f"{'='*100}\n")
    
    return results_df, simplified_df, summary_df

if __name__ == "__main__":
    results = run_comparison_with_noise(
        noise_levels=[0.0, 0.1, 0.2, 0.3, 0.5],
        noise_type='gaussian'
    )