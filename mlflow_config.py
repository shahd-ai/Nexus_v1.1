#!/usr/bin/env python3
"""
================================================================================
MLFLOW CONFIGURATION & INTEGRATION
================================================================================
Centralized MLflow setup for experiment tracking, model logging, and monitoring.

Features:
  - Automatic experiment creation
  - Run tracking for all operations
  - Metrics & parameters logging
  - Model artifact storage
  - Hyperparameter tracking
  - Performance comparison
================================================================================
"""

import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# MLflow Configuration
# ============================================================================

class MLflowConfig:
    """MLflow configuration and initialization"""
    
    # MLflow tracking server
    MLFLOW_TRACKING_URI = "file:./mlruns"  # Local tracking
    # For remote server: "http://mlflow-server:5000"
    
    # Experiment names
    EXPERIMENTS = {
        'eda': 'RLT-EDA',
        'preparation': 'RLT-Data-Preparation',
        'training': 'RLT-Training',
        'advanced': 'RLT-Advanced-Training',
        'research': 'RLT-Research-Benchmarking',
        'deployment': 'RLT-Deployment'
    }
    
    @staticmethod
    def init_mlflow():
        """Initialize MLflow tracking"""
        mlflow.set_tracking_uri(MLflowConfig.MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI: {MLflowConfig.MLFLOW_TRACKING_URI}")
    
    @staticmethod
    def create_experiment(exp_name: str) -> str:
        """Create or get experiment ID"""
        try:
            exp_id = mlflow.create_experiment(exp_name)
            logger.info(f"Created experiment: {exp_name} (ID: {exp_id})")
            return exp_id
        except Exception:
            # Experiment already exists
            exp = mlflow.get_experiment_by_name(exp_name)
            logger.info(f"Using existing experiment: {exp_name} (ID: {exp.experiment_id})")
            return exp.experiment_id

# ============================================================================
# MLflow Experiment Tracker
# ============================================================================

class MLflowTracker:
    """Track experiments with MLflow"""
    
    def __init__(self, experiment_name: str):
        """Initialize tracker for specific experiment"""
        MLflowConfig.init_mlflow()
        self.experiment_name = experiment_name
        self.exp_id = MLflowConfig.create_experiment(experiment_name)
        self.run = None
        self.active = False
    
    def start_run(self, run_name: str = None, tags: Dict = None):
        """Start MLflow run"""
        mlflow.set_experiment(self.experiment_name)
        self.run = mlflow.start_run(run_name=run_name)
        
        # Log tags
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)
        
        self.active = True
        logger.info(f"Started run: {run_name or 'default'}")
        return self.run
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        if not self.active:
            logger.warning("No active run")
            return
        
        for key, value in params.items():
            mlflow.log_param(key, value)
        logger.info(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics"""
        if not self.active:
            logger.warning("No active run")
            return
        
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                mlflow.log_metric(key, float(value), step=step)
        logger.info(f"Logged {len(metrics)} metrics")
    
    def log_model(self, model, model_name: str, artifact_path: str = "models"):
        """Log model artifact"""
        if not self.active:
            logger.warning("No active run")
            return
        
        try:
            mlflow.sklearn.log_model(model, artifact_path)
            logger.info(f"Logged model: {model_name}")
        except Exception:
            # For custom models
            import pickle
            model_path = f"./temp_{model_name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            mlflow.log_artifact(model_path)
            os.remove(model_path)
            logger.info(f"Logged custom model: {model_name}")
    
    def log_dataset(self, df: pd.DataFrame, dataset_name: str):
        """Log dataset info"""
        if not self.active:
            logger.warning("No active run")
            return
        
        mlflow.log_param(f"dataset_{dataset_name}_shape", str(df.shape))
        mlflow.log_param(f"dataset_{dataset_name}_rows", df.shape[0])
        mlflow.log_param(f"dataset_{dataset_name}_cols", df.shape[1])
        logger.info(f"Logged dataset info: {dataset_name}")
    
    def log_dataframe(self, df: pd.DataFrame, filename: str):
        """Log CSV file"""
        if not self.active:
            logger.warning("No active run")
            return
        
        csv_path = f"./temp_{filename}.csv"
        df.to_csv(csv_path, index=False)
        mlflow.log_artifact(csv_path)
        os.remove(csv_path)
        logger.info(f"Logged dataframe: {filename}")
    
    def log_dict(self, data: Dict, filename: str):
        """Log dictionary as JSON"""
        if not self.active:
            logger.warning("No active run")
            return
        
        json_path = f"./temp_{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        mlflow.log_artifact(json_path)
        os.remove(json_path)
        logger.info(f"Logged dictionary: {filename}")
    
    def log_artifact(self, artifact_path: str):
        """Log arbitrary artifact"""
        if not self.active:
            logger.warning("No active run")
            return
        
        mlflow.log_artifact(artifact_path)
        logger.info(f"Logged artifact: {artifact_path}")
    
    def end_run(self, status: str = "FINISHED"):
        """End MLflow run"""
        if self.active:
            mlflow.end_run()
            self.active = False
            logger.info(f"Ended run with status: {status}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type:
            logger.error(f"Error in run: {exc_type.__name__}: {exc_val}")
            mlflow.end_run(status="FAILED")
        else:
            mlflow.end_run(status="FINISHED")
        self.active = False

# ============================================================================
# Specialized Trackers
# ============================================================================

class EDATracker(MLflowTracker):
    """Track EDA operations"""
    
    def __init__(self):
        super().__init__(MLflowConfig.EXPERIMENTS['eda'])
    
    def log_eda_results(self, dataset_name: str, results: Dict):
        """Log EDA results"""
        self.start_run(run_name=f"EDA-{dataset_name}")
        
        self.log_params({
            'dataset_name': dataset_name,
            'rows': results.get('rows', 0),
            'columns': results.get('columns', 0),
            'numeric_cols': results.get('numeric_cols', 0),
            'categorical_cols': results.get('categorical_cols', 0)
        })
        
        self.log_dict(results, f"eda_results_{dataset_name}")
        logger.info(f"Logged EDA for: {dataset_name}")

class TrainingTracker(MLflowTracker):
    """Track model training"""
    
    def __init__(self, task: str = "training"):
        if task == "training":
            exp_name = MLflowConfig.EXPERIMENTS['training']
        elif task == "advanced":
            exp_name = MLflowConfig.EXPERIMENTS['advanced']
        else:
            exp_name = MLflowConfig.EXPERIMENTS['training']
        
        super().__init__(exp_name)
    
    def log_training(self, model_name: str, params: Dict, metrics: Dict, model=None):
        """Log complete training run"""
        self.start_run(run_name=model_name, tags={'type': 'model_training'})
        
        self.log_params(params)
        self.log_metrics(metrics)
        
        if model:
            self.log_model(model, model_name)
        
        logger.info(f"Logged training: {model_name}")

class ResearchTracker(MLflowTracker):
    """Track research benchmarking"""
    
    def __init__(self):
        super().__init__(MLflowConfig.EXPERIMENTS['research'])
    
    def log_benchmark(self, scenario: str, method: str, results: Dict):
        """Log benchmark results"""
        run_name = f"{scenario}-{method}"
        self.start_run(run_name=run_name)
        
        self.log_params({
            'scenario': scenario,
            'method': method
        })
        
        self.log_metrics(results)
        logger.info(f"Logged benchmark: {run_name}")

# ============================================================================
# Utility Functions
# ============================================================================

def get_run_info(exp_name: str):
    """Get experiment runs information"""
    exp = mlflow.get_experiment_by_name(exp_name)
    if not exp:
        logger.warning(f"Experiment not found: {exp_name}")
        return None
    
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    logger.info(f"Found {len(runs)} runs in {exp_name}")
    return runs

def compare_runs(exp_name: str, metric_name: str):
    """Compare runs by metric"""
    runs = get_run_info(exp_name)
    if runs is None or runs.empty:
        logger.warning("No runs found for comparison")
        return None
    
    comparison = runs[['run_id', 'start_time', f'metrics.{metric_name}']].copy()
    comparison = comparison.sort_values(f'metrics.{metric_name}', ascending=False)
    logger.info(f"Comparison by {metric_name}:\n{comparison}")
    return comparison

def cleanup_runs(exp_name: str, keep_recent: int = 5):
    """Keep only recent runs"""
    runs = get_run_info(exp_name)
    if runs is None or len(runs) <= keep_recent:
        return
    
    old_runs = runs.iloc[keep_recent:]['run_id'].tolist()
    for run_id in old_runs:
        mlflow.delete_run(run_id)
    logger.info(f"Deleted {len(old_runs)} old runs")

# ============================================================================
# Context Managers
# ============================================================================

def track_eda(dataset_name: str):
    """Context manager for EDA tracking"""
    tracker = EDATracker()
    tracker.start_run(run_name=f"EDA-{dataset_name}")
    return tracker

def track_training(model_name: str, task: str = "training"):
    """Context manager for training tracking"""
    tracker = TrainingTracker(task)
    tracker.start_run(run_name=model_name)
    return tracker

def track_research(scenario: str, method: str):
    """Context manager for research tracking"""
    tracker = ResearchTracker()
    tracker.start_run(run_name=f"{scenario}-{method}")
    return tracker

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Simple tracking
    with MLflowTracker('Test-Experiment') as tracker:
        tracker.start_run("test-run")
        tracker.log_params({'param1': 1, 'param2': 2})
        tracker.log_metrics({'metric1': 0.95, 'metric2': 0.87})
        print("✓ Basic tracking complete")
    
    # Example 2: EDA tracking
    eda_tracker = EDATracker()
    eda_tracker.log_eda_results("test_dataset", {
        'rows': 1000,
        'columns': 50,
        'numeric_cols': 40,
        'categorical_cols': 10
    })
    eda_tracker.end_run()
    print("✓ EDA tracking complete")
    
    # Example 3: Training tracking
    train_tracker = TrainingTracker()
    train_tracker.start_run("test-model")
    train_tracker.log_params({'n_estimators': 50, 'max_depth': 8})
    train_tracker.log_metrics({'accuracy': 0.95, 'f1_score': 0.92})
    train_tracker.end_run()
    print("✓ Training tracking complete")
    
    print("\n✓ All MLflow examples executed successfully!")
    print(f"View experiments at: {MLflowConfig.MLFLOW_TRACKING_URI}")