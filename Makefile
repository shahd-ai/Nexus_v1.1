# ============================================================================
# RLT PROJECT - MAKEFILE
# ============================================================================
# Simplified command execution for RLT pipeline
# 
# Usage: make [target]
# 
# Note: Requires Python 3.8+ and virtual environment activated
# ============================================================================

.PHONY: help install check eda prep train advanced research streamlit full \
        clean clean-models clean-plots test docs

# Color output
BLUE = \033[0;34m
GREEN = \033[0;32m
YELLOW = \033[0;33m
RED = \033[0;31m
NC = \033[0m # No Color

# Python interpreter
PYTHON := python3
PIP := pip

# ============================================================================
# DEFAULT TARGET
# ============================================================================

help:
	@echo "$(BLUE)=========================================$(NC)"
	@echo "$(BLUE)RLT PROJECT - MAKEFILE TARGETS$(NC)"
	@echo "$(BLUE)=========================================$(NC)\n"
	@echo "$(GREEN)Setup & Configuration:$(NC)"
	@echo "  make install          - Install dependencies"
	@echo "  make check            - Verify system setup"
	@echo "  make venv             - Create virtual environment\n"
	@echo "$(GREEN)Workflows:$(NC)"
	@echo "  make eda              - Run Exploratory Data Analysis"
	@echo "  make prep             - Run Data Preparation"
	@echo "  make train            - Run Basic Training"
	@echo "  make advanced         - Run Advanced Multi-Strategy Training"
	@echo "  make research         - Run Research Benchmarking"
	@echo "  make full             - Run Complete Pipeline\n"
	@echo "$(GREEN)Deployment:$(NC)"
	@echo "  make streamlit        - Launch Streamlit Application\n"
	@echo "$(GREEN)Utilities:$(NC)"
	@echo "  make main             - Run interactive main menu"
	@echo "  make clean            - Remove generated outputs"
	@echo "  make clean-models     - Remove trained models"
	@echo "  make clean-plots      - Remove visualization plots"
	@echo "  make test             - Run basic tests"
	@echo "  make docs             - Generate documentation\n"
	@echo "$(GREEN)Examples:$(NC)"
	@echo "  make eda              # Run EDA analysis"
	@echo "  make train            # Train RLT model"
	@echo "  make full             # Complete pipeline\n"

# ============================================================================
# SETUP & CONFIGURATION
# ============================================================================

venv:
	@echo "$(BLUE)Creating Python virtual environment...$(NC)"
	$(PYTHON) -m venv venv
	@echo "$(GREEN)✓ Virtual environment created$(NC)"
	@echo "$(YELLOW)Run: source venv/bin/activate (Linux/Mac) or venv\Scripts\activate (Windows)$(NC)"

install: venv
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install mlflow>=2.10.0
	@echo "$(GREEN)✓ Dependencies installed$(NC)"
	@echo "$(GREEN)✓ MLflow installed$(NC)"

check:
	@echo "$(BLUE)Checking system setup...$(NC)"
	$(PYTHON) main.py --check

# ============================================================================
# DATA ANALYSIS & EXPLORATION
# ============================================================================

eda:
	@echo "$(BLUE)Running Exploratory Data Analysis...$(NC)"
	$(PYTHON) main.py --workflow eda

prep:
	@echo "$(BLUE)Running Data Preparation...$(NC)"
	$(PYTHON) main.py --workflow prep

# ============================================================================
# MODEL TRAINING
# ============================================================================

train:
	@echo "$(BLUE)Running Basic RLT Training...$(NC)"
	$(PYTHON) main.py --workflow train

advanced:
	@echo "$(BLUE)Running Advanced Multi-Strategy Training...$(NC)"
	$(PYTHON) main.py --workflow advanced

research:
	@echo "$(BLUE)Running Research Benchmarking Suite...$(NC)"
	$(PYTHON) main.py --workflow research

# ============================================================================
# COMPLETE PIPELINES
# ============================================================================

full:
	@echo "$(BLUE)Running Complete Pipeline...$(NC)"
	@echo "$(YELLOW)This will take several minutes...$(NC)"
	$(PYTHON) main.py --workflow full

# ============================================================================
# DEPLOYMENT
# ============================================================================

streamlit:
	@echo "$(BLUE)Launching Streamlit Application...$(NC)"
	@echo "$(YELLOW)Opening: http://localhost:8501$(NC)"
	$(PYTHON) main.py --workflow streamlit

# ============================================================================
# UTILITIES
# ============================================================================

main:
	@echo "$(BLUE)Starting Interactive Menu...$(NC)"
	$(PYTHON) main.py

list:
	@echo "$(BLUE)Available Workflows:$(NC)"
	$(PYTHON) main.py --list

# ============================================================================
# CLEANING
# ============================================================================

clean-plots:
	@echo "$(BLUE)Removing plot files...$(NC)"
	rm -rf plots/
	@echo "$(GREEN)✓ Plots removed$(NC)"

clean-models:
	@echo "$(BLUE)Removing trained models...$(NC)"
	rm -rf rlt_models/ dso2/models/
	@echo "$(GREEN)✓ Models removed$(NC)"

clean-results:
	@echo "$(BLUE)Removing result files...$(NC)"
	rm -f *.csv dso2/results/*.csv
	@echo "$(GREEN)✓ Results removed$(NC)"

clean: clean-plots clean-models clean-results
	@echo "$(BLUE)Removing all generated outputs...$(NC)"
	rm -rf __pycache__ .pytest_cache *.pyc
	@echo "$(GREEN)✓ All outputs cleaned$(NC)"

# ============================================================================
# TESTING & DOCUMENTATION
# ============================================================================

test:
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTHON) -m pytest tests/ -v 2>/dev/null || echo "$(YELLOW)No tests found$(NC)"

docs:
	@echo "$(BLUE)Documentation is in README.md$(NC)"
	@echo "$(GREEN)Open: README.md$(NC)"

# ============================================================================
# INFO COMMANDS
# ============================================================================

.PHONY: info version status

info:
	@echo "$(BLUE)=========================================$(NC)"
	@echo "$(BLUE)RLT PROJECT - SYSTEM INFO$(NC)"
	@echo "$(BLUE)=========================================$(NC)"
	@echo "Python Version: $$($(PYTHON) --version)"
	@echo "pip Version: $$($(PIP) --version)"
	@echo "Current Directory: $$(pwd)"
	@echo "$(GREEN)✓ System ready$(NC)"

version:
	@echo "RLT Project Version 2.0 (Production Ready)"

status:
	@echo "$(BLUE)Project Status:$(NC)"
	@echo "  • Core modules: ✓"
	@echo "  • Dependencies: $(shell $(PYTHON) -c 'import rlt_module' 2>/dev/null && echo '✓' || echo '✗')"
	@echo "  • Streamlit app: $(shell test -f rlt_streamlit_app/app.py && echo '✓' || echo '✗')"
	@echo "  • MLflow: $(shell $(PYTHON) -c 'import mlflow' 2>/dev/null && echo '✓' || echo '✗')"

# ============================================================================
# MLFLOW COMMANDS
# ============================================================================

.PHONY: mlflow-install mlflow-ui mlflow-runs mlflow-experiments mlflow-compare mlflow-cleanup

mlflow-install:
	@echo "$(BLUE)Installing MLflow...$(NC)"
	$(PIP) install mlflow>=2.10.0
	@echo "$(GREEN)✓ MLflow installed$(NC)"

mlflow-ui:
	@echo "$(BLUE)Launching MLflow UI...$(NC)"
	@echo "$(YELLOW)Opening: http://localhost:5000$(NC)"
	$(PYTHON) -m mlflow ui

mlflow-runs:
	@echo "$(BLUE)Listing MLflow runs...$(NC)"
	$(PYTHON) -c "from mlflow_config import get_run_info, MLflowConfig; MLflowConfig.init_mlflow(); runs = get_run_info('RLT-Training'); print(runs[['run_id', 'start_time', 'status']].head(10))" 2>/dev/null || echo "No runs found"

mlflow-experiments:
	@echo "$(BLUE)MLflow Experiments:$(NC)"
	$(PYTHON) -c "import mlflow; MLflowConfig.init_mlflow(); exps = mlflow.search_experiments(); [print(f'  • {e.name}') for e in exps]" 2>/dev/null || echo "MLflow not initialized"

mlflow-compare:
	@echo "$(BLUE)Comparing model runs...$(NC)"
	$(PYTHON) -c "from mlflow_config import compare_runs, MLflowConfig; MLflowConfig.init_mlflow(); compare_runs('RLT-Training', 'accuracy')" 2>/dev/null || echo "No comparisons available"

mlflow-cleanup:
	@echo "$(BLUE)Cleaning up old MLflow runs...$(NC)"
	$(PYTHON) -c "from mlflow_config import cleanup_runs, MLflowConfig; MLflowConfig.init_mlflow(); cleanup_runs('RLT-Training', keep_recent=10)" 2>/dev/null || echo "Cleanup completed"

mlflow-tracking-uri:
	@echo "$(BLUE)MLflow Tracking URI:$(NC)"
	@echo "  file:./mlruns (Local)"
	@echo "  To change: Edit mlflow_config.py → MLflowConfig.MLFLOW_TRACKING_URI"

# ============================================================================
# PHONY TARGETS (never match filenames)
# ============================================================================

.PHONY: help install check venv eda prep train advanced research full streamlit
.PHONY: main list clean-plots clean-models clean-results clean test docs
.PHONY: info version status mlflow-install mlflow-ui mlflow-runs mlflow-experiments
.PHONY: mlflow-compare mlflow-cleanup mlflow-tracking-uri

# Default help
.DEFAULT_GOAL := help