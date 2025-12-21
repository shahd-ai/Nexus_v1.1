#!/usr/bin/env python3
"""
================================================================================
RLT PROJECT - MAIN ORCHESTRATOR WITH MLFLOW
================================================================================
Central execution hub for all RLT pipeline components with MLflow tracking.

Features:
  - Menu-driven interface
  - Automated MLflow experiment tracking
  - Complete audit trail of all operations
  - Performance metrics logging
  - Model artifact storage

Usage:
    python main.py                  # Interactive menu
    python main.py --workflow eda   # Run specific workflow
    python main.py --help           # Show all options
================================================================================
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List
import time
import mlflow

from mlflow_config import MLflowTracker, MLflowConfig

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(text: str):
    """Print colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.ENDC}\n")

def print_section(text: str):
    """Print colored section"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}► {text}{Colors.ENDC}")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")

def check_dependencies():
    """Verify all required modules are installed"""
    print_section("Checking dependencies...")
    
    required_modules = [
        'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn',
        'PIL', 'scipy', 'streamlit', 'mlflow'
    ]
    
    missing = []
    for module in required_modules:
        try:
            __import__(module)
            print_success(f"{module} is installed")
        except ImportError:
            print_error(f"{module} is NOT installed")
            missing.append(module)
    
    if missing:
        print_error(f"\nMissing modules: {', '.join(missing)}")
        print_warning("Run: pip install -r requirements.txt")
        return False
    
    print_success("\nAll dependencies verified!\n")
    return True

def check_files():
    """Verify all required Python files exist"""
    print_section("Checking required files...")
    
    required_files = [
        'understanding_data.py',
        'preparation_data.py',
        'rlt_module.py',
        'Enhanced_rlt.py',
        'run_rlt.py',
        'rlt_simulation_testing.py',
        'rlt_streamlit_app/app.py',
        'mlflow_config.py'
    ]
    
    missing = []
    for file in required_files:
        if Path(file).exists():
            print_success(f"{file}")
        else:
            print_error(f"{file} NOT found")
            missing.append(file)
    
    if missing:
        print_error(f"\nMissing files: {', '.join(missing)}")
        return False
    
    print_success("\nAll files present!\n")
    return True

def run_script(script_path: str, description: str, experiment_name: str) -> bool:
    """Execute a Python script with MLflow tracking"""
    print_section(f"Executing: {description}")
    print(f"Script: {script_path}")
    print("-" * 80)
    
    try:
        # IMPORTANT: End any active run first
        mlflow.end_run()
        
        # Initialize MLflow tracking
        MLflowConfig.init_mlflow()
        tracker = MLflowTracker(experiment_name)
        tracker.start_run(run_name=description)
        
        # Log environment info
        tracker.log_params({
            'script': script_path,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'python_version': sys.version.split()[0],
            'working_directory': os.getcwd()
        })
        
        start_time = time.time()
        result = subprocess.run([sys.executable, script_path], check=True)
        elapsed = time.time() - start_time
        
        # Log execution metrics
        tracker.log_metrics({
            'execution_time_seconds': elapsed,
            'status': 1  # Success
        })
        
        tracker.end_run(status="FINISHED")
        print_success(f"{description} completed in {elapsed:.2f}s\n")
        return True
        
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed with error code {e.returncode}\n")
        try:
            mlflow.end_run(status="FAILED")
        except:
            pass
        return False
    except Exception as e:
        print_error(f"{description} error: {str(e)}\n")
        try:
            mlflow.end_run(status="FAILED")
        except:
            pass
        return False

def run_streamlit():
    """Launch Streamlit app with MLflow tracking"""
    print_section("Launching Streamlit Application")
    print("-" * 80)
    
    streamlit_path = Path('rlt_streamlit_app/app.py')
    if not streamlit_path.exists():
        print_error("Streamlit app not found at rlt_streamlit_app/app.py")
        return False
    
    try:
        # Initialize MLflow tracking
        MLflowConfig.init_mlflow()
        tracker = MLflowTracker(MLflowConfig.EXPERIMENTS['deployment'])
        tracker.start_run(run_name="Streamlit-Application")
        
        tracker.log_params({
            'app_type': 'streamlit',
            'port': 8501,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        print_success("Streamlit is starting...")
        print(f"{Colors.YELLOW}📱 Open your browser and navigate to: http://localhost:8501{Colors.ENDC}")
        print(f"{Colors.YELLOW}Press Ctrl+C to stop the server{Colors.ENDC}\n")
        
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run',
            str(streamlit_path),
            '--logger.level=error'
        ])
        
        tracker.end_run(status="FINISHED")
        return True
    except KeyboardInterrupt:
        print_success("\nStreamlit server stopped")
        try:
            tracker.end_run(status="FINISHED")
        except:
            pass
        return True
    except Exception as e:
        print_error(f"Failed to launch Streamlit: {str(e)}")
        try:
            tracker.end_run(status="FAILED")
        except:
            pass
        return False

# ============================================================================
# WORKFLOW DEFINITIONS
# ============================================================================

WORKFLOWS = {
    'check': {
        'name': 'System Check',
        'description': 'Verify dependencies and files',
        'steps': [],
        'experiment': None
    },
    'eda': {
        'name': 'Exploratory Data Analysis',
        'description': 'Analyze datasets and generate visualizations',
        'steps': ['understanding_data.py'],
        'experiment': MLflowConfig.EXPERIMENTS['eda']
    },
    'prep': {
        'name': 'Data Preparation',
        'description': 'Clean and impute missing values',
        'steps': ['preparation_data.py'],
        'experiment': MLflowConfig.EXPERIMENTS['preparation']
    },
    'train': {
        'name': 'Basic Training',
        'description': 'Train RLT models on sample data',
        'steps': ['run_rlt.py'],
        'experiment': MLflowConfig.EXPERIMENTS['training']
    },
    'advanced': {
        'name': 'Advanced Training',
        'description': 'Multi-strategy RLT training and comparison',
        'steps': ['run_rlt.py'],
        'experiment': MLflowConfig.EXPERIMENTS['advanced']
    },
    'research': {
        'name': 'Research Benchmarking',
        'description': 'Generate comprehensive benchmark results',
        'steps': ['rlt_simulation_testing.py'],
        'experiment': MLflowConfig.EXPERIMENTS['research']
    },
    'streamlit': {
        'name': 'Interactive Web App',
        'description': 'Launch interactive Streamlit application',
        'steps': ['streamlit'],
        'experiment': MLflowConfig.EXPERIMENTS['deployment']
    },
    'full': {
        'name': 'Complete Pipeline',
        'description': 'Run all analyses and training in sequence',
        'steps': [
            'understanding_data.py',
            'run_rlt.py',
            'rlt_simulation_testing.py'
        ],
        'experiment': 'RLT-Complete-Pipeline'
    }
}

def display_menu():
    """Display interactive menu"""
    print_header("RLT PROJECT - EXECUTION MENU")
    print(f"{Colors.BOLD}Available Workflows:{Colors.ENDC}\n")
    
    for idx, (key, workflow) in enumerate(WORKFLOWS.items(), 1):
        print(f"{Colors.BLUE}{idx}. {workflow['name']:<30}{Colors.ENDC}")
        print(f"   └─ {workflow['description']}\n")
    
    print(f"{Colors.BLUE}0. Exit{Colors.ENDC}\n")

def interactive_menu():
    """Interactive menu loop"""
    while True:
        display_menu()
        
        try:
            choice = input(f"{Colors.YELLOW}Select workflow (0-{len(WORKFLOWS)}): {Colors.ENDC}").strip()
            
            if choice == '0':
                print_success("Exiting...")
                break
            
            workflow_keys = list(WORKFLOWS.keys())
            if choice.isdigit() and 1 <= int(choice) <= len(WORKFLOWS):
                workflow_key = workflow_keys[int(choice) - 1]
                execute_workflow(workflow_key)
            else:
                print_error("Invalid choice. Please try again.")
        
        except KeyboardInterrupt:
            print_error("\nInterrupted by user")
            break
        except Exception as e:
            print_error(f"Error: {str(e)}")

def execute_workflow(workflow_key: str) -> bool:
    """Execute a complete workflow with MLflow tracking"""
    if workflow_key not in WORKFLOWS:
        print_error(f"Unknown workflow: {workflow_key}")
        return False
    
    workflow = WORKFLOWS[workflow_key]
    print_header(workflow['name'])
    
    # Initialize MLflow for this workflow
    workflow_tracker = None
    if workflow.get('experiment'):
        MLflowConfig.init_mlflow()
        workflow_tracker = MLflowTracker(workflow['experiment'])
        workflow_tracker.start_run(run_name=workflow['name'])
        
        # Log workflow metadata
        workflow_tracker.log_params({
            'workflow': workflow_key,
            'description': workflow['description'],
            'steps_count': len(workflow['steps'])
        })
    
    # Check system first
    if not check_dependencies():
        if workflow_tracker:
            workflow_tracker.end_run(status="FAILED")
        return False
    if not check_files():
        if workflow_tracker:
            workflow_tracker.end_run(status="FAILED")
        return False
    
    # END workflow tracker before running scripts
    if workflow_tracker:
        workflow_tracker.end_run(status="IN_PROGRESS")
    
    # Execute workflow steps
    if workflow_key == 'check':
        return True
    
    for step in workflow['steps']:
        if step == 'streamlit':
            if not run_streamlit():
                return False
        else:
            experiment_name = workflow.get('experiment', 'RLT-Default')
            if not run_script(step, step, experiment_name):
                print_warning(f"Continuing despite {step} issue...")
    
    # Final completion message
    print_header("✓ Workflow Completed Successfully!")
    return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='RLT Project - Central Execution Hub',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Interactive menu
  python main.py --workflow eda     # Run EDA workflow
  python main.py --workflow train   # Run training
  python main.py --workflow full    # Run complete pipeline
  python main.py --list             # List all workflows
        """
    )
    
    parser.add_argument('--workflow', type=str, choices=list(WORKFLOWS.keys()),
                        help='Execute specific workflow')
    parser.add_argument('--list', action='store_true',
                        help='List all available workflows')
    parser.add_argument('--check', action='store_true',
                        help='Check system setup only')
    
    args = parser.parse_args()
    
    # List workflows
    if args.list:
        print_header("Available Workflows")
        for key, workflow in WORKFLOWS.items():
            print(f"{Colors.BOLD}{key:<12}{Colors.ENDC} - {workflow['name']:<30}")
            print(f"{'':12}   {workflow['description']}\n")
        return
    
    # Check system
    if args.check:
        print_header("System Check")
        check_dependencies()
        check_files()
        return
    
    # Execute specific workflow
    if args.workflow:
        success = execute_workflow(args.workflow)
        sys.exit(0 if success else 1)
    
    # Interactive menu (default)
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print_error("\nExiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()
