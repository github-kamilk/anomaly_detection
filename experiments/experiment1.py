#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
experiment1.py
--------------
Script for conducting Experiment 1:
 - Loads data from .npz files for each dataset and scenario.
 - Trains models (supervised + semi-supervised).
 - Saves results (e.g., score/proba) into separate .npz files.

Assumptions:
 - .npz files are named in the format: E1_{dataset}_data.npz
 - Each file contains:
     X_test, y_test
     X_train_A, y_train_A,
     X_train_B, y_train_B, ... (for scenarios A, B, C, D, E)
"""

import sys
from utils.utils import (
    setup_logging, SOMEHOW_SUPERVISED, 
    run_experiment, timer, log_system_info, get_model_configs
)
from utils.prepare_data import prepare_data_e1

##############################################################################
# 1. Experiment-specific constants
##############################################################################

DATASETS = ["NSL-KDD", "CreditCard", "AnnThyroid", "EmailSpam-bert"]
SCENARIOS = ["A", "B", "C", "D", "E"]

# Input folder with .npz files
DATA_FOLDER = "datasets/E1"
# Output folder for results
RESULTS_FOLDER = "results_e1_model_42_dataset_random"
EXPERIMENT_NUM = 1
SEED = 42

##############################################################################
# 2. Experiment-specific filtering logic
##############################################################################

def experiment1_filter(scenario, model_name):
    """Specific filtering logic for Experiment 1
    In scenario A, only use unsupervised models
    In other scenarios, only use semi-supervised models
    """
    if scenario == 'A' and model_name in SOMEHOW_SUPERVISED:
        return False
    if scenario != 'A' and model_name not in SOMEHOW_SUPERVISED:
        return False
    return True

##############################################################################
# 3. Entry point
##############################################################################

if __name__ == "__main__":
    logger = setup_logging("experiment1", log_to_file=True)
    model_configs = get_model_configs(seed=42)
   
    # Log system information
    log_system_info(EXPERIMENT_NUM, DATASETS, SCENARIOS, model_configs.keys())
    
    try:
        with timer("Complete Experiment 1"):
            run_experiment(
                experiment_num=EXPERIMENT_NUM,
                datasets=DATASETS,
                scenarios=SCENARIOS,
                model_configs=model_configs,
                data_folder=DATA_FOLDER,
                results_folder=RESULTS_FOLDER,
                seed=SEED,
                data_generator_function = prepare_data_e1,
                model_scenario_filter=experiment1_filter
            )
    except Exception as e:
        logger.critical(f"Experiment failed with error: {str(e)}", exc_info=True)
        sys.exit(1)