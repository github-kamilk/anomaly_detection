#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
experiment2.py
--------------
Script for conducting Experiment 2:
 - Loads data from .npz files for each dataset and scenario.
 - Trains models (supervised + semi-supervised).
 - Saves results (e.g., score/proba) into separate .npz files.

Assumptions:
 - .npz files are named in the format: E2_{dataset}_data.npz
 - Each file contains:
     X_test, y_test
     X_train_A, y_train_A,
     X_train_B, y_train_B, ... (for scenarios A, B, C, D)
"""

import sys
from utils.utils import (
    setup_logging, run_experiment, 
    timer, log_system_info, get_model_configs
)
from utils.prepare_data import prepare_data_e2

##############################################################################
# 1. Experiment-specific constants
##############################################################################

DATASETS = ["NSL-KDD", "CreditCard", "AnnThyroid", "EmailSpam-bert"]
SCENARIOS = ["A", "B", "C", "D"]

# Input folder with .npz files
DATA_FOLDER = "datasets/E2"
# Output folder for results
RESULTS_FOLDER = "results_e2"
EXPERIMENT_NUM = 2
SEED = 42

##############################################################################
# 2. Entry point
##############################################################################

if __name__ == "__main__":
    logger = setup_logging("experiment2", log_to_file=True)
    model_configs = get_model_configs(seed=None)
    # Log system information
    log_system_info(EXPERIMENT_NUM, DATASETS, SCENARIOS, model_configs.keys())
    
    try:
        with timer("Complete Experiment 2"):
            run_experiment(
                experiment_num=EXPERIMENT_NUM,
                datasets=DATASETS,
                scenarios=SCENARIOS,
                model_configs=model_configs,
                data_folder=DATA_FOLDER,
                results_folder=RESULTS_FOLDER,
                data_generator_function = False,
                seed=SEED
                # No filter for experiment 2 - run all models in all scenarios
            )
    except Exception as e:
        logger.critical(f"Experiment failed with error: {str(e)}", exc_info=True)
        sys.exit(1)