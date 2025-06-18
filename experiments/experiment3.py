#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from utils.utils import (
    setup_logging, SOMEHOW_SUPERVISED, 
    run_experiment, timer, log_system_info, get_model_configs
)

##############################################################################
# 1. Experiment-specific constants
##############################################################################

DATASETS = ["NSL-KDD-local", "CreditCard-local", "AnnThyroid-local", "EmailSpam-bert-local",
            "NSL-KDD-global", "CreditCard-global", "AnnThyroid-global", "EmailSpam-bert-global",
            "NSL-KDD-clustered", "CreditCard-clustered", "AnnThyroid-clustered", "EmailSpam-bert-clustered",
            "CreditCard-dependency", "AnnThyroid-dependency"]
SCENARIOS = ['small', 'medium', 'large', 'extra']

# Input folder with .npz files
DATA_FOLDER = "datasets/E3"
# Output folder for results
RESULTS_FOLDER = "results_e3"
EXPERIMENT_NUM = 3
SEED = 42

##############################################################################
# 2. Experiment-specific filtering logic
##############################################################################

def experiment3_filter(scenario, dataset):
    if scenario == 'extra' and "dependency" in dataset:
        return False
    return True

##############################################################################
# 3. Entry point
##############################################################################

if __name__ == "__main__":
    logger = setup_logging("experiment3", log_to_file=True)
    model_configs = get_model_configs(seed=None)
   
    # Log system information
    log_system_info(EXPERIMENT_NUM, DATASETS, SCENARIOS, model_configs.keys())
    
    try:
        with timer("Complete Experiment 3"):
            run_experiment(
                experiment_num=EXPERIMENT_NUM,
                datasets=DATASETS,
                scenarios=SCENARIOS,
                model_configs=model_configs,
                data_folder=DATA_FOLDER,
                results_folder=RESULTS_FOLDER,
                seed=SEED,
                data_generator_function = False,
                model_scenario_filter=experiment3_filter,
                common_test_set=False
            )
    except Exception as e:
        logger.critical(f"Experiment failed with error: {str(e)}", exc_info=True)
        sys.exit(1)