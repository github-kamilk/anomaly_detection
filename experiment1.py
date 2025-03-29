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
 - Models FTTransformer, FEAWAD, DevNet are imported from an external module or library.
"""

import os
import time
import numpy as np
import random
import logging
from datetime import datetime
from contextlib import contextmanager

from adbench.baseline.FTTransformer.run import FTTransformer
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.vae import VAE
from pyod.models.deep_svdd import DeepSVDD
from adbench.baseline.DAGMM.run import DAGMM
from pyod.models.devnet import DevNet
from pyod.models.so_gaal import SO_GAAL
from pyod.models.lunar import LUNAR
from adbench.baseline.FEAWAD.run import FEAWAD
from adbench.baseline.GANomaly.run import GANomaly


##############################################################################
# 1. Setup logging
##############################################################################

def setup_logging(log_level=logging.INFO, log_to_file=True):
    """
    Set up the logging configuration
    """
    # Create a logs directory if it doesn't exist
    if log_to_file and not os.path.exists("logs"):
        os.makedirs("logs")
    
    # Define the timestamp format for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', 
                                       datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if requested
    if log_to_file:
        file_handler = logging.FileHandler(f"logs/experiment1_{timestamp}.log")
        file_handler.setLevel(log_level)
        file_format = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', 
                                        datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
    
    logging.info("Logging initialized")
    return root_logger


##############################################################################
# 2. Constant definitions and iteration lists
##############################################################################

DATASETS = ["NSL-KDD", "CreditCard", "AnnThyroid", "EmailSpam"]
SCENARIOS = ["A", "B", "C", "D", "E"]

# You can store ready model instances or (name, class) pairs
# in a dictionary for looping:
MODEL_CONFIGS = {
    "FTTransformer": lambda:  FTTransformer(seed=random.randint(0, 2**32 - 1), model_name='FTTransformer'),
    "FEAWAD": lambda: FEAWAD(seed=random.randint(0, 2**32 - 1)),         
    "DevNet": lambda: DevNet(),
    "AE": lambda: AutoEncoder(),
    "VAE": lambda: VAE(),
    "DeepSVDD": lambda: DeepSVDD(),
    "DAGMM": lambda: DAGMM(seed=random.randint(0, 2**32 - 1)),
    "SO_GAAL": lambda: SO_GAAL(),
    "LUNAR": lambda: LUNAR(),
    "GANoma": lambda: GANomaly(seed=random.randint(0, 2**32 - 1))          
}

# Input folder with .npz files
DATA_FOLDER = "datasets/E1"
SEED = 42
# Output folder for results. Will be created if it doesn't exist.
RESULTS_FOLDER = "results_e1"
SOMEHOW_SUPERVISED = ['FTTransformer', 'FEAWAD', 'DevNet']

##############################################################################
# 3. Helper functions
##############################################################################

def get_prediction_score(model_name, model, X_test):
    """Get prediction scores based on model type"""
    if model_name in ['FTTransformer', 'DAGMM', 'FEAWAD', 'GANoma']:
        return model.predict_score(X_test)
    elif model_name in ['AE', 'VAE', 'DeepSVDD', 'DevNet', 'SO_GAAL', 'LUNAR']:
        return model.predict_proba(X_test)[:, 1]


@contextmanager
def timer(task_name):
    """Context manager for timing operations"""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        if elapsed < 60:
            logging.info(f"Task '{task_name}' completed in {elapsed:.2f} seconds")
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            logging.info(f"Task '{task_name}' completed in {minutes} minutes and {seconds:.2f} seconds")


##############################################################################
# 4. Main function for conducting the experiment
##############################################################################

def run_experiment_1():
    """
    Main function – for each dataset and each scenario:
     - loads .npz file
     - trains models
     - saves their results into .npz files
    """
    # Make sure the results folder exists
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)
        logging.info(f"Created results directory: {RESULTS_FOLDER}")
    
    # Track experiment progress
    total_combinations = len(DATASETS) * len(SCENARIOS) * len(MODEL_CONFIGS)
    current_combination = 0
    
    logging.info(f"Starting Experiment 1 with {total_combinations} total model combinations")
    
    # Initialize experiment statistics
    successful_runs = 0
    failed_runs = 0
    skipped_runs = 0
    
    for dataset_name in DATASETS:
        # Path to the .npz file with prepared data
        data_path = os.path.join(DATA_FOLDER, f"E1_{dataset_name}_{SEED}_data.npz")

        if not os.path.isfile(data_path):
            logging.warning(f"File {data_path} does not exist. Skipping dataset: {dataset_name}")
            skipped_runs += len(SCENARIOS) * len(MODEL_CONFIGS)
            continue

        logging.info(f"Loading dataset: {dataset_name}")
        with timer(f"Load dataset {dataset_name}"):
            data = np.load(data_path, allow_pickle=True)

        # Load the common test set
        X_test = data["X_test"]
        y_test = data["y_test"]
        logging.info(f"Test set shape: {X_test.shape}, Anomalies: {np.sum(y_test == 1)}/{len(y_test)} ({np.mean(y_test) * 100:.2f}%)")

        # For each scenario, load the corresponding training sets
        for scenario in SCENARIOS:
            x_train_key = f"X_train_{scenario}"
            y_train_key = f"y_train_{scenario}"

            if x_train_key not in data or y_train_key not in data:
                logging.warning(f"Missing key {x_train_key} or {y_train_key} in file {data_path}. Skipping scenario {scenario}")
                skipped_runs += len(MODEL_CONFIGS)
                continue

            X_train = data[x_train_key]
            y_train = data[y_train_key]

            logging.info(f"Processing dataset: {dataset_name}, scenario: {scenario}")
            logging.info(f"Training set shape: {X_train.shape}, Anomalies: {np.sum(y_train == 1)}/{len(y_train)} ({np.mean(y_train) * 100:.2f}%)")

            ######################################################################
            # 5. Model training and result prediction
            ######################################################################
            for model_name, model_initializer in MODEL_CONFIGS.items():
                if scenario == 'A' and model_name in SOMEHOW_SUPERVISED:
                    continue
                if scenario != 'A' and model_name not in SOMEHOW_SUPERVISED:
                    continue
                current_combination += 1
                progress = (current_combination / total_combinations) * 100
                
                logging.info(f"Progress: {current_combination}/{total_combinations} ({progress:.1f}%) - Training model: {model_name}")
                
                try:
                    with timer(f"Initialize {model_name}"):
                        model = model_initializer()
                    
                    # Fit the model to the data
                    with timer(f"Train {model_name} on {dataset_name} scenario {scenario}"):
                        model.fit(X_train, y_train)
                    
                    with timer(f"Predict with {model_name}"):
                        scores = get_prediction_score(model_name, model, X_test)  

                    if scores is None:
                        logging.warning(f"Model '{model_name}' did not return results. Skipping save.")
                        failed_runs += 1
                        continue

                    # Prepare output filename
                    results_filename = f"E1_{dataset_name}_{scenario}_{model_name}_results.npz"
                    results_path = os.path.join(RESULTS_FOLDER, results_filename)

                    # Save: scores and y_test (can be used later for metrics)
                    np.savez(
                        results_path,
                        scores=scores,
                        y_test=y_test
                    )

                    # Calculate and log some basic metrics for immediate feedback
                    anomaly_scores_stats = {
                        'min': np.min(scores),
                        'max': np.max(scores),
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                    }
                    
                    logging.info(f"Results saved to: {results_path}")
                    logging.info(f"Score statistics - min: {anomaly_scores_stats['min']:.4f}, max: {anomaly_scores_stats['max']:.4f}, " 
                                 f"mean: {anomaly_scores_stats['mean']:.4f}, std: {anomaly_scores_stats['std']:.4f}")
                    
                    successful_runs += 1
                    
                except Exception as e:
                    logging.error(f"Error processing {model_name} on {dataset_name} scenario {scenario}: {str(e)}", exc_info=True)
                    failed_runs += 1
                    continue

    # Log experiment summary
    logging.info("=" * 80)
    logging.info("Experiment 1 Completed")
    logging.info(f"Total combinations: {total_combinations}")
    logging.info(f"Successful runs: {successful_runs}")
    logging.info(f"Failed runs: {failed_runs}")
    logging.info(f"Skipped runs: {skipped_runs}")
    logging.info("=" * 80)


##############################################################################
# 6. Entry point
##############################################################################

if __name__ == "__main__":
    logger = setup_logging(log_level=logging.INFO, log_to_file=True)
    
    # Log system information
    import platform
    import sys
    
    logging.info("=" * 80)
    logging.info(f"Starting experiment at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Platform: {platform.platform()}")
    logging.info(f"Datasets: {', '.join(DATASETS)}")
    logging.info(f"Scenarios: {', '.join(SCENARIOS)}")
    logging.info(f"Models: {', '.join(MODEL_CONFIGS.keys())}")
    logging.info("=" * 80)
    
    try:
        with timer("Complete Experiment 1"):
            run_experiment_1()
    except Exception as e:
        logging.critical(f"Experiment failed with error: {str(e)}", exc_info=True)
        sys.exit(1)