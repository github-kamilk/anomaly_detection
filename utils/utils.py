#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils.py
--------
Common utilities for anomaly detection experiments.
Contains shared functionality for loading data, model training,
and evaluation to minimize code duplication across experiments.
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
from adbench.baseline.DevNet.run import DevNet
from pyod.models.so_gaal_new import SO_GAAL
from pyod.models.lunar import LUNAR
from adbench.baseline.FEAWAD.run import FEAWAD
from adbench.baseline.GANomaly.run import GANomaly


##############################################################################
# 1. Setup logging
##############################################################################

def setup_logging(experiment_name, log_level=logging.INFO, log_to_file=True):
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
        file_handler = logging.FileHandler(f"logs/{experiment_name}_{timestamp}.log")
        file_handler.setLevel(log_level)
        file_format = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', 
                                        datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
    
    logging.info("Logging initialized")
    return root_logger


##############################################################################
# 2. Model configurations
##############################################################################

# Standard model configurations that can be used across experiments
MODEL_CONFIGS = {
    "FTTransformer": lambda:  FTTransformer(seed=random.randint(0, 2**32 - 1), model_name='FTTransformer'),
    "FEAWAD": lambda: FEAWAD(seed=random.randint(0, 2**32 - 1)),         
    "DevNet": lambda: DevNet(seed=random.randint(0, 2**32 - 1)),
    "AE": lambda: AutoEncoder(random_state=random.randint(0, 2**32 - 1)),
    "VAE": lambda: VAE(random_state=random.randint(0, 2**32 - 1)),
    "DeepSVDD": lambda: DeepSVDD, 
    "DAGMM": lambda: DAGMM(seed=random.randint(0, 2**32 - 1)),
    "SO_GAAL": lambda: SO_GAAL(random_state=random.randint(0, 2**32 - 1)),
    "LUNAR": lambda: LUNAR(),
    "GANoma": lambda: GANomaly(seed=random.randint(0, 2**32 - 1))          
}

SOMEHOW_SUPERVISED = ['FTTransformer', 'FEAWAD', 'DevNet']


##############################################################################
# 3. Helper functions
##############################################################################

def get_prediction_score(model_name, model, X_test, X_train=None):
    """Get prediction scores based on model type"""
    if model_name in ['FTTransformer', 'FEAWAD', 'GANoma', 'DevNet']:
        return model.predict_score(X_test)
    elif model_name in ['AE', 'VAE', 'DeepSVDD', 'SO_GAAL', 'LUNAR']:
        return model.decision_function(X_test)
    elif model_name == 'DAGMM':
        return model.predict_score(X_train, X_test)


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
# 4. Experiment runner
##############################################################################

def run_experiment(experiment_num, datasets, scenarios, model_configs, data_folder, results_folder, 
                  seed=42, model_scenario_filter=None):
    """
    Generic experiment runner for anomaly detection models.
    
    Parameters:
    -----------
    experiment_num : int
        Experiment number (e.g., 1 or 2)
    datasets : list
        List of dataset names to process
    scenarios : list
        List of scenario names to process
    model_configs : dict
        Dictionary of model configurations
    data_folder : str
        Path to input data folder
    results_folder : str
        Path to output results folder
    seed : int, optional
        Random seed for reproducibility
    model_scenario_filter : callable, optional
        Function that takes (scenario, model_name) and returns boolean
        indicating whether to include this combination
    """
    # Make sure the results folder exists
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        logging.info(f"Created results directory: {results_folder}")
    
    # Track experiment progress
    total_combinations = len(datasets) * len(scenarios) * len(model_configs)
    current_combination = 0
    
    logging.info(f"Starting Experiment {experiment_num} with {total_combinations} total model combinations")
    
    # Initialize experiment statistics
    successful_runs = 0
    failed_runs = 0
    skipped_runs = 0
    
    for dataset_name in datasets:
        # Path to the .npz file with prepared data
        data_path = os.path.join(data_folder, f"E{experiment_num}_{dataset_name}_{seed}_data.npz")

        if not os.path.isfile(data_path):
            logging.warning(f"File {data_path} does not exist. Skipping dataset: {dataset_name}")
            skipped_runs += len(scenarios) * len(model_configs)
            continue

        logging.info(f"Loading dataset: {dataset_name}")
        with timer(f"Load dataset {dataset_name}"):
            data = np.load(data_path, allow_pickle=True)

        # Load the common test set
        X_test = data["X_test"]
        y_test = data["y_test"]
        logging.info(f"Test set shape: {X_test.shape}, Anomalies: {np.sum(y_test == 1)}/{len(y_test)} ({np.mean(y_test) * 100:.2f}%)")

        # For each scenario, load the corresponding training sets
        for scenario in scenarios:
            x_train_key = f"X_train_{scenario}"
            y_train_key = f"y_train_{scenario}"

            if x_train_key not in data or y_train_key not in data:
                logging.warning(f"Missing key {x_train_key} or {y_train_key} in file {data_path}. Skipping scenario {scenario}")
                skipped_runs += len(model_configs)
                continue

            X_train = data[x_train_key]
            y_train = data[y_train_key]

            logging.info(f"Processing dataset: {dataset_name}, scenario: {scenario}")
            logging.info(f"Training set shape: {X_train.shape}, Anomalies: {np.sum(y_train == 1)}/{len(y_train)} ({np.mean(y_train) * 100:.2f}%)")

            for model_name, model_initializer in model_configs.items():
                # Apply filter if provided
                if model_scenario_filter and not model_scenario_filter(scenario, model_name):
                    skipped_runs += 1
                    continue
                    
                current_combination += 1
                progress = (current_combination / total_combinations) * 100
                
                logging.info(f"Progress: {current_combination}/{total_combinations} ({progress:.1f}%) - Training model: {model_name}")
                
                try:
                    with timer(f"Initialize {model_name}"):
                        if model_name != "DeepSVDD":
                            model = model_initializer()
                        else:
                            model = DeepSVDD(n_features=X_train.shape[1])
                    
                    # Fit the model to the data
                    with timer(f"Train {model_name} on {dataset_name} scenario {scenario}"):
                        if model_name not in SOMEHOW_SUPERVISED and model_name != 'GANoma' and model_name != 'DAGMM':
                            model.fit(X_train)
                        else:
                            model.fit(X_train, y_train)
                    
                    with timer(f"Predict with {model_name}"):
                        if model_name != 'DAGMM':
                            scores = get_prediction_score(model_name, model, X_test)
                        else:  
                            scores = get_prediction_score(model_name, model, X_test, X_train)  

                    if scores is None:
                        logging.warning(f"Model '{model_name}' did not return results. Skipping save.")
                        failed_runs += 1
                        continue

                    # Prepare output filename
                    results_filename = f"E{experiment_num}_{dataset_name}_{scenario}_{model_name}_results.npz"
                    results_path = os.path.join(results_folder, results_filename)

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
    logging.info(f"Experiment {experiment_num} Completed")
    logging.info(f"Total combinations: {total_combinations}")
    logging.info(f"Successful runs: {successful_runs}")
    logging.info(f"Failed runs: {failed_runs}")
    logging.info(f"Skipped runs: {skipped_runs}")
    logging.info("=" * 80)


def log_system_info(experiment_num, datasets, scenarios, models):
    """Log system information at the start of an experiment"""
    import platform
    import sys
    
    logging.info("=" * 80)
    logging.info(f"Starting experiment at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Platform: {platform.platform()}")
    logging.info(f"Datasets: {', '.join(datasets)}")
    logging.info(f"Scenarios: {', '.join(scenarios)}")
    logging.info(f"Models: {', '.join(models)}")
    logging.info("=" * 80)