#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to process E1 experiment results and save them to a single CSV file.
Assumes files named in the format: E1_{dataset}_{scenario}_{model}_results.npz
"""

import os
import numpy as np
import pandas as pd
import glob
import argparse
import logging
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    precision_score, 
    recall_score, 
    f1_score,
    precision_recall_curve, 
    auc
)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def calculate_metrics(y_true, scores):
    metrics = {}
    precision, recall, _ = precision_recall_curve(y_true, scores)
    metrics['auc-PR'] = auc(recall, precision)
    metrics['auc'] = roc_auc_score(y_true, scores)
    metrics['ap'] = average_precision_score(y_true, scores)

    threshold = np.percentile(scores, 90)
    y_pred_binary = (scores >= threshold).astype(int)

    metrics['precision'] = precision_score(y_true, y_pred_binary)
    metrics['recall'] = recall_score(y_true, y_pred_binary)
    metrics['f1'] = f1_score(y_true, y_pred_binary)

    return metrics

def process_results(main_folder, output_file="experiment1_results.csv"):
    logger = setup_logging()
    logger.info(f"Processing results from folder: {main_folder}")

    if not os.path.exists(main_folder):
        logger.error(f"Folder {main_folder} does not exist!")
        return

    search_pattern = os.path.join(main_folder, "**", "E1_*_results.npz")
    result_files = glob.glob(search_pattern, recursive=True)
    logger.info(f"Found {len(result_files)} result file(s)")

    if not result_files:
        logger.error("No result files found.")
        return

    results_list = []

    for file_path in result_files:
        filename = os.path.basename(file_path)
        parts = filename.replace("E1_", "").replace("_results.npz", "").split("_")

        if len(parts) >= 3:
            dataset = parts[0]
            scenario = parts[1]
            model = "_".join(parts[2:])
        else:
            logger.warning(f"Invalid filename format: {filename}")
            continue

        try:
            data = np.load(file_path, allow_pickle=True)
            scores = data['scores']
            y_test = data['y_test']

            if model.upper() == 'FEAWAD':
                if scores.ndim == 2 and scores.shape[1] == 2:
                    scores = scores[:, 0]
                else:
                    logger.warning(f"FEAWAD format unexpected in {filename}, shape: {scores.shape}")
                            
            metrics = calculate_metrics(y_test, scores)
            metrics['dataset'] = dataset
            metrics['scenario'] = scenario
            metrics['model'] = model
            metrics['anomaly_ratio'] = np.mean(y_test)
            metrics['total_samples'] = len(y_test)
            metrics['anomaly_count'] = int(np.sum(y_test))

            results_list.append(metrics)
            logger.info(
                f"Processed {filename} | AUC: {metrics['auc']:.4f}, AP: {metrics['ap']:.4f}"
            )

        except Exception as e:
            logger.error(f"Error processing file {filename}: {str(e)}")
            continue

    if not results_list:
        logger.error("No files were successfully processed.")
        return

    results_df = pd.DataFrame(results_list)
    metric_cols = ['auc-PR', 'auc', 'ap', 'precision', 'recall', 'f1']
    results_df[metric_cols] = results_df[metric_cols].round(4)

    results_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process anomaly experiment results and generate a single CSV.')
    parser.add_argument(
        '--results_folder',
        default='results_e1_multiple_runs',
        help='Root folder containing subfolders with E1_*_results.npz files.'
    )
    parser.add_argument(
        '--output_file',
        default='experiment1_results.csv',
        help='CSV filename for saving the results.'
    )

    args = parser.parse_args()
    process_results(
        main_folder=args.results_folder,
        output_file=args.output_file
    )
