#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
------------------
Script for processing and analyzing results from Experiment 1.
Loads all result files from the "results_e1" folder, computes metrics,
and presents them in a DataFrame grouped by model, dataset, and scenario.

Assumes result files are named in the format:
E1_{dataset}_{scenario}_{model_name}_results.npz
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    precision_score, 
    recall_score, 
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse
import logging
from datetime import datetime

# Logging setup
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

# Function for computing evaluation metrics
def calculate_metrics(y_true, scores):
    """
    Computes various performance metrics for anomaly detection models.
    
    Args:
        y_true: true labels (0: normal, 1: anomaly)
        scores: anomaly scores assigned by the model
        
    Returns:
        A dictionary containing metrics
    """
    metrics = {}

    # Ranking metrics
    metrics['auc'] = roc_auc_score(y_true, scores)
    metrics['ap'] = average_precision_score(y_true, scores)
    
    # Set threshold at the 90th percentile to generate binary predictions
    threshold = np.percentile(scores, 90)
    y_pred_binary = (scores >= threshold).astype(int)
    
    # Binary metrics
    metrics['precision'] = precision_score(y_true, y_pred_binary)
    metrics['recall'] = recall_score(y_true, y_pred_binary)
    metrics['f1'] = f1_score(y_true, y_pred_binary)
    
    return metrics

def process_results(results_folder="results_e1", output_file=None, analyze_top_n=3):
    """
    Main function to process all result files.
    
    Args:
        results_folder: folder containing result files
        output_file: optional path to save the results CSV
        analyze_top_n: number of top models for detailed analysis
    """
    logger = setup_logging()
    logger.info(f"Starting to process results from folder: {results_folder}")
    
    # Check if results folder exists
    if not os.path.exists(results_folder):
        logger.error(f"Folder {results_folder} does not exist!")
        return
    
    # Find all .npz files
    result_files = glob.glob(os.path.join(results_folder, "E1_*_results.npz"))
    logger.info(f"Found {len(result_files)} result files")
    
    if not result_files:
        logger.error("No result files found!")
        return
    
    results_list = []
    
    # Process each result file
    for file_path in result_files:
        filename = os.path.basename(file_path)
        parts = filename.replace("E1_", "").replace("_results.npz", "").split("_")
        
        if len(parts) >= 3:
            dataset = parts[0]
            scenario = parts[1]
            model = "_".join(parts[2:])  # model name may include underscores
            
            try:
                # Load data
                data = np.load(file_path, allow_pickle=True)
                scores = data['scores']
                y_test = data['y_test']
                
                # Compute metrics
                metrics = calculate_metrics(y_test, scores)
                
                # Add context info
                metrics['dataset'] = dataset
                metrics['scenario'] = scenario
                metrics['model'] = model
                
                # Add anomaly distribution info
                metrics['anomaly_ratio'] = np.mean(y_test)
                metrics['total_samples'] = len(y_test)
                metrics['anomaly_count'] = int(np.sum(y_test))
                
                results_list.append(metrics)
                logger.info(f"Processed {filename} - AUC: {metrics['auc']:.4f}, AP: {metrics['ap']:.4f}")
                
            except Exception as e:
                logger.error(f"Error while processing {filename}: {str(e)}")
                continue
    
    if not results_list:
        logger.error("No result files could be processed!")
        return
    
    results_df = pd.DataFrame(results_list)
    
    # Format metric columns
    metric_cols = ['auc', 'ap', 'precision', 'recall', 'f1']
    results_df[metric_cols] = results_df[metric_cols].round(4)
    
    unique_models = results_df['model'].unique()
    unique_datasets = results_df['dataset'].unique()
    unique_scenarios = results_df['scenario'].unique()
    logger.info(f"Processed results for {len(unique_models)} models, "
                f"{len(unique_datasets)} datasets, and {len(unique_scenarios)} scenarios")
    
    logger.info("Average metrics per model:")
    average_by_model = results_df.groupby('model')[metric_cols].mean().sort_values('auc', ascending=False)
    print(average_by_model)
    
    # Save to CSV
    if output_file:
        results_df.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
    
    # Create summary tables
    pivot_auc = results_df.pivot_table(
        index=['dataset', 'scenario'], 
        columns='model', 
        values='auc',
        aggfunc='mean'
    ).round(4)
    
    pivot_ap = results_df.pivot_table(
        index=['dataset', 'scenario'], 
        columns='model', 
        values='ap',
        aggfunc='mean'
    ).round(4)
    
    print("\nAUC Summary Table:")
    print(pivot_auc)
    
    print("\nAP (Average Precision) Summary Table:")
    print(pivot_ap)
    
    # Analyze top models
    top_models = average_by_model.head(analyze_top_n).index.tolist()
    logger.info(f"Top {analyze_top_n} models: {', '.join(top_models)}")
    
    top_results = results_df[results_df['model'].isin(top_models)]
    detailed_pivot = top_results.pivot_table(
        index=['dataset', 'scenario'], 
        columns='model', 
        values=['auc', 'ap']
    ).round(4)
    
    print("\nDetailed results for top models:")
    print(detailed_pivot)
    
    return results_df, pivot_auc, pivot_ap

def generate_visualizations(results_df, output_folder="visualizations"):
    """
    Generates visualizations for the experiment results.
    
    Args:
        results_df: DataFrame containing results
        output_folder: folder to save visualizations
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Boxplot comparison of models across all scenarios
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='model', y='auc', data=results_df)
    plt.title('Model Comparison - AUC')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'models_comparison_auc.png'), dpi=300)
    
    # 2. Heatmaps for each metric
    for metric in ['auc', 'ap']:
        pivot = results_df.pivot_table(
            index='model', 
            columns='dataset', 
            values=metric,
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=.5)
        plt.title(f'Mean {metric.upper()} per Dataset and Model')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'heatmap_{metric}.png'), dpi=300)
    
    # 3. Bar plots per scenario
    scenario_groups = results_df.groupby('scenario')
    for scenario, group in scenario_groups:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='model', y='auc', data=group, ci=None)
        plt.title(f'Model Comparison - Scenario {scenario}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'scenario_{scenario}_comparison.png'), dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process anomaly experiment results')
    parser.add_argument('--results_folder', default='results_e1', help='Folder with result files')
    parser.add_argument('--output_file', default='experiment1_results.csv', help='Output CSV file')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    args = parser.parse_args()
    
    # Process results
    results_df, pivot_auc, pivot_ap = process_results(
        results_folder=args.results_folder, 
        output_file=args.output_file
    )
    
    # Generate visualizations if requested
    if args.visualize:
        generate_visualizations(results_df)
