#!/usr/bin/env python3
"""
extract_training_times.py

Parse all .log files in a given directory and extract model training times.
Outputs a CSV with columns: model;dataset;scenario;time (in seconds).
"""
import os
import re
import csv
import glob
import argparse

def parse_time_str(time_str):
    # Matches "X minutes and Y.ZZ seconds" or "Y.ZZ seconds"
    match = re.match(r'(?:(\d+)\s+minutes?\s+and\s+)?([\d\.]+)\s+seconds?', time_str)
    if not match:
        return None
    minutes = int(match.group(1)) if match.group(1) else 0
    seconds = float(match.group(2))
    return minutes * 60 + seconds

# Regex to capture model, dataset, scenario and time
pattern = re.compile(
    r"Task 'Train (?P<model>.+?) on (?P<dataset>.+?) scenario (?P<scenario>.+?)' completed in (?P<time>\d+ minutes and [\d\.]+ seconds|[\d\.]+ seconds)"
)

def process_file(file_path, rows):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                model = m.group('model')
                dataset = m.group('dataset')
                scenario = m.group('scenario')
                time_str = m.group('time')
                time_sec = parse_time_str(time_str)
                if time_sec is not None:
                    rows.append({
                        'model': model,
                        'dataset': dataset,
                        'scenario': scenario,
                        'time': time_sec
                    })

def main(input_dir, output_csv):
    rows = []
    log_files = glob.glob(os.path.join(input_dir, '*.log'))
    if not log_files:
        print(f"No .log files found in {input_dir}")
        return

    for file_path in log_files:
        process_file(file_path, rows)

    # Write to CSV with semicolon delimiter
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['model', 'dataset', 'scenario', 'time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract model training times from .log files into a CSV.'
    )
    parser.add_argument('--input_dir',
                        default='logs_e1', 
                        help='Directory containing .log files')
    parser.add_argument('--output_csv',
                        default='results/training_times.csv',
                        help='Path for the output CSV file')
    args = parser.parse_args()
    main(args.input_dir, args.output_csv)
