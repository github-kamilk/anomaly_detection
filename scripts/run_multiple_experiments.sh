#!/bin/bash

num_runs=8

for ((i=1; i<=num_runs; i++))
do
  echo "=== RUN $i START ==="
  python -m experiments.experiment1
  timestamp=$(date +"%Y%m%d_%H%M%S")
  mv results_e1 "results_e1_run_${i}_${timestamp}"
  mkdir -p results_e1

  echo "=== RUN $i FINISHED ==="
done
