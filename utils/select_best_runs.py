#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_best_runs.py
===================
Recursively scans all result files inside the given folder (e.g. `results_e1 ➜ sub‑folders for each run`).
For the selected scenarios (default **A** and **E**) it picks, **for every model**, the repetition (run) that achieved the highest **AUC‑PR**.

Instead of creating plots it stores everything you need **to plot later** in a single **NPZ** file (default `best_runs_AE.npz`).

Each model is an entry (key) in that NPZ; the value is a dictionary with:

* `precision`, `recall`, `thresholds` – arrays returned by `precision_recall_curve`,
* `auc_pr` – area under the PR curve,
* `dataset`, `scenario`, `rep`, `file` – metadata of the best run.

Directory & file layout
-----------------------
We assume a structure like this:

```
results_e1/
  ├── results_e1_run_0/
  │     ├── E1_{dataset}_{scenario}_{model}_results.npz
  │     └── ...
  ├── results_e1_run_1_20250406_061212/
  │     └── ...
  └── ...
```

* **rep** (repetition) is simply the **sub‑folder name** (e.g. `results_e1_run_0`).
* The filename itself **does not** contain the repetition and follows the pattern
  `E1_{dataset}_{scenario}_{model}_results.npz`.

If your scheme differs, tweak `parse_filename()` accordingly.

Usage example
-------------
```bash
python select_best_runs.py \
    --results_folder results_e1 \
    --output_file best_runs_AE.npz \
    --scenario_filter A,E
```

Loading the NPZ later
---------------------
```python
import numpy as np
best = np.load('best_runs_AE.npz', allow_pickle=True)
info = best['DevNet'].item()  # example model
print(info['auc_pr'])
```
"""

import os
import glob
import argparse
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve, auc

###############################################################################
# Helper functions
###############################################################################

def parse_filename(filename: str) -> Tuple[str, str, str]:
    """Extracts (dataset, scenario, model) from a result‑file name.

    Expected pattern: `E1_{dataset}_{scenario}_{model}_results.npz`.
    The function **does not** handle the repetition – that comes from the parent
    folder name.
    """
    base = os.path.basename(filename)
    name = base.replace("E1_", "").replace("_results.npz", "")
    parts = name.split("_")

    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {filename}")

    dataset = parts[0]
    scenario = parts[1]
    model = "_".join(parts[2:])  # model name may itself contain underscores
    return dataset, scenario, model


def compute_pr(y_true: np.ndarray, scores: np.ndarray):
    """Returns precision, recall, thresholds and AUC‑PR for a set of scores."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    auc_pr = auc(recall, precision)
    return precision, recall, thresholds, auc_pr

###############################################################################
# Main logic
###############################################################################

def main(results_folder: str, output_file: str, scenarios: List[str]):
    if not os.path.exists(results_folder):
        raise FileNotFoundError(f"Folder '{results_folder}' does not exist.")

    # Recursively look for .npz files starting with E1_
    pattern = os.path.join(results_folder, "**", "E1_*_results.npz")
    result_files = glob.glob(pattern, recursive=True)

    if not result_files:
        raise RuntimeError("No .npz files matching the pattern were found.")

    best_runs: Dict[str, Dict] = {}

    for path in result_files:
        try:
            dataset, scenario, model = parse_filename(path)
        except ValueError as err:
            print(f"[WARN] {err}")
            continue

        if scenario.upper() not in scenarios:
            # Irrelevant scenario, skip
            continue

        rep = os.path.basename(os.path.dirname(path))  # repetition ID = folder

        data = np.load(path, allow_pickle=True)
        scores = data["scores"]
        y_test = data["y_test"]

        # Special case: FEAWAD sometimes returns 2D scores (n_samples, 2)
        if model.upper() == "FEAWAD" and scores.ndim == 2 and scores.shape[1] >= 2:
            scores = scores[:, 0]

        precision, recall, thresholds, auc_pr = compute_pr(y_test, scores)

        # Keep the run with the highest AUC‑PR for this model
        if (model not in best_runs) or (auc_pr > best_runs[model]["auc_pr"]):
            best_runs[model] = {
                "dataset": dataset,
                "scenario": scenario,
                "rep": rep,
                "precision": precision,
                "recall": recall,
                "thresholds": thresholds,
                "auc_pr": auc_pr,
                "file": os.path.basename(path),
            }

    if not best_runs:
        raise RuntimeError(
            "No runs left after filtering for scenarios: " + ", ".join(scenarios)
        )

    # ---------------------------------------------------------------------
    # Save as NPZ (keys = model names, values = dictionaries)
    # ---------------------------------------------------------------------
    np.savez(output_file, **best_runs)
    print(
        f"Saved data for {len(best_runs)} models to NPZ file: '{output_file}'.\n"
        "You can now load this file to create PR and F1‑vs‑threshold plots."
    )

###############################################################################
# CLI argument parser
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Selects the repetition (run) with the highest AUC‑PR for every "
            "model in scenarios A and E (or custom list) and stores the data "
            "in a single NPZ file."
        )
    )
    parser.add_argument(
        "--results_folder",
        default="results_e1_multiple_runs",
        help="Root folder that contains the run sub‑folders with .npz files.",
    )
    parser.add_argument(
        "--output_file",
        default="best_runs_E1.npz",
        help="Name of the output NPZ file.",
    )
    parser.add_argument(
        "--scenario_filter",
        default="A,E",
        help="Comma‑separated list of scenarios to keep (default: A,E)",
    )

    args = parser.parse_args()
    scenario_list = [s.strip().upper() for s in args.scenario_filter.split(",") if s]

    main(args.results_folder, args.output_file, scenario_list)
