# Causal Discovery Meets Explainable Predictive Maintenance

A graph-based experiment pipeline for **HPC anomaly prediction** using the **M100 dataset**.  
This project combines strong tabular baselines, temporal feature engineering, SHAP explainability, and causal discovery to study whether the most predictive telemetry variables also align with learned causal structure.

## Overview

The notebook implements an end-to-end workflow for **predictive maintenance in HPC systems** using time-aggregated IPMI telemetry from the **Marconi100** supercomputer at CINECA.

Core idea:

- train high-performing anomaly prediction models on node telemetry
- explain predictions with **SHAP**
- discover structure with **PC** and **PCMCI**
- compare **predictive importance** vs. **causal relevance**
- generate **local causal explanations** for anomalous cases

The experiment is tailored to the paper:

> **Causal Discovery Meets Explainable Predictive Maintenance: A Graph-Based Approach to Anomaly Prediction in HPC Systems**

## Dataset

The notebook uses the **M100 dataset** (Borghesi et al., 2023), specifically:

- **rack 0 only**
- **16 computing nodes**
- **15-minute aggregation windows**
- node-level telemetry from IPMI sensors
- anomaly labels from **Nagios**

Label semantics:

- `0` = OK
- `2` = WARNING
- `3` = CRITICAL
- `NaN` = missing Nagios label

To ensure a consistent feature space across nodes, the pipeline keeps only the **intersection of columns** shared by all rack-0 node files.

## What the notebook does

### 1. Data loading and schema alignment
- loads all rack-0 parquet files
- computes the intersection of columns across nodes
- concatenates node data into one master dataframe
- adds `node_id`
- parses timestamps
- saves an intermediate processed parquet file

### 2. Target construction
- validates the raw Nagios label column
- drops rows with missing labels
- binarizes anomalies
- creates a **prediction horizon target** using a `t+1` shift:
  - features at time `t`
  - predict anomaly at time `t+1`

### 3. Missing value handling and feature cleanup
- identifies telemetry feature columns
- applies **two-stage imputation**:
  - per-node forward fill
  - column median fallback
- removes zero-variance features
- builds final `X` and `y`

### 4. Exploratory data analysis
- target distribution
- temporal anomaly patterns
- sensor correlation analysis
- clustered heatmaps
- feature distributions for normal vs. anomalous cases

### 5. Temporal train/test split
- performs an **80/20 time-based split**
- avoids random shuffling and future leakage
- evaluates on later timestamps only

### 6. Temporal feature engineering
For the top correlated sensors, the notebook creates lag-based features:

- `lag1`
- `lag2`
- rolling mean over 4 windows
- rolling std over 4 windows
- first difference (`delta1`)

This expands the original telemetry representation with short-term temporal dynamics.

### 7. Predictive modeling
The notebook compares multiple model families:

#### Gradient-boosted tree models
- **LightGBM**
- **CatBoost**
- **XGBoost**

These are the main supervised baselines and the downstream source for SHAP analysis.

#### Additional complementary models
- **Semi-supervised Autoencoder**
- **Temporal Convolutional Network (TCN)**
- **GraphSAGE**

These provide deep-learning and graph-based comparisons for the paper.

> Note: the Logistic Regression baseline is present in the notebook but commented out.

### 8. Explainability with SHAP
- computes SHAP values for the best tree-based model
- ranks global feature importance
- generates summary and bar-plot style importance views
- optionally samples the test set for tractability

### 9. Causal discovery
Two causal perspectives are used:

#### PC algorithm
- runs causal discovery on top SHAP-ranked features plus target
- learns a static causal graph

#### PCMCI
- runs temporal causal discovery on a representative node
- identifies lagged parents of the anomaly target
- supports temporal interpretation such as `sensor(t-k) -> anomaly(t)`

### 10. Predictive vs. causal comparison
- compares SHAP top features against causal parents
- quantifies overlap
- generates a comparison table for analysis and paper writing

### 11. Local causal explanations
For selected true-positive anomalies, the notebook:
- finds instance-level top SHAP drivers
- intersects them with causal parents
- builds minimal causal subgraphs
- produces human-readable local causal narratives

This is one of the central contributions of the workflow.

### 12. Ablation study
The notebook tests whether causal parents matter more than similarly ranked non-causal features by:
- removing causal features
- removing non-causal features
- retraining the best model
- comparing the performance drop

### 13. Robustness check
A second target is created for a broader prediction horizon:

- anomaly in the **next 4 windows** instead of only the next window

This tests whether conclusions remain stable for a 1-hour horizon.

### 14. Export for paper artifacts
The notebook exports tables and figures such as:

- dataset statistics
- model comparison table
- PCMCI causal edges
- SHAP vs causal comparison
- ROC curves
- SHAP importance figure

### 15. Final experiment summary
A final summary cell prints:
- dataset scope
- feature counts
- anomaly rate
- model performance
- top SHAP features
- temporal causal parents
- overlap and ablation findings
- limitations

---

## Repository structure

A typical layout for this project would look like:

```text
.
├── graphsys_experiment_updated_with_comments.ipynb
├── README.md
├── dataset/
│   └── M100/
│       └── parquet_dataset/
│           ├── 0/
│           │   ├── 2.parquet
│           │   ├── 3.parquet
│           │   └── ...
│           └── processed/
└── outputs/
    └── paper_artifacts/
```

## Expected input paths

The notebook is configured for the following dataset path:

```python
../../dataset/M100/parquet_dataset/0
```

and writes processed / exported outputs to:

```python
../../dataset/M100/parquet_dataset/processed
../../outputs/paper_artifacts
```

You may need to adjust these paths depending on your local setup.

## Main dependencies

The notebook imports and uses:

- pandas
- numpy
- pyarrow
- matplotlib
- seaborn
- scikit-learn
- lightgbm
- catboost
- xgboost
- shap
- networkx
- causallearn
- tigramite
- torch
- torch-geometric

## Installation

Example setup with `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install \
  pandas numpy pyarrow matplotlib seaborn scikit-learn \
  lightgbm catboost xgboost shap networkx causallearn tigramite \
  torch torch-geometric jupyter
```

Or with `pip`:

```bash
pip install \
  pandas numpy pyarrow matplotlib seaborn scikit-learn \
  lightgbm catboost xgboost shap networkx causallearn tigramite \
  torch torch-geometric jupyter
```

## Outputs

The notebook exports several paper-ready artifacts, including:

- `dataset_statistics.csv`
- `model_comparison.csv`
- `pcmci_causal_edges.csv`
- `shap_vs_causal_comparison.csv`
- `roc_curves.png`
- `shap_importance.png`

## Research questions supported by this notebook

This repository is useful if you want to study questions such as:

- Which telemetry variables best predict future anomalies in HPC nodes?
- Do the strongest predictive variables align with variables identified as causal parents?
- Can global model explanations be converted into local causal diagnostics?
- Do causal features carry more predictive signal than non-causal but correlated features?

## Key methodological choices

- **common-schema enforcement** to avoid structural missingness across nodes
- **time-based splitting** to prevent leakage
- **lag/rolling features** to capture short-term temporal dynamics
- **GBDT models** for strong tabular baselines
- **SHAP** for predictive interpretability
- **PC + PCMCI** for static and temporal causal discovery
- **ablation** to test whether causal variables carry stronger signal

## Limitations

The notebook itself highlights several limitations:

- uses only **one rack** rather than the full system
- PCMCI is run on a **single representative node**
- ParCorr assumes mainly **linear relationships**
- 15-minute aggregation may miss very fast anomaly dynamics
- causal discovery may be affected by hidden confounders

## Citation

If you use this workflow, please cite:

- the **[M100 dataset by Borghesi et al.](https://zenodo.org/records/7541722)**

## Notes

- Some sections are designed for **paper production**, not just experimentation.
- Several figures and tables are generated specifically for direct use in a manuscript.
- The notebook is heavily commented and can also serve as a teaching or reproducibility resource for explainable predictive maintenance and causal analysis in HPC telemetry.
