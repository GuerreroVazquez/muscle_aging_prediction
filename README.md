# muscle_aging_prediction
# Muscle Age Prediction & Gene Signature Identification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.placeholder.svg)](https://doi.org/10.5281/zenodo.placeholder)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the computational pipeline used to predict biological age from human skeletal muscle transcriptomics and identify personalised gene signatures of ageing. This workflow supports the findings presented in **Chapter 2** of the PhD thesis *"Computational approaches for therapeutic target discovery to ameliorate muscle wasting during ageing"* (University of Galway, 2025).

## Pipeline Workflow

The analysis is structured into the following stages:

1.  **Data Ingestion & Harmonisation:**
    *   Integration of raw RNA-Seq counts from 6 diverse studies (GEO).
    *   Batch correction using **ComBat-seq** (via `pyCombat`).
    *   Normalisation and pre-processing.

2.  **Differential Expression Analysis (DEA):**
    *   Comparisons between Young, Middle-Aged, and Old cohorts using **DESeq2** logic.
    *   Sex-stratified analysis.

3.  **Machine Learning Models:**
    *   **Ridge Regression:** Linear baseline for age prediction.
    *   **CatBoost:** Gradient boosting model to capture non-linear ageing dynamics.
    *   **Data Augmentation:** Implementation of SMOTE to balance age groups in the training set.

4.  **Explainability & Feature Selection:**
    *   **SHAP Analysis:** Calculation of global and local feature importance to derive personalised ageing signatures.
    *   **Elbow Method:** Dynamic thresholding for selecting top predictive genes.

## Dependencies

*   Python 3.9+
*   `catboost`
*   `scikit-learn`
*   `shap`
*   `combat` / `neuroCombat`
*   `pandas` & `numpy`
*   `matplotlib` & `seaborn`

## Usage

### 1. Data Preparation
Place raw count matrices in the `data/raw` directory. (Note: Due to size/licensing, raw GEO data is not included; please refer to the `data_sources.txt` for accession numbers).

### 2. Run the Pipeline
The main analysis can be executed via the master script:

```bash
python run_analysis.py --config config.yaml
```
run_analysis.py: Orchestrates batch correction, training, and evaluation.

config.yaml: Defines parameters (e.g., train/test split ratios, hyperparameters).

### Output
* Models: Saved CatBoost and Ridge models in .cbm and .pickle formats.

* Signatures: CSV files containing the list of top predictive genes and their SHAP scores.

* Plots: Performance metrics (RMSE, R2), SHAP summary plots, and PCA visualisations.

<h2>Repository Structure</h2>
<pre>
Identify_muscle_age_genes/tree/dev
├── data/              # Contains processed RNA-seq data and metadata
├── notebooks/         # Jupyter notebooks for data analysis and modeling
├── scripts/           # Python scripts for data processing, model training, and evaluation
├── results/           # Output directory for model predictions and analysis results
├── README.md          # This file
└── requirements.txt  # List of Python dependencies
</pre>


<h2>Data Processing Pipeline</h2>
<ol>
    <li><strong>Data Extraction:</strong> RNA-seq data from publicly available datasets is downloaded and preprocessed.</li>
    <li><strong>Quality Control:</strong> FastQC is used for quality control, and Cutadapt removes adapters.</li>
    <li><strong>Read Mapping:</strong> Kallisto is used to map reads to the Human Genome GRCh38 reference.</li>
    <li><strong>Batch Correction:</strong> ComBat algorithm (implemented via <code>pyComBat</code>) is applied to minimize batch effects. Data is Z-score normalized.</li>
    <li><strong>Differential Expression Analysis:</strong> DESeq2 (in R, accessed via <code>rpy2</code>) is used to identify differentially expressed genes (DEGs) between age groups.</li>
    <li><strong>Data Augmentation:</strong> SMOTE is used to address imbalanced sample distribution across age groups, improving machine learning model performance.</li>
</ol>

<h2>Model Training and Evaluation</h2>
<p>We used two primary machine learning models:</p>
<ul>
    <li><strong>Ridge Regression:</strong> A linear model with L2 regularization, implemented using scikit-learn.</li>
    <li><strong>CatBoost:</strong> A gradient boosting framework based on decision trees, known for its robust handling of categorical features and built-in mechanisms to combat overfitting.</li>
</ul>
<p>The data is partitioned into training and testing sets using both 60:40 and 90:10 splits. Hyperparameter optimization is performed using the Optuna framework.</p>



