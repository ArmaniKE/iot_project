# IoT/IIoT Intrusion Detection System

This project builds a multiclass Network Intrusion Detection System for IoT and IIoT traffic using the Edge-IIoTset cyber security dataset. It detects normal traffic and 14 attack categories from packet/network features using a reproducible machine learning pipeline.

The project is intentionally structured as a small but serious ML system: data cleaning, leakage prevention, feature transformation, model comparison, artifact saving, reporting, and visualization are separated into reusable modules.

## Dataset

Source: Edge-IIoTset Cyber Security Dataset of IoT and IIoT.

The raw dataset contains packet-level and protocol-level features from ARP, ICMP, HTTP, TCP, UDP, DNS, MQTT, and Modbus/TCP traffic. The target task is multiclass classification using `Attack_type`.

Classes:

```text
Backdoor
DDoS_HTTP
DDoS_ICMP
DDoS_TCP
DDoS_UDP
Fingerprinting
MITM
Normal
Password
Port_Scanning
Ransomware
SQL_injection
Uploading
Vulnerability_scanner
XSS
```

## Architecture

```text
data/
  ML-EdgeIIoT-dataset.csv      Raw dataset
  cleaned_vectors.csv          Cleaned dataset
  X.csv                        Encoded feature matrix for inspection
  y.npy                        Encoded target vector for inspection

models/
  best_intrusion_model.joblib  Saved best sklearn pipeline

output/
  confusion_matrix.png
  feature_importance.png
  class_distribution.png
  model_comparison.png
  per_class_f1.png

app/
  streamlit_app.py             Visual IDS dashboard

reports/
  classification_report.csv
  data_quality_report.json
  label_mapping.csv
  model_metrics.json
  test_predictions.csv

src/
  run_pipeline.py              Main CLI entrypoint
  iot_ids/                     Reusable project package
    config.py                  Paths, constants, feature/drop lists
    preprocessing.py           Exploration, cleaning, verification, encoding
    modeling.py                Pipelines, model comparison, artifact saving
    evaluation.py              Confusion matrix and feature importance plots
    pipeline.py                End-to-end orchestration
    utils.py                   Shared file/report helpers
  1_data_prep/                 Step-by-step wrappers for coursework flow
  2_training/
  3_evaluation/
```

## What Makes The Project More Serious

- Uses a reusable Python package instead of isolated scripts.
- Removes leakage-prone identifiers such as IP addresses, timestamps, payloads, URI fields, and message contents.
- Verifies cleaned data for missing values, duplicates, and leaked columns.
- Uses scikit-learn `Pipeline` and `ColumnTransformer` so categorical encoding is fitted only on the training split.
- Compares multiple models: Decision Tree, Random Forest, and Extra Trees.
- Selects the best model by macro F1-score, which is more meaningful for imbalanced multiclass attacks.
- Saves the complete best model pipeline with preprocessing included.
- Generates machine-readable reports for metrics, predictions, class mapping, and data quality.
- Produces visual outputs for confusion matrix and feature importance.

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Place the raw dataset at:

```text
data/ML-EdgeIIoT-dataset.csv
```

## Run The Full Pipeline

```bash
python src/run_pipeline.py all
```

This performs:

1. Raw data exploration
2. Cleaning and leakage prevention
3. Cleaned data verification
4. Encoded `X.csv` and `y.npy` export
5. Model training and comparison
6. Best model saving
7. Report generation
8. Plot generation

## Run Individual Steps

```bash
python src/run_pipeline.py explore
python src/run_pipeline.py clean
python src/run_pipeline.py verify
python src/run_pipeline.py transform
python src/run_pipeline.py train
python src/run_pipeline.py plots
```

## Run The Dashboard

After the pipeline creates the model, reports, and plots, launch the visual IDS dashboard:

```bash
streamlit run app/streamlit_app.py
```

The dashboard is the final product of the project. It shows the project goal, data quality summary, class distribution, model comparison, confusion matrix, feature importance, per-class F1 score, classification report, and a CSV upload area for trying predictions on traffic rows.

The original numbered scripts also work:

```bash
python src/1_data_prep/1_explore_data.py
python src/1_data_prep/2_clean_data.py
python src/1_data_prep/3_verify_clean.py
python src/1_data_prep/4_transform_data.py
python src/2_training/train_models.py
python src/3_evaluation/plot_confusion_matrix.py
python src/3_evaluation/plot_feature_importance.py
```

## Model Outputs

The training step writes:

- `models/best_intrusion_model.joblib`: complete fitted sklearn pipeline
- `reports/model_metrics.json`: metrics for all candidate models
- `reports/classification_report.csv`: precision, recall, and F1 by class
- `reports/test_predictions.csv`: true and predicted labels for test samples
- `reports/label_mapping.csv`: numeric label to attack type mapping

## Evaluation

The project evaluates models with:

- Accuracy
- Balanced accuracy
- Macro F1-score
- Weighted F1-score
- Per-class classification report
- Confusion matrix
- Feature importance
- Class distribution
- Model comparison chart
- Per-class F1 score chart

Macro F1-score is used for model selection because the dataset is imbalanced and minority attack classes such as `MITM` and `Fingerprinting` should matter, not only the largest classes.

## Current Dataset Summary

The checked local dataset contains:

- Raw data: 157,800 rows and 63 columns
- Cleaned data: 152,353 rows and 52 columns
- Final model features: 50 columns
- Number of classes: 15

## Notes

Generated data, models, plots, and reports are not required to be committed if file size is a concern. The pipeline can regenerate them from the raw dataset.
