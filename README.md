# IoT Intrusion Detection System (IDS) using Machine Learning

This project implements a modular Network Intrusion Detection System (NIDS) designed for IoT environments. It processes raw network traffic logs from the **Edge-IIoTset** dataset, cleans the data to prevent leakage, transforms features, and uses ensemble Machine Learning models to classify 15 different types of network activity (normal traffic and 14 specific cyberattacks).

The main focus of this implementation is **clean architecture** and **resource efficiency**, preventing high-memory crashes during large-scale data transformation.

---

## Project Structure

The codebase is strictly separated into numbered, independent steps. Each script handles exactly one part of the pipeline:

* **`data/`**: Storage directory for raw and processed datasets (ignored by Git due to file size limits).
* **`output/`**: Directory containing analytical plots generated after model evaluation.
* **`src/`**: Source code modularized by tasks:
    * **`1_data_prep/`**:
        * `1_explore_data.py`: Initial dataset analysis (shapes, classes, data types).
        * `2_clean_data.py`: Dropping data leakage indicators (IPs, timestamps) and handling corrupted columns.
        * `3_verify_clean.py`: Automated integrity check for null values and duplicates.
        * `4_transform_data.py`: Safe categorical feature encoding and target vectorization.
    * **`2_training/`**:
        * `train_models.py`: Training and benchmarking Decision Tree and Random Forest classifiers.
    * **`3_evaluation/`**:
        * `plot_confusion_matrix.py`: Visualizing a $15 \times 15$ heatmap of model predictions.
        * `plot_feature_importance.py`: Plotting the top 10 most critical network features used by the model.

---

## How to Setup and Run

### 1. Environment Configuration
Clone the repository, set up a Python virtual environment, and install the required dependencies:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt