import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
X_PATH = os.path.join(BASE_DIR, "data", "X.csv")
Y_PATH = os.path.join(BASE_DIR, "data", "y.npy")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

def main():
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("Error: processed data files X.csv or y.npy not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading datasets for Confusion Matrix...")
    X = pd.read_csv(X_PATH)
    y = np.load(Y_PATH)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training evaluation model...")
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    print("Generating Confusion Matrix plot...")
    rf_preds = rf_model.predict(X_test)
    cm = confusion_matrix(y_test, rf_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    
    plot_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Success: Plot saved to {plot_path}")

if __name__ == "__main__":
    main()