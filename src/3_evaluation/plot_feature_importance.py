import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
X_PATH = os.path.join(BASE_DIR, "data", "X.csv")
Y_PATH = os.path.join(BASE_DIR, "data", "y.npy")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

def main():
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("Error: processed data files X.csv or y.npy not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading datasets for Feature Importance...")
    X = pd.read_csv(X_PATH)
    y = np.load(Y_PATH)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training evaluation model...")
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    print("Analyzing and plotting feature importances...")
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_k = 10
    top_indices = indices[:top_k]
    top_features = [X.columns[i] for i in top_indices]
    top_weights = importances[top_indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_weights, y=top_features, palette="viridis", hue=top_features, legend=False)
    plt.title('Top 10 Most Important Network Features (Random Forest)')
    plt.xlabel('Relative Importance (Weight)')
    plt.ylabel('Features')
    
    plot_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Success: Plot saved to {plot_path}")

if __name__ == "__main__":
    main()