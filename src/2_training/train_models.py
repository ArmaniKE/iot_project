import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
X_PATH = os.path.join(BASE_DIR, "data", "X.csv")
Y_PATH = os.path.join(BASE_DIR, "data", "y.npy")

def main():
    if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
        print("Error: p rocessed data files X.csv or y.npy not found.")
        return

    print("Loading datasets...")
    X = pd.read_csv(X_PATH)
    y = np.load(Y_PATH)
    
    print("Splitting data into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Decision Tree
    print("\nTraining Decision Tree Classifier...")
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    
    print("Evaluating Decision Tree...")
    dt_preds = dt_model.predict(X_test)
    print("\nDecision Tree Performance:")
    print(classification_report(y_test, dt_preds, digits=4))

    # Random Forest
    print("\nTraining Random Forest Classifier (this might take 1-2 minutes)...")
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    print("Evaluating Random Forest...")
    rf_preds = rf_model.predict(X_test)
    print("\nRandom Forest Performance:")
    print(classification_report(y_test, rf_preds, digits=4))

if __name__ == "__main__":
    main()