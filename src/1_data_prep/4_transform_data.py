import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_vectors.csv")
X_PATH = os.path.join(BASE_DIR, "data", "X.csv")
Y_PATH = os.path.join(BASE_DIR, "data", "y.npy")

def transform_dataset():
    if not os.path.exists(CLEAN_DATA_PATH):
        print(f"Error: cleaned file not found at {CLEAN_DATA_PATH}")
        return

    print("Loading cleaned dataset for transformation...")
    df = pd.read_csv(CLEAN_DATA_PATH, low_memory=False)
    
    print("Separating features from target variables...")
    target_cols = ['Result', 'Attack_label', 'Attack_type']
    existing_targets = [col for col in target_cols if col in df.columns]
    
    X = df.drop(columns=existing_targets)
    y = df['Attack_type']
    
    print("Applying Label Encoding to categorical features...")
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    feature_encoder = LabelEncoder()
    for col in categorical_features:
        # cast to string to prevent encoding type conflicts
        X[col] = feature_encoder.fit_transform(X[col].astype(str))
    
    print("Encoding target variable classes into numeric labels...")
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    print("Saving processed features and target arrays...")
    X.to_csv(X_PATH, index=False)
    np.save(Y_PATH, y_encoded)
    
    print("\nTransformation Process Complete:")
    print(f"Features matrix (X) shape: Rows: {X.shape[0]}, Columns: {X.shape[1]}")
    print(f"Target vector (y) shape: {y_encoded.shape[0]}")
    print(f"Total unique classes encoded: {len(target_encoder.classes_)}")

if __name__ == "__main__":
    transform_dataset()