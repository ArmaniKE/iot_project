import os
import pandas as pd

# get project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "ML-EdgeIIoT-dataset.csv")

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: file not found at {DATA_PATH}")
        return

    print("Loading raw dataset for exploration...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    print("\nDataset Shape:")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    print("\nTarget Variable Distribution (Attack_type):")
    print(df['Attack_type'].value_counts())
    
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    print(f"\nDetected Categorical Columns ({len(categorical_columns)}):")
    print(categorical_columns)

if __name__ == "__main__":
    main()