import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_vectors.csv")

def verify_dataset():
    if not os.path.exists(CLEAN_DATA_PATH):
        print(f"Error: cleaned file not found at {CLEAN_DATA_PATH}")
        return

    print("Loading cleaned dataset for verification...")
    df = pd.read_csv(CLEAN_DATA_PATH, low_memory=False)
    
    print("\n--- Integrity Check ---")
    
    missing_values = df.isnull().sum().sum()
    print(f"Total missing values (NaN): {missing_values}")
    
    duplicate_count = df.duplicated().sum()
    print(f"Total duplicate rows: {duplicate_count}")
    
    forbidden_cols = ['ip.src_host', 'ip.dst_host', 'frame.time']
    leaked_cols = [col for col in forbidden_cols if col in df.columns]
    print(f"Leaked network identifiers found: {len(leaked_cols)} {leaked_cols}")
    
    print("\n--- Data Type Verification ---")
    critical_cols = ['tcp.srcport', 'tcp.dstport', 'dns.qry.name.len']
    for col in critical_cols:
        if col in df.columns:
            print(f"Column '{col}' data type: {df[col].dtype}")

if __name__ == "__main__":
    verify_dataset()