import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "ML-EdgeIIoT-dataset.csv")
CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_vectors.csv")

def clean_dataset():
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: raw file not found at {RAW_DATA_PATH}")
        return

    print("Loading raw dataset for cleaning...")
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    
    print("Converting numeric features incorrectly parsed as text...")
    corrupted_numeric_cols = ['tcp.srcport', 'tcp.dstport', 'dns.qry.name.len']
    for col in corrupted_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    print("Dropping network identifiers to prevent data leakage...")
    columns_to_drop = [
        "frame.time", "ip.src_host", "ip.dst_host", 
        "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
        "http.file_data", "http.request.full_uri", 
        "http.request.uri.query", "http.referer",
        "tcp.payload", "mqtt.msg"
    ]
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)
    
    print("Handling missing values and removing duplicates...")
    df.fillna(0, inplace=True)
    
    initial_row_count = df.shape[0]
    df.drop_duplicates(inplace=True)
    removed_duplicates = initial_row_count - df.shape[0]
    print(f"Removed duplicates: {removed_duplicates} rows")
    
    print(f"Saving cleaned dataset to {CLEAN_DATA_PATH}...")
    df.to_csv(CLEAN_DATA_PATH, index=False)
    
    print("\nCleaning Process Complete:")
    print(f"Final Shape: Rows: {df.shape[0]}, Columns: {df.shape[1]}")

if __name__ == "__main__":
    clean_dataset()