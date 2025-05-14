import os
import zipfile
import subprocess
import pandas as pd

# CONFIG
KAGGLE_DATASET = "shashanknecrothapa/ames-housing-dataset"
ZIP_NAME = "ames-housing-dataset.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")
OUTPUT_CSV = os.path.join(DATA_DIR, "AmesHousing.csv")

def download_dataset():
    zip_path = os.path.join(DATA_DIR, ZIP_NAME)
    if not os.path.exists(zip_path):
        print(f"Downloading {ZIP_NAME} via Kaggle CLI...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", DATA_DIR],
            check=True
        )
    else:
        print(f"{ZIP_NAME} already exists. Skipping download.")
    return zip_path

def extract_zip(zip_path):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(DATA_DIR)

def load_and_save():
    # Adjust filename if needed (check inside data/)
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV found in data/ after extraction.")
    raw_csv = os.path.join(DATA_DIR, csv_files[0])
    print(f"Loading raw data from {raw_csv}...")
    df = pd.read_csv(raw_csv)
    print(f"Saving cleaned data to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = download_dataset()
    extract_zip(zip_path)
    load_and_save()
    print("Ingestion complete.")

if __name__ == "__main__":
    main()