import os
import zipfile

# Path to your zip file
zip_path = r"data\raw\ot_delaycause1_DL.zip"
extract_dir = "data/raw/airline_ontime_2023"

# Create folder if not exists
os.makedirs(extract_dir, exist_ok=True)

# Extract
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Extracted to {extract_dir}")