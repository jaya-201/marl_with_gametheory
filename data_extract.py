import zipfile
import os

# Path to your downloaded zip file
zip_path = "data/raw/Aggregated_market_carrier.zip"
extract_dir = "market_carrier_data"

# Create folder to extract into
os.makedirs(extract_dir, exist_ok=True)

# Extract
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Extraction complete. Files extracted to:", extract_dir)