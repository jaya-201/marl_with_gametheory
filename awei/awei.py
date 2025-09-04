import pandas as pd

raw_filename = 'mktdata79q1to16q3.dta' 
raw_filepath = f'data/raw/market_carrier_data/{raw_filename}'

print(f"Loading data from {raw_filepath} to find popular routes...")
df = pd.read_stata(raw_filepath)

column_mapping = {'ap1': 'Origin', 'ap2': 'Destination'}
df.rename(columns=column_mapping, inplace=True)

print("\nTop 10 Busiest Origin Airports")
print(df['Origin'].value_counts().head(10))

print("\nTop 10 Busiest Destination Airports")
print(df['Destination'].value_counts().head(10))

print("\nTop 10 Busiest Routes (Origin-Destination Pairs)")
print(df.groupby(['Origin', 'Destination']).size().nlargest(10))