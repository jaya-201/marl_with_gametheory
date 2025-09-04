import pandas as pd
import numpy as np
import os

np.random.seed(42)
markets = [f"MKT{i}" for i in range(1, 11)]   # 10 markets
airlines = ["AA", "DL", "UA", "B6"]           # 4 airlines
quarters = [1, 2, 3, 4]                       # 4 quarters

data = []

for market in markets:
    for quarter in quarters:
        num_airlines = np.random.randint(2, 5)
        competing_airlines = np.random.choice(airlines, num_airlines, replace=False)
        
        for airline in competing_airlines:
            passengers = np.random.randint(50, 500)  # realistic passenger counts
            avg_fare = np.random.uniform(100, 600)   # realistic fare range
            data.append([market, quarter, airline, passengers, avg_fare])

# Build DataFrame
df = pd.DataFrame(data, columns=["MarketID", "Quarter", "Airline", "Passengers", "AvgFare"])

# --- Save to data/ folder ---
os.makedirs("data", exist_ok=True)
save_path = "data/synthetic_airline_data.csv"
df.to_csv(save_path, index=False)

print(f"Synthetic dataset saved at: {save_path}")
print(df.head(10))