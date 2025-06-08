import pandas as pd
import numpy as np

# Simulate unconditional death dates for Czech vaccination dataset.

# - Reads original CSV with vaccination dates and death dates.
# - Keeps all vaccination dates unchanged.
# - Simulates new death dates randomly between the earliest and latest vaccination dates,
#  ignoring vaccination timing (deaths may occur before or after vaccination).
# - Applies a fixed death rate (default 10%) to decide who dies.
# - Saves the simulated dataset to a new CSV file with the same structure.
# - Purpose: create unbiased test data to evaluate plotting and normalization logic.

input_csv = r"C:\github\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
output_csv = r"C:\github\CzechFOI-DRATE\TERRA\sim_Vesely_106_202403141131.csv"

# Load data
df = pd.read_csv(input_csv)

# Date columns
death_col = "DatumUmrti"
vax_cols = [f"Datum_{i}" for i in range(1,8)]

# Convert all date columns to datetime (errors='coerce' turns empty/invalid to NaT)
df[death_col] = pd.to_datetime(df[death_col], errors='coerce')
for col in vax_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# Find global min and max dates from vaccination dates only
min_date = df[vax_cols].min().min()
max_date = df[vax_cols].max().max()

print(f"Simulating deaths between {min_date.date()} and {max_date.date()}")

death_rate = 0.10  # 10% death rate - adjust as needed

np.random.seed(42)  # reproducible

def simulate_death_date():
    if np.random.rand() < death_rate:
        # random death date between min and max date
        random_days = np.random.randint(0, (max_date - min_date).days + 1)
        return min_date + pd.Timedelta(days=random_days)
    else:
        return pd.NaT

# Simulate deaths unconditionally (ignoring vax dates)
df[death_col] = [simulate_death_date() for _ in range(len(df))]

# Convert back to string dates in original format (empty string if no death)
df[death_col] = df[death_col].dt.strftime('%Y-%m-%d')
df[death_col] = df[death_col].fillna('')

# Save simulated CSV with same structure
df.to_csv(output_csv, index=False)

print(f"Simulated file saved as: {output_csv}")
