import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

# Simulate unbiased death dates based on age-specific unconditional death rates - working 05.06.2025 23:32 ! 

# This script reads a dataset of individuals with birth year, death date, and vaccine dose dates.
# It estimates per-age death rates (ignoring time or dose alignment), then simulates new, unbiased
# death dates uniformly over the observation period based on these rates. The vaccination dates are preserved.

# Input:  CSV with birth year, death date, and up to 7 vaccination dates
# Output: CSV with simulated death dates based on unconditional probabilities


# === File paths ===
INPUT_CSV = r"C:\github\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
OUTPUT_CSV = r"C:\github\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"

# === Parameters ===
START_DATE = pd.Timestamp('2020-01-01')  # Baseline for day number conversion
REFERENCE_YEAR = 2023                    # Year used to calculate age from birth year
MAX_AGE = 113                            # Max age allowed in simulation
DOSE_DATE_COLS = [f'Datum_{i}' for i in range(1, 8)]  # Names of dose date columns
NEEDED_COLS = ['Rok_narozeni', 'DatumUmrti'] + DOSE_DATE_COLS  # Required columns in input
NUM_WORKERS = max(mp.cpu_count() - 1, 1)  # Use all but one CPU core

# Convert date series to day number since START_DATE
def to_day_number(date_series):
    return (date_series - START_DATE).dt.days

# Convert all relevant columns to datetime
def parse_dates(df):
    for col in DOSE_DATE_COLS + ["DatumUmrti"]:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

# Compute age from birth year and REFERENCE_YEAR
def calculate_age(df):
    df["Age"] = REFERENCE_YEAR - df["Rok_narozeni"]
    return df

# Estimate death rate per age group (fraction who died)
def estimate_age_death_rates(df, end_measure):
    age_rates = {}
    for age, group in df.groupby("Age"):
        death_count = group["DatumUmrti"].notna().sum()
        rate = death_count / len(group) if len(group) > 0 else 0.01
        rate = np.clip(rate, 1e-4, 0.999)  # Clamp to avoid extremes
        age_rates[age] = rate
    return age_rates

# Simulate new death dates for people of the same age group
def simulate_deaths_unconditionally(df_age, end_measure, death_rate):
    if df_age.empty:
        return df_age

    age = df_age["Age"].iloc[0]
    print(f"Simulating age {age} (unconditional)...")

    n = len(df_age)
    np.random.seed()  # Ensure unique seed per process

    # Randomly determine who dies using the death rate
    will_die = np.random.rand(n) < death_rate

    # For those who die, assign a random day between 0 and end_measure
    death_days = np.where(
        will_die,
        np.random.randint(0, end_measure + 1, size=n),
        np.nan
    )

    df_age = df_age.copy()

    # Convert simulated death days to date strings; leave empty if censored
    df_age["DatumUmrti"] = [
        (START_DATE + pd.Timedelta(days=int(d))).strftime('%Y-%m-%d') if not np.isnan(d) else ''
        for d in death_days
    ]

    return df_age

# Wrapper for multiprocessing
def simulate_wrapper(args):
    df_age, end_measure, death_rate = args
    return simulate_deaths_unconditionally(df_age, end_measure, death_rate)

# === Main process ===
def main():
    print("Loading CSV...")
    df = pd.read_csv(INPUT_CSV, usecols=NEEDED_COLS, low_memory=False)

    print("Parsing dates and calculating ages...")
    df = parse_dates(df)
    df = calculate_age(df)
    df = df[df["Age"].between(0, MAX_AGE)]  # Filter out unrealistic ages

    print("Calculating end_measure...")
    df["death_day"] = to_day_number(df["DatumUmrti"])
    end_measure = int(df["death_day"].dropna().max())  # Last observed death day
    print(f"End measure = {end_measure}")

    print("Estimating death rate per age group...")
    age_death_rates = estimate_age_death_rates(df, end_measure)

    # Prepare tuple of (age group df, end_measure, death_rate) for each age group
    age_groups = [(group, end_measure, age_death_rates[age]) for age, group in df.groupby("Age")]

    print(f"Starting multiprocessing with {NUM_WORKERS} workers...")
    with mp.Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(simulate_wrapper, age_groups), total=len(age_groups)))

    print("Concatenating results and saving CSV...")
    df_result = pd.concat(results)

    # Convert dose dates back to string format for output
    for col in DOSE_DATE_COLS:
        df_result[col] = pd.to_datetime(df_result[col], errors='coerce').dt.strftime('%Y-%m-%d')
        df_result[col] = df_result[col].fillna('')

    df_result.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Done. Output saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
