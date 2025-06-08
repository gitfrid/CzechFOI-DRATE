# === Import required libraries ===
import pandas as pd
import numpy as np
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

# Simulate unbiased death dates based on age-specific hazard rates. working 06.06.2025 02:11

# This script processes a Czech vaccination and death dataset to estimate daily hazard rates 
# (death probabilities) per age group, then simulates unbiased death dates using an exponential 
# distribution with censoring. It preserves existing vaccination dates but updates death dates 
# to remove bias due to immortal time.

# Input:  CSV Original Czech-FOI individual-level vaccination and death record dataset file
# with birth year, death date, and up to 7 vaccination dates per person  

# Output:  CSV file with the same structure, 
# Simulated unbiased death dataset with estimated hazard-based death days


# === File paths  ===
INPUT_CSV = r"C:\github\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
OUTPUT_CSV = r"C:\github\CzechFOI-DRATE\TERRA\sim_HR_NOBIAS_Vesely_106_202403141131.csv"

# === Global Parameters ===
START_DATE = pd.Timestamp('2020-01-01')  # Base date for day-number calculations
REFERENCE_YEAR = 2023                    # Year used for age calculation
MAX_AGE = 113                            # Upper bound on valid age
DOSE_DATE_COLS = [f'Datum_{i}' for i in range(1, 8)]  # Columns representing up to 7 dose dates
NEEDED_COLS = ['Rok_narozeni', 'DatumUmrti'] + DOSE_DATE_COLS  # Required columns to read from CSV
NUM_WORKERS = max(mp.cpu_count() - 1, 1)  # Use all CPU cores except one

# === Utility: Convert datetime series to integer days since START_DATE ===
def to_day_number(date_series):
    return (date_series - START_DATE).dt.days

# === Parse all date columns (doses + death) as datetime objects ===
def parse_dates(df):
    for col in DOSE_DATE_COLS + ["DatumUmrti"]:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

# === Compute individual's age from birth year ===
def calculate_age(df):
    df["Age"] = REFERENCE_YEAR - df["Rok_narozeni"]
    return df

# === Estimate hazard rate (deaths per person-day) for each age group ===
def estimate_age_specific_hazards(df, end_measure):
    age_hazard = {}
    for age, group in df.groupby("Age"):
        death_days = to_day_number(group["DatumUmrti"]).fillna(np.nan)
        death_days = death_days.clip(upper=end_measure)  # Cap death days at end_measure
        followup_days = death_days.fillna(end_measure)   # Censor non-deaths at end_measure
        total_person_days = followup_days.sum()
        n_deaths = death_days.notna().sum()
        hazard = n_deaths / total_person_days if total_person_days > 0 else 0.0001
        hazard = np.clip(hazard, 1e-8, 1 - 1e-8)  # Keep hazard within realistic bounds
        age_hazard[age] = hazard
    return age_hazard

# === Simulate unbiased death day (in days since START_DATE) using exponential hazard ===
def generate_unbiased_death_days(hazard, end_measure, n):
    
    # Simulate death days with a constant hazard rate per day using exponential distribution.
    # Death days are censored at end_measure (no death after that).

    # Sample death times from exponential distribution with rate=hazard
    death_days = np.random.exponential(scale=1/hazard, size=n)

    # Censor deaths beyond end_measure
    death_days = np.where(death_days <= end_measure, death_days, np.nan)

    return death_days

# === Simulate new unbiased death days for one age group ===
def simulate_for_age_group(df_age, end_measure, hazard):
    if df_age.empty:
        return df_age

    age = df_age["Age"].iloc[0]
    print(f"Simulating age {age} (no bias, age-specific hazard)...")

    # Simulate death day using exponential hazard
    death_days_sim = generate_unbiased_death_days(hazard, end_measure, len(df_age))
    df_age = df_age.copy()
    df_age["death_day_sim"] = death_days_sim

    # Calculate day number of latest dose for each individual
    dose_days = pd.DataFrame({col: to_day_number(df_age[col]) for col in DOSE_DATE_COLS})
    last_dose_day = dose_days.max(axis=1).fillna(-1)

    # Optional logic to censor doses after death - commented out by design
    # mask_death_before_last_dose = df_age["death_day_sim"] < last_dose_day
    # for col in DOSE_DATE_COLS:
    #     df_age.loc[mask_death_before_last_dose, col] = pd.NaT

    # for col in DOSE_DATE_COLS:
    #     dose_day_numbers = to_day_number(df_age[col])
    #     valid_mask = (df_age["death_day_sim"].isna()) | (df_age["death_day_sim"] >= dose_day_numbers)
    #     valid_mask = valid_mask | mask_death_before_last_dose
    #     df_age.loc[~valid_mask, col] = pd.NaT

    # Convert simulated death day back to string date format or blank if censored
    df_age["DatumUmrti"] = df_age["death_day_sim"].apply(
        lambda x: (START_DATE + pd.Timedelta(days=int(x))).strftime('%Y-%m-%d') if not pd.isna(x) else ''
    )
    df_age.drop(columns=["death_day_sim"], inplace=True)

    print(f"Finished age {age}")
    return df_age

# === Wrapper for multiprocessing map ===
def simulate_for_age_group_wrapper(args):
    df_age, end_measure, hazard = args
    return simulate_for_age_group(df_age, end_measure, hazard)

# === Main script entry point ===
def main():
    print("Loading CSV...")
    df = pd.read_csv(INPUT_CSV, usecols=NEEDED_COLS, low_memory=False)

    print("Parsing dates and calculating ages...")
    df = parse_dates(df)
    df = calculate_age(df)
    df = df[df["Age"].between(0, MAX_AGE)]

    print("Calculating end_measure...")
    df["death_day"] = to_day_number(df["DatumUmrti"])
    end_measure = int(df["death_day"].dropna().max())  # Last observed death day
    print(f"End measure = {end_measure}")

    print("Estimating hazard per age group...")
    age_hazards = estimate_age_specific_hazards(df, end_measure)

    # Prepare data for multiprocessing
    age_groups = [(group, end_measure, age_hazards[age]) for age, group in df.groupby("Age")]

    print(f"Starting multiprocessing with {NUM_WORKERS} workers...")
    with mp.Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(simulate_for_age_group_wrapper, age_groups), total=len(age_groups)))

    print("Concatenating results and saving CSV...")
    df_result = pd.concat(results)

    # Convert all dose columns to ISO date strings (blank if NaT)
    for col in DOSE_DATE_COLS:
        df_result[col] = pd.to_datetime(df_result[col], errors='coerce').dt.strftime('%Y-%m-%d')
        df_result[col] = df_result[col].fillna('')

    df_result.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Output saved to: {OUTPUT_CSV}")

# === Run main if this script is executed ===
if __name__ == "__main__":
    main()
