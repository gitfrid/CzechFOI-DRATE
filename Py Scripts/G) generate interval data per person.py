# Script to process vaccination and death date data by age bins,
# creating interval datasets per person with vaccination dose timing,
# and saving interval CSV files for further analysis (for example with cox, dowhy, g-estimate )

# - Reads raw input CSV with birth year, death date, and up to 7 vaccine dose dates.
# - Computes age, filters valid ages, converts dates to days since a reference date.
# - Splits each person's timeline into intervals between doses and death.
# - Outputs interval files grouped by age for downstream time-to-event modeling.


import pandas as pd
import numpy as np
import os

# === Parameters ===
# Path to the input CSV file containing raw vaccination and death data
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
# Alternative input CSVs (comment/uncomment as needed)
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_Vesely_106_SIMULATED.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_TWO_DEATHRATES_10X_Vesely_106_SIMULATED.csv"

# Read only the header row to inspect columns
df_head = pd.read_csv(INPUT_CSV, nrows=0)
print("Columns in CSV:", df_head.columns.tolist())

# Output directory to save generated interval CSV files
OUTPUT_DIR = r"C:\CzechFOI-DRATE\intervals_per_agebin"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output dir if it doesn't exist

# Reference year to compute age from birth year
REFERENCE_YEAR = 2023

# Maximum age allowed in dataset (filter out invalid ages)
MAX_AGE = 113

# Reference start date to convert all dates to numeric days
START_DATE = pd.Timestamp('2020-01-01')

# Specify which ages to process; None or empty list means process all ages
age_bins_to_process = [70]  # e.g. only age 70, or set to [] or None for all

# Dose date columns exactly as in the input CSV (case sensitive)
dose_cols = [f"Datum_{i}" for i in range(1, 8)]

# === Load CSV data ===
# Parse dose columns and death date column as datetime objects
df = pd.read_csv(INPUT_CSV, parse_dates=dose_cols + ['DatumUmrti'], dayfirst=False)

# Normalize column names: lowercase and strip spaces for easier handling
df.columns = df.columns.str.lower().str.strip()

# Also lowercase dose column names for consistency with df.columns
dose_cols = [c.lower() for c in dose_cols]

# === Data preparation ===

# Compute age from birth year relative to REFERENCE_YEAR
df['age'] = REFERENCE_YEAR - pd.to_numeric(df['rok_narozeni'], errors='coerce')

# Filter dataset to keep only rows with valid ages in [0, MAX_AGE]
df = df[df['age'].between(0, MAX_AGE)].copy()

# Helper function to convert datetime columns to integer days since START_DATE
def to_day(series):
    return (series - START_DATE).dt.days

# Convert death date to days since START_DATE
df['death_day'] = to_day(df['datumumrti'])

# Convert each dose date column to days since START_DATE
for c in dose_cols:
    df[c + '_day'] = to_day(df[c])

# Drop rows missing death day, assuming death day is required for intervals
df = df.dropna(subset=['death_day'])

# Optional: filter only specific age bins if requested
if age_bins_to_process:
    df = df[df['age'].isin(age_bins_to_process)]

# === Function to generate intervals per person ===
def generate_intervals(row):
    """
    Given a row for one person, generate a list of interval dicts
    covering unvaccinated and vaccinated periods, separated by dose days,
    and ending at death day.

    Each interval dict contains:
        - person_id: unique identifier for the person (here, row index)
        - age: person's age
        - start_day: interval start (days since START_DATE)
        - end_day: interval end (days since START_DATE)
        - dose_number: 0 for unvaccinated, 1..N for dose intervals
    """
    # Extract dose days present for this person and sort ascending
    doses = [row[c + '_day'] for c in dose_cols if not pd.isna(row[c + '_day'])]
    doses = sorted(doses)
    death_day = row['death_day']

    intervals = []
    prev_day = 0  # timeline start (can adjust if needed)

    # Unvaccinated interval before first dose if any doses exist
    if doses:
        intervals.append({'person_id': row.name, 'age': row['age'],
                          'start_day': prev_day, 'end_day': doses[0], 'dose_number': 0})
        prev_day = doses[0]
    else:
        # No doses: single interval from 0 to death_day as unvaccinated
        intervals.append({'person_id': row.name, 'age': row['age'],
                          'start_day': prev_day, 'end_day': death_day, 'dose_number': 0})
        return intervals

    # Intervals between doses and from last dose to death
    for i in range(len(doses)):
        start = doses[i]
        # End is next dose day or death day if last dose
        end = doses[i + 1] if i + 1 < len(doses) else death_day
        intervals.append({'person_id': row.name, 'age': row['age'],
                          'start_day': start, 'end_day': end, 'dose_number': i + 1})

    return intervals

# === Generate intervals DataFrame for all persons ===
all_intervals = []
for idx, row in df.iterrows():
    all_intervals.extend(generate_intervals(row))

intervals_df = pd.DataFrame(all_intervals)

# === Save intervals grouped by age bin ===
for age_bin, group_df in intervals_df.groupby('age'):
    fname = os.path.join(OUTPUT_DIR, f"real_intervals_age_{int(age_bin)}.csv")
    group_df.to_csv(fname, index=False)
    print(f"Saved intervals for age {age_bin}, rows={len(group_df)} -> {fname}")
