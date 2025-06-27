import pandas as pd
import numpy as np
import os

# === Parameters ===
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_Vesely_106_SIMULATED.csv"
OUTPUT_DIR = r"C:\CzechFOI-DRATE\intervals_per_agebin"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REFERENCE_YEAR = 2023
MAX_AGE = 113
START_DATE = pd.Timestamp('2020-01-01')

# Define end of observation period for alive people (days since START_DATE)
# Example: set to today relative to START_DATE, or fixed date, e.g. 2025-01-01
END_OBS_DATE = pd.Timestamp('2025-01-01')
END_DAY = (END_OBS_DATE - START_DATE).days

age_bins_to_process = []  # empty = all ages

dose_cols = [f"Datum_{i}" for i in range(1, 8)]

# === Load CSV data ===
df = pd.read_csv(INPUT_CSV, parse_dates=dose_cols + ['DatumUmrti'], dayfirst=False)

# Normalize column names
df.columns = df.columns.str.lower().str.strip()
dose_cols = [c.lower() for c in dose_cols]

# === Data preparation ===
df['age'] = REFERENCE_YEAR - pd.to_numeric(df['rok_narozeni'], errors='coerce')
df = df[df['age'].between(0, MAX_AGE)].copy()

def to_day(series):
    return (series - START_DATE).dt.days

df['death_day'] = to_day(df['datumumrti'])
for c in dose_cols:
    df[c + '_day'] = to_day(df[c])

# Do NOT drop rows missing death_day (keep alive people)
# Instead, fill missing death_day with END_DAY (censoring date)
df['death_day'] = df['death_day'].fillna(END_DAY)

if age_bins_to_process:
    df = df[df['age'].isin(age_bins_to_process)]

def generate_intervals(row):
    doses = [row[c + '_day'] for c in dose_cols if not pd.isna(row[c + '_day'])]
    doses = sorted(doses)
    death_day = row['death_day']

    intervals = []
    prev_day = 0  # timeline start

    if doses:
        intervals.append({'person_id': row.name, 'age': row['age'],
                          'start_day': prev_day, 'end_day': doses[0], 'dose_number': 0})
        prev_day = doses[0]
    else:
        intervals.append({'person_id': row.name, 'age': row['age'],
                          'start_day': prev_day, 'end_day': death_day, 'dose_number': 0})
        return intervals

    for i in range(len(doses)):
        start = doses[i]
        end = doses[i + 1] if i + 1 < len(doses) else death_day
        intervals.append({'person_id': row.name, 'age': row['age'],
                          'start_day': start, 'end_day': end, 'dose_number': i + 1})

    return intervals

all_intervals = []
for idx, row in df.iterrows():
    all_intervals.extend(generate_intervals(row))

intervals_df = pd.DataFrame(all_intervals)

# Save all intervals to a single CSV
output_file = os.path.join(OUTPUT_DIR, "minbias_interval_person_all_ages_Vesely_106_202403141131.csv")
intervals_df.to_csv(output_file, index=False)
print(f"Saved all intervals to {output_file}, rows={len(intervals_df)}")
