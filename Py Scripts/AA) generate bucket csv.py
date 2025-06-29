import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import csv

"""
Method Description:
-------------------
This script processes individual-level vaccination and death data for a specified age group
to generate aggregated "bucket" data representing person-days and deaths grouped by:

- age
- weeks since first vaccine dose
- calendar month

The goal is to create a dataset suitable for bias-adjusted mortality analyses.

The method iterates through each individual, expanding their observation period from day 0
until their death or censoring date, assigning person-day counts and death events per day.
Each day is categorized by the number of weeks since the first vaccine dose (or -1 if unvaccinated),
and the calendar month. The final output aggregates this detailed daily data into buckets
summarizing total person-days and deaths per age, vaccination timing bucket, and month.

Bucket File Structure (Output CSV):
-----------------------------------
Columns:
- age: integer age (constant for filtered data)
- week_since_first_dose: integer number of weeks since first vaccine dose; -1 indicates pre-vaccination/unvaccinated period
- calendar_month: string YYYY-MM format representing the calendar month of the person-day
- person_days: total count of person-days in that bucket (typically sum of 1-day increments)
- deaths: total count of deaths occurring within that bucket

This file can be used directly for mortality rate calculations and bias adjustment modeling.

"""

# === CONFIGURATION ===

# Input CSV with individual-level dose and death dates
#INPUT_CSV = r"C:\CzechFOI-BUCKET\TERRA\Vesely_106_202403141131.csv"
INPUT_CSV = r"C:\CzechFOI-BUCKET\TERRA\sim_MINBIAS_Vesely_106_202403141131.csv"

# Output aggregated bucket CSV path
OUTPUT_CSV = r"C:\CzechFOI-BUCKET\TERRA\AG70_sim_MINBIAS_bucket_Vesely_106_202403141131.csv"

# Temporary intermediate CSV to store expanded daily data before aggregation
TEMP_CSV = OUTPUT_CSV.replace(".csv", "_daily.csv")  

REFERENCE_YEAR = 2023        # Reference year to compute age
START_DATE = pd.Timestamp("2020-01-01")  # Study start date for relative day calculations
MAX_AGE = 113               # Maximum age (not used here but indicative)
AGE_TO_INCLUDE = 70          # Age group of interest to filter data on

# === Columns to use from input CSV ===
DOSE1_COL = "Datum_1"        # First dose date column
DEATH_COL = "DatumUmrti"     # Death date column
BIRTHYEAR_COL = "Rok_narozeni"  # Birth year column

# === Load data ===

# Dose columns 1 through 7 for parsing dates
dose_cols = [f"Datum_{i}" for i in range(1, 8)]

# Read CSV, parse date columns
df = pd.read_csv(INPUT_CSV, parse_dates=dose_cols + [DEATH_COL], dayfirst=False)

# === Preprocess: compute age, dose1 day, death day ===

# Compute age as difference between reference year and birth year
df['age'] = REFERENCE_YEAR - df[BIRTHYEAR_COL]

# Filter to only include rows with the specified age group
df = df[df['age'] == AGE_TO_INCLUDE].copy()

# Calculate number of days from study start to first dose date
df['dose1_day'] = (pd.to_datetime(df[DOSE1_COL], errors='coerce') - START_DATE).dt.days

# Calculate number of days from study start to death date
df['death_day'] = (pd.to_datetime(df[DEATH_COL], errors='coerce') - START_DATE).dt.days

# For individuals without death date, censor at max observed death day
END_OBS_DAY = df['death_day'].dropna().max()
df['death_day'] = df['death_day'].fillna(END_OBS_DAY)

# === Generate daily person-day rows and write to temp CSV ===

print("⏳ Generating person-day rows (streaming to disk)...")

with open(TEMP_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        'age',
        'week_since_first_dose',
        'calendar_month',
        'death',
        'person_day'
    ])
    writer.writeheader()

    # Iterate over individuals
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        death_day = int(row.death_day)
        dose1_day = row.dose1_day
        age = int(row.age)

        # Generate one row per day from day 0 up to and including death_day (or censor day)
        for day in range(0, death_day + 1):
            # Flag if death occurs on this day
            is_dead_today = (day == death_day)

            # Determine calendar month string for this day (YYYY-MM)
            calendar_month = (START_DATE + pd.Timedelta(days=day)).strftime('%Y-%m')

            # Compute weeks since first dose; -1 if before dose or dose unknown
            week_since_dose = np.floor((day - dose1_day) / 7) if pd.notna(dose1_day) and day >= dose1_day else -1

            # Write daily person-day record
            writer.writerow({
                'age': age,
                'week_since_first_dose': int(week_since_dose),
                'calendar_month': calendar_month,
                'death': int(is_dead_today),
                'person_day': 1
            })

# === Read the expanded daily file and aggregate into buckets ===

print("📊 Aggregating buckets...")

df_days = pd.read_csv(TEMP_CSV)

# Aggregate daily data by age, weeks since first dose, and calendar month
bucket_df = (
    df_days
    .groupby(['age', 'week_since_first_dose', 'calendar_month'], as_index=False)
    .agg({'person_day': 'sum', 'death': 'sum'})  # Sum person-days and deaths in each bucket
    .rename(columns={
        'person_day': 'person_days',
        'death': 'deaths'
    })
)

# === Save the aggregated bucket data to CSV ===

bucket_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Bucket file saved to: {OUTPUT_CSV}")
print(bucket_df.head(10))
