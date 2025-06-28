
import pandas as pd
import numpy as np
import os

# Generate time intervals between vaccine doses and death (or censoring) per person.

# This script processes individual-level vaccination and death data from a CSV file,
# calculates age from birth year, converts dates to days relative to a start date,
# handles censoring of alive individuals, and generates intervals for each person
# representing time before first dose, between doses, and from last dose to death/censoring.

# Output is a long-format CSV file with one row per interval containing:
# - person_id (row index)
# - age
# - start_day (days since START_DATE)
# - end_day (days since START_DATE)
# - dose_number (0 = before any dose, 1+ = after respective dose)

# Handles four cases per individual:
# 1) Alive, no doses: single interval from day 0 to censoring day (dose_number=0)
# 2) Alive, with doses: intervals split at doses, last interval ends at censoring day
# 3) Died, no doses: single interval from day 0 to death day (dose_number=0)
# 4) Died, with doses: intervals split at doses, last interval ends at death day
# Each interval records vaccination status (dose_number) and event (death or censoring).
# Censoring day = last day of observation (max death day overal AGs) if no death occurred (end of study follow-up).


# === Parameters ===

# Input CSV file paths (uncomment desired one)
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_TWO_DEATHRATES_10X_Vesely_106_SIMULATED.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_TWO_DEATHRATES_10X_Vesely_106_SIMULATED.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"

OUTPUT_DIR = r"C:\CzechFOI-DRATE\intervals_per_agebin"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # create output dir if it doesn't exist

REFERENCE_YEAR = 2023  # Year used to calculate age from birth year
MAX_AGE = 113          # Maximum age allowed for filtering
START_DATE = pd.Timestamp('2020-01-01')  # Reference start date for timeline (day 0)

age_bins_to_process = []  # List of ages to include, empty means all ages

# Dose date columns in the dataset (up to 7 doses)
dose_cols = [f"Datum_{i}" for i in range(1, 8)]

# === Load CSV data ===

# Read CSV with date parsing for dose and death columns
df = pd.read_csv(INPUT_CSV, parse_dates=dose_cols + ['DatumUmrti'], dayfirst=False)

# Normalize column names (lowercase and strip whitespace)
df.columns = df.columns.str.lower().str.strip()
dose_cols = [c.lower() for c in dose_cols]

# === Dynamically determine END_OBS_DATE and END_DAY ===

def compute_end_obs_day(df_deaths, start_date):
    # Compute the end observation day as the max death date relative to start_date.
    # If no deaths recorded, returns 0.
    max_death_date = df_deaths['datumumrti'].dropna().max()
    return (max_death_date - start_date).days if pd.notnull(max_death_date) else 0

# Calculate censoring day as the max death date observed in the data
END_DAY = compute_end_obs_day(df, START_DATE)
END_OBS_DATE = START_DATE + pd.Timedelta(days=END_DAY)

# === Data preparation ===

# Calculate age from birth year column ('rok_narozeni')
df['age'] = REFERENCE_YEAR - pd.to_numeric(df['rok_narozeni'], errors='coerce')

# Filter dataframe to valid age range
df = df[df['age'].between(0, MAX_AGE)].copy()

def to_day(series):
    # Convert datetime series to integer day counts relative to START_DATE.
    return (series - START_DATE).dt.days

# Convert death and dose dates to days relative to START_DATE
df['death_day'] = to_day(df['datumumrti'])
for c in dose_cols:
    df[c + '_day'] = to_day(df[c])

# Fill missing death_day with END_DAY to censor alive individuals at observation end
df['death_day'] = df['death_day'].fillna(END_DAY)

# Filter to selected ages if age_bins_to_process is not empty
if age_bins_to_process:
    df = df[df['age'].isin(age_bins_to_process)]

# === Interval generation function ===

def generate_intervals(row):
    
    # Generate list of intervals for a person:
    # - intervals before first dose (dose_number=0)
    # - intervals between doses (dose_number incremented)
    # - last interval ends at death or censoring
    
    # Extract and sort dose days, ignore missing
    doses = [row[c + '_day'] for c in dose_cols if not pd.isna(row[c + '_day'])]
    doses = sorted(doses)
    death_day = row['death_day']

    intervals = []
    prev_day = 0  # start timeline at day 0

    if doses:
        # Interval before first dose
        intervals.append({'person_id': row.name, 'age': row['age'],
                          'start_day': prev_day, 'end_day': doses[0], 'dose_number': 0})
        prev_day = doses[0]
    else:
        # No doses: one interval from day 0 to death/censoring
        intervals.append({'person_id': row.name, 'age': row['age'],
                          'start_day': prev_day, 'end_day': death_day, 'dose_number': 0})
        return intervals

    # Intervals between doses and from last dose to death/censoring
    for i in range(len(doses)):
        start = doses[i]
        end = doses[i + 1] if i + 1 < len(doses) else death_day
        intervals.append({'person_id': row.name, 'age': row['age'],
                          'start_day': start, 'end_day': end, 'dose_number': i + 1})

    return intervals

# === Process all rows to generate intervals ===

all_intervals = []
for idx, row in df.iterrows():
    all_intervals.extend(generate_intervals(row))

# Convert all intervals to DataFrame
intervals_df = pd.DataFrame(all_intervals)

# === Save intervals to CSV ===

output_file = os.path.join(OUTPUT_DIR, "minbias_TWO_DEATHRATES_10X_interval_person_all_ages_Vesely_106_202403141131.csv")
intervals_df.to_csv(output_file, index=False)

print(f"Saved all intervals to {output_file}, rows={len(intervals_df)}")
