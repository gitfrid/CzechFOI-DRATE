import pandas as pd
import numpy as np
import os

# === Description of output format ===
# Each interval includes:
# - person_id: Row index (unique individual ID)
# - age: Age in REFERENCE_YEAR
# - start_day: Interval start in days since START_DATE
# - end_day: Interval end in days since START_DATE
# - dose_number: 0 = pre-vaccination interval, 1+ = intervals after respective vaccine doses
# - event: 1 if death occurred at interval end (and death day is before censor day), 0 if censored/alive
#
# Censoring occurs at the maximum observed death day in the dataset (END_DAY).
#
# Each individual contributes one or more intervals:
# - Alive, no doses: one interval [0, censor_day], dose_number=0, event=0
# - Alive, with doses: multiple intervals split at doses, last ends at censor_day, event=0
# - Died, no doses: one interval [0, death_day], dose_number=0, event=1
# - Died, with doses: multiple intervals split at doses, last ends at death_day, event=1
#
# This output format supports standard survival analysis frameworks such as Cox regression.

# === Parameters ===

# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv" # real data
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_Vesely_106_SIMULATED.csv" # simulated data rendom const deaths death day >= last dose day 
# NPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_TWO_DEATHRATES_10X_Vesely_106_202403141131.csv"
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_DEATHRISK_2X_Vesely_106_202403141131.csv"

OUTPUT_DIR = r"C:\CzechFOI-DRATE\intervals_per_agebin"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REFERENCE_YEAR = 2023
MAX_AGE = 113
START_DATE = pd.Timestamp('2020-01-01')
age_bins_to_process = [70]
dose_cols = [f"Datum_{i}" for i in range(1, 8)]

# === Load and prepare data ===

df = pd.read_csv(INPUT_CSV, parse_dates=dose_cols + ['DatumUmrti'], dayfirst=False)

df.columns = df.columns.str.lower().str.strip()
dose_cols = [c.lower() for c in dose_cols]

def compute_end_obs_day(df_deaths, start_date):
    max_death_date = df_deaths['datumumrti'].dropna().max()
    return (max_death_date - start_date).days if pd.notnull(max_death_date) else 0

END_DAY = compute_end_obs_day(df, START_DATE)
END_OBS_DATE = START_DATE + pd.Timedelta(days=END_DAY)

df['age'] = REFERENCE_YEAR - pd.to_numeric(df['rok_narozeni'], errors='coerce')
df = df[df['age'].between(0, MAX_AGE)].copy()

date_conversion_summary = {}

def to_day(series):
    original_na = series.isna()
    series_converted = pd.to_datetime(series, errors='coerce')
    conversion_failures = ~original_na & series_converted.isna()
    errors = conversion_failures.sum()
    name = series.name
    total = len(series)
    date_conversion_summary[name] = errors
    if errors > 0 and errors > 0.01 * total:
        print(f"Note: {errors} date conversion failures in column '{name}' ({errors/total:.1%} of rows)")
    return (series_converted - START_DATE).dt.days

df['death_day'] = to_day(df['datumumrti'])
for c in dose_cols:
    df[c + '_day'] = to_day(df[c])

df['death_day'] = df['death_day'].fillna(END_DAY)

if age_bins_to_process:
    df = df[df['age'].isin(age_bins_to_process)]

# === Generate survival intervals per individual ===

skipped_doses_after_censor = 0  # Track how many dose dates are skipped after END_DAY

def generate_intervals(row):
    global skipped_doses_after_censor

    # Extract all dose day values
    original_doses = [row[c + '_day'] for c in dose_cols]
    # Keep only valid and non-NaN dose dates
    valid_doses = [d for d in original_doses if pd.notna(d)]

    # Filter out doses that occur after the censoring date
    doses = []
    for d in valid_doses:
        if d <= END_DAY:
            doses.append(d)
        else:
            skipped_doses_after_censor += 1  # Count skipped doses

    doses = sorted(doses)
    death_day = row['death_day']
    intervals = []

    try:
        if not doses:
            # No doses: one interval from 0 to death/censor day
            intervals.append({
                'person_id': row.name,
                'age': row['age'],
                'start_day': 0,
                'end_day': death_day,
                'dose_number': 0,
                'event': int(death_day < END_DAY)
            })
            return intervals

        # Interval before first dose
        intervals.append({
            'person_id': row.name,
            'age': row['age'],
            'start_day': 0,
            'end_day': doses[0],
            'dose_number': 0,
            'event': int((doses[0] == death_day) and (death_day < END_DAY))
        })

        for i in range(len(doses)):
            start = doses[i]
            end = doses[i + 1] if i + 1 < len(doses) else death_day

            if end < start:
                print(f"❌ Invalid interval: person_id={row.name}, dose_number={i+1}, start_day={start}, end_day={end}")
                print(f"    All dose days: {original_doses}")
                print(f"    Filtered doses: {doses}")
                print(f"    death_day: {death_day}")
                continue

            intervals.append({
                'person_id': row.name,
                'age': row['age'],
                'start_day': start,
                'end_day': end,
                'dose_number': i + 1,
                'event': int((end == death_day) and (death_day < END_DAY))
            })

        return intervals
    except Exception as e:
        print(f"Error generating intervals for person_id={row.name}: {e}")
        return []

all_intervals = []
interval_errors = 0
for idx, row in df.iterrows():
    try:
        intervals = generate_intervals(row)
        if not intervals:
            interval_errors += 1
        all_intervals.extend(intervals)
    except Exception as e:
        print(f"Error processing row {idx}: {e}")
        interval_errors += 1

intervals_df = pd.DataFrame(all_intervals)

# === Save output ===

output_file = os.path.join(OUTPUT_DIR, "AG70_sim_nobias_intervals_for_cox_model_Vesely_106_202403141131.csv")
intervals_df.to_csv(output_file, index=False)

# === Final summary ===

num_individuals = df.shape[0]
num_deaths = intervals_df['event'].sum()

print(f"\n✅ Saved all intervals to: {output_file}")
print(f"👥 Total individuals processed: {num_individuals}")
print(f"📊 Total interval rows: {len(intervals_df)}")
print(f"💀 Total deaths (event == 1): {num_deaths}")
print(f"⚠️  Total interval generation errors: {interval_errors}")
print(f"📅 Skipped doses after censoring day (END_DAY={END_DAY}): {skipped_doses_after_censor}")
print("📅 Date conversion summary:")
for col, err in date_conversion_summary.items():
    print(f"  - {col}: {err} failed conversions")
