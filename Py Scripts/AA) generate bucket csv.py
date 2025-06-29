import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import csv

# === CONFIGURATION ===

# INPUT_CSV = r"C:\CzechFOI-BUCKET\TERRA\Vesely_106_202403141131.csv"
INPUT_CSV = r"C:\CzechFOI-BUCKET\TERRA\sim_MINBIAS_Vesely_106_202403141131.csv"
OUTPUT_CSV = r"C:\CzechFOI-BUCKET\TERRA\AG70_sim_MINBIAS_bucket_Vesely_106_202403141131.csv"

TEMP_CSV = OUTPUT_CSV.replace(".csv", "_daily.csv")  # intermediate daily file

REFERENCE_YEAR = 2023
START_DATE = pd.Timestamp("2020-01-01")
MAX_AGE = 113
AGE_TO_INCLUDE = 70

# === Columns to use ===
DOSE1_COL = "Datum_1"
DEATH_COL = "DatumUmrti"
BIRTHYEAR_COL = "Rok_narozeni"

# === Load data ===

dose_cols = [f"Datum_{i}" for i in range(1, 8)]
df = pd.read_csv(INPUT_CSV, parse_dates=dose_cols + [DEATH_COL], dayfirst=False)

# === Preprocess: age, dose1, death ===

df['age'] = REFERENCE_YEAR - df[BIRTHYEAR_COL]
df = df[df['age'] == AGE_TO_INCLUDE].copy()

df['dose1_day'] = (pd.to_datetime(df[DOSE1_COL], errors='coerce') - START_DATE).dt.days
df['death_day'] = (pd.to_datetime(df[DEATH_COL], errors='coerce') - START_DATE).dt.days

# Replace missing death with censoring day (max observed death)
END_OBS_DAY = df['death_day'].dropna().max()
df['death_day'] = df['death_day'].fillna(END_OBS_DAY)

# === Generate and stream person-day rows to disk ===

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

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        death_day = int(row.death_day)
        dose1_day = row.dose1_day
        age = int(row.age)

        for day in range(0, death_day + 1):
            is_dead_today = (day == death_day)
            calendar_month = (START_DATE + pd.Timedelta(days=day)).strftime('%Y-%m')
            week_since_dose = np.floor((day - dose1_day) / 7) if pd.notna(dose1_day) and day >= dose1_day else -1

            writer.writerow({
                'age': age,
                'week_since_first_dose': int(week_since_dose),
                'calendar_month': calendar_month,
                'death': int(is_dead_today),
                'person_day': 1
            })

# === Convert to DataFrame and bucket ===

print("📊 Aggregating buckets...")
df_days = pd.read_csv(TEMP_CSV)

bucket_df = (
    df_days
    .groupby(['age', 'week_since_first_dose', 'calendar_month'], as_index=False)
    .agg({'person_day': 'sum', 'death': 'sum'})
    .rename(columns={
        'person_day': 'person_days',
        'death': 'deaths'
    })
)

# === Save output ===

bucket_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Bucket file saved to: {OUTPUT_CSV}")
print(bucket_df.head(10))
