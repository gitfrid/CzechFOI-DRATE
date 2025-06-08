import pandas as pd
import numpy as np
import os

# SCRIPT A - Aggregate raw daily data by age group and vaccination status
#    - Loads and cleans individual-level death and vaccine dose dates from raw data
#    - Calculates each person’s age at the reference date (e.g., start of observation)
#    - Determines vaccination status at time of death (vaccinated/unvaccinated)
#    - Constructs the following daily tables (rows = days, columns = age 0–113):
#        1) Total deaths
#        2) Deaths among vaccinated individuals
#        3) Deaths among unvaccinated individuals
#        4) Total population by age (constant over time)
#        5) Daily new vaccinated individuals (first dose)
#        6) Daily decrease in unvaccinated individuals (equal to daily new vaccinated, negated)
#        7) All administered doses and first doses (daily counts)
#    - Saves all tables in CSV format for downstream plotting or analysis

# === CONFIG ===
INPUT_CSV = r"C:\github\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
OUTPUT_DIR = r"C:\github\CzechFOI-DRATE\TERRA"
MAX_AGE = 113
MAX_DAYS = 1533
START_DATE = pd.to_datetime("2020-01-01")
AGE_REFERENCE_DATE = pd.to_datetime("2023-01-01")
DOSE_COLUMNS = [f"Datum_{i}" for i in range(1, 8)]
IMMUNITY_LAG_DAYS = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(INPUT_CSV, parse_dates=["DatumUmrti"] + DOSE_COLUMNS, low_memory=False)

# Exclude individuals who died before their first dose (if they had one)
dose_given = df[DOSE_COLUMNS[0]].notna()
death_before_first_dose = df["DatumUmrti"] < df[DOSE_COLUMNS[0]]
exclude_mask = dose_given & death_before_first_dose

# Filter them out — they did not live long enough to be vaccinated
df = df[~exclude_mask].reset_index(drop=True)
print(f"Excluded {exclude_mask.sum()} individuals who died before receiving their first dose.")

# === COMPUTE AGE AT START ===
df['AgeAtStart'] = (AGE_REFERENCE_DATE.year - df['Rok_narozeni']).clip(0, MAX_AGE).astype(int)

# === DEATH DAY RELATIVE TO START_DATE ===
df['DeathDay'] = (df['DatumUmrti'] - START_DATE).dt.days.clip(lower=0, upper=MAX_DAYS)

# === FIRST DOSE DAY WITH IMMUNITY LAG ===
first_dose_day = (df[DOSE_COLUMNS[0]] - START_DATE).dt.days.fillna(MAX_DAYS + 1).astype(int)
first_dose_day_lagged = first_dose_day + IMMUNITY_LAG_DAYS

# === PREPARE ARRAYS ===
N = len(df)
death_day = df['DeathDay'].values
age = df['AgeAtStart'].values
first_dose = first_dose_day_lagged.values

# === TOTAL (CONSTANT) POPULATION BY AGE ===
pop_total_by_age = np.bincount(age, minlength=MAX_AGE + 1)
population_total = np.tile(pop_total_by_age, (MAX_DAYS + 1, 1))

# === CUMULATIVE FIRST-DOSE VACCINATED BY AGE ===
vx_day = first_dose_day.values.clip(0, MAX_DAYS)
vx_hist = np.zeros((MAX_DAYS + 1, MAX_AGE + 1), dtype=int)
np.add.at(vx_hist, (vx_day, age), 1)
population_vx = np.cumsum(vx_hist, axis=0)

# === UNVACCINATED = TOTAL - VACCINATED ===
population_uvx = population_total - population_vx

# === DAILY CHANGES IN VACCINATED AND UNVACCINATED ===
population_vx_daily = np.diff(population_vx, axis=0, prepend=np.zeros((1, MAX_AGE + 1), dtype=int))
population_uvx_daily = -population_vx_daily

# === DEATH HISTOGRAMS ===
bins_days = np.arange(MAX_DAYS + 2)
bins_age = np.arange(MAX_AGE + 2)

death_hist, _, _ = np.histogram2d(death_day, age, bins=[bins_days, bins_age])

vx_mask = first_dose <= death_day
vx_death_hist, _, _ = np.histogram2d(death_day[vx_mask], age[vx_mask], bins=[bins_days, bins_age])

uvx_death_hist = death_hist - vx_death_hist

# === DOSE HISTOGRAMS ===
all_dose_days = []
first_dose_days = []

for col in DOSE_COLUMNS:
    dose_day = (df[col] - START_DATE).dt.days
    mask = dose_day.between(0, MAX_DAYS)
    all_dose_days.append(np.stack([dose_day[mask], df.loc[mask, 'AgeAtStart']], axis=1))
    if col == DOSE_COLUMNS[0]:
        first_dose_days.append(np.stack([dose_day[mask], df.loc[mask, 'AgeAtStart']], axis=1))

all_dose_arr = np.vstack(all_dose_days)
first_dose_arr = np.vstack(first_dose_days)

all_doses, _, _ = np.histogram2d(all_dose_arr[:, 0], all_dose_arr[:, 1], bins=[bins_days, bins_age])
first_doses, _, _ = np.histogram2d(first_dose_arr[:, 0], first_dose_arr[:, 1], bins=[bins_days, bins_age])

# === SAVE FUNCTION ===
def save_csv(data, filename):
    df_out = pd.DataFrame(data.astype(int), columns=[str(a) for a in range(MAX_AGE + 1)])
    df_out.insert(0, 'DAY', range(data.shape[0]))
    df_out.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)


# Verify that cumulative sum of daily vaccinated matches cumulative vaccinated population
reconstructed_population_vx = np.cumsum(population_vx_daily, axis=0)

if not np.array_equal(reconstructed_population_vx, population_vx):
    print("Warning: population_vx_daily cumulative sum does NOT match population_vx!")
else:
    print("Check passed: population_vx_daily cumulative sum matches population_vx.")

    
# === SAVE OUTPUT FILES ===
print("Saving CSV files...")

save_csv(population_total[:1533], "population_total.csv")
save_csv(population_vx[:1533], "population_vx.csv")
save_csv(population_uvx[:1533], "population_uvx.csv")

# Save daily changes (only days 0 to 1532)
save_csv(population_vx_daily[:1533], "population_vx_daily.csv")
save_csv(population_uvx_daily[:1533], "population_uvx_daily.csv")

save_csv(death_hist[:1533], "deaths_total.csv")
save_csv(vx_death_hist[:1533], "deaths_vx.csv")
save_csv(uvx_death_hist[:1533], "deaths_uvx.csv")

save_csv(all_doses[:1533], "all_doses.csv")
save_csv(first_doses[:1533], "first_doses.csv")

print(f"Script completed successfully with immunity lag = {IMMUNITY_LAG_DAYS}")
