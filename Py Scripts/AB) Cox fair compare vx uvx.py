import pandas as pd
import numpy as np
from lifelines import CoxTimeVaryingFitter
import warnings
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler


"""
Survival Analysis of Vaccinated vs. Unvaccinated Mortality Risk Using Time-Varying Cox Regression
-----------------------------------------------------------------------------------------------

Overview:
This script is designed to evaluate survival analysis frameworks under realistic but non-causal assumptions.
It aims to test and validate methods that correct for **immortal time bias**, **selection bias**, and other
confounding factors arising from the non-random assignment of vaccination status in observational datasets.
Such corrections are essential for fair comparisons of mortality risk between vaccinated (vx) and unvaccinated (uvx) groups.

Simulated Dataset:
The script can simulate a population with a fixed probability of death, where vaccination is only allowed 
before death (i.e., **death_day > dose_day**). This creates realistic bias patterns commonly found in 
observational studies, such as immortal time bias and selection bias.

Purpose:
To compare mortality risk (via hazard ratios) between vaccinated and unvaccinated groups using survival analysis.
The goal is to evaluate how different dose assignment strategies affect hazard ratio estimates and whether
bias-correction methods (e.g., time-varying Cox regression) can neutralize these biases. 
A successful method should yield a hazard ratio (HR) close to **1.0**, indicating no difference in death risk between vx and uvx groups.

Objective:
Assess how various vaccination assignment strategies, under the strict condition **death_day > dose_day**, 
influence the estimated hazard ratios for death. This allows testing the robustness of bias-correcting survival models.

Cases:
Three scenarios are analyzed:

    CASE 1: Real-world data — death and dose dates read from the Czech FOI dataset.
            It can read simulated dataset created by NK) generate csv simulate deaths minimal bias.py
            Optional it can read the Cech Freedom of Information raw datfile "Vesely_106_202403141131.csv" 
            to compare simulated datasets with the real world dataset

    CASE 2: Simulated doses assigned with a flat random distribution. 
            Individuals are selected randomly, but only if their death_day > dose_day constraint is satisfied.

    CASE 3: Simulated doses assigned using a bell curve (normal distribution centered within the dose window). 
            Individuals are filtered by the constraint (death_day > dose_day), and then one is randomly selected for dosing.

"""

# ----------------------------
# Configuration parameters
# ----------------------------
DOSE_SCHEDULE = 1  # 1, 2, or 3 as per your specification
FILE_PATH = r"C:\CzechFOI-BUCKET\TERRA\sim_MINBIAS_Vesely_106_202403141131.csv"
REFERENCE_DATE = datetime(2021, 1, 1)
AG70 = 70
T_MAX = 1095
np.random.seed(42)
comparison_plot = True  # Plots comparison of the three


# ----------------------------
# Helper function to convert absolute date string to relative day integer
# ----------------------------
def date_to_day(d):
    try:
        return (pd.to_datetime(d) - REFERENCE_DATE).days
    except Exception:
        return None

# ----------------------------
# Function to assign doses flat (case 2) or bell curve (case 3)
# ----------------------------
def assign_doses_simulated(N, death_days, dose_days_range, total_target_doses, use_bell_curve=False):
    vx_days = np.full(N, -1, dtype=int)
    skip_count = 0
    vaccinated_count = 0

    if use_bell_curve:
        # Bell curve dose days sampling
        mean_day = (dose_days_range[0] + dose_days_range[1]) // 2
        std_dev = (dose_days_range[1] - dose_days_range[0]) / 6  # ~99.7% within range
        sampled_days = []
        while len(sampled_days) < total_target_doses:
            day = int(np.random.normal(loc=mean_day, scale=std_dev))
            if dose_days_range[0] <= day <= dose_days_range[1]:
                sampled_days.append(day)
    else:
        # Flat dose deployment - random uniform sampling
        sampled_days = np.random.choice(range(dose_days_range[0], dose_days_range[1] + 1),
                                        size=total_target_doses, replace=True)

    for day in sampled_days:
        if use_bell_curve:
            # Candidates: unvaccinated AND death day > dose day (strictly greater)
            candidates = np.where((vx_days == -1) & (death_days > day))[0]
            if len(candidates) == 0:
                skip_count += 1
                continue
        else:
            # Candidates: all unvaccinated (no filtering by death day)
            candidates = np.where(vx_days == -1)[0]
            if len(candidates) == 0:
                skip_count += 1
                continue

        chosen = np.random.choice(candidates)
        # For flat assignment, skip if death day <= dose day (strictly less or equal)
        if not use_bell_curve and death_days[chosen] <= day:
            skip_count += 1
            continue

        vx_days[chosen] = day
        vaccinated_count += 1

        if vaccinated_count >= total_target_doses:
            break

    print(f"Simulated dose assignment skipped {skip_count} times due to constraints")
    print(f"Simulated dose assignment assigned {vaccinated_count} doses out of {total_target_doses} target")

    return vx_days

# ----------------------------
# Load or simulate data based on DOSE_SCHEDULE
# ----------------------------
if DOSE_SCHEDULE == 1 and os.path.isfile(FILE_PATH):
    # Case 1: Read real data from CSV (like USE_REAL_DATA=True before)
    print(f"Loading real data from: {FILE_PATH}")
    df_csv = pd.read_csv(FILE_PATH, low_memory=False)
    df_csv["age"] = 2021 - df_csv["Rok_narozeni"]
    df_csv = df_csv[df_csv["age"] == AG70].copy()
    print(f"Loaded {len(df_csv)} rows for age {AG70}")

    data = []
    for idx, row in df_csv.iterrows():
        person_id = idx
        vx_dates = [row[f"Datum_{i}"] for i in range(1, 8)]
        vx_days = sorted([date_to_day(d) for d in vx_dates if pd.notna(d)])
        death_day = date_to_day(row["DatumUmrti"]) if pd.notna(row["DatumUmrti"]) else None

        if not vx_days:
            data.append({
                'id': person_id,
                'start': 0,
                'stop': death_day if death_day is not None else T_MAX,
                'event': int(death_day is not None and death_day <= T_MAX),
                'vx': 0,
                'age': AG70
            })
        else:
            first_vx = vx_days[0]
            if death_day is not None and death_day <= first_vx:
                data.append({
                    'id': person_id,
                    'start': 0,
                    'stop': death_day,
                    'event': 1,
                    'vx': 0,
                    'age': AG70
                })
            else:
                data.append({
                    'id': person_id,
                    'start': 0,
                    'stop': first_vx,
                    'event': 0,
                    'vx': 0,
                    'age': AG70
                })
                # Change here: only assign vaccinated interval if death_day > first_vx
                if death_day is None or death_day > first_vx:
                    data.append({
                        'id': person_id,
                        'start': first_vx,
                        'stop': death_day if death_day is not None else T_MAX,
                        'event': int(death_day is not None and death_day <= T_MAX),
                        'vx': 1,
                        'age': AG70
                    })

    df = pd.DataFrame(data)

else:
    # Cases 2 or 3: Simulate population and dose assignment
    print("Simulating population and dose data...")
    N = 137000
    age = AG70
    death_rate = 0.05

    death_days = np.where(np.random.rand(N) < death_rate,
                          np.random.randint(0, T_MAX, size=N),
                          T_MAX)

    target_vax_rate = 0.80
    total_doses = int(N * target_vax_rate)
    dose_days_range = (350, 500)

    if DOSE_SCHEDULE == 2:
        print("Assigning doses flat and random between day 350-500")
        vx_days = assign_doses_simulated(N, death_days, dose_days_range, total_doses, use_bell_curve=False)
    elif DOSE_SCHEDULE == 3:
        print("Assigning doses by bell curve between day 350-500")
        vx_days = assign_doses_simulated(N, death_days, dose_days_range, total_doses, use_bell_curve=True)
    else:
        raise ValueError("Invalid DOSE_SCHEDULE value. Must be 1, 2, or 3.")

    data = []
    for i in range(N):
        person_id = i
        death_day = death_days[i]
        vx_day = vx_days[i]

        if vx_day == -1:
            data.append({
                'id': person_id,
                'start': 0,
                'stop': death_day,
                'event': int(death_day < T_MAX),
                'vx': 0,
                'age': age
            })
        else:
            # Use death_day > vx_day (strictly greater) here:
            if death_day <= vx_day:
                data.append({
                    'id': person_id,
                    'start': 0,
                    'stop': death_day,
                    'event': 1,
                    'vx': 0,
                    'age': age
                })
            else:
                data.append({
                    'id': person_id,
                    'start': 0,
                    'stop': vx_day,
                    'event': 0,
                    'vx': 0,
                    'age': age
                })
                data.append({
                    'id': person_id,
                    'start': vx_day,
                    'stop': death_day,
                    'event': int(death_day < T_MAX),
                    'vx': 1,
                    'age': age
                })

    df = pd.DataFrame(data)

# ----------------------------
# Diagnostics and summary
# ----------------------------
print("Summary of prepared data:")
print(df['vx'].value_counts())
print(df['event'].value_counts())

# ----------------------------
# Clean data: remove zero or negative duration intervals
# ----------------------------
df = df[df['stop'] > df['start']].copy()

# Add a baseline covariate column of 1.0 for model intercept
df['baseline'] = 1.0

# Define covariates for modeling
covariate_cols = ['vx', 'age', 'baseline']

# Convert covariates to numeric types, coercing errors to NaN
for col in covariate_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Check and drop rows with missing values in covariates after conversion
if df[covariate_cols].isna().any().any():
    print("\nWarning: NaNs found in covariates after conversion. Dropping such rows.")
    df.dropna(subset=covariate_cols, inplace=True)

# Scale covariates to zero mean and unit variance before modeling
scaler = StandardScaler()
df[covariate_cols] = scaler.fit_transform(df[covariate_cols])

# ----------------------------
# Diagnostic printouts to verify data integrity
# ----------------------------
print("\nData snapshot:")
print(df.head(10))

print("\nCovariate unique values (scaled):")
for col in covariate_cols:
    print(f"{col}: min={df[col].min():.3f}, max={df[col].max():.3f}")

print("\nEvent counts and total per vx group:")
print(df.groupby('vx')['event'].agg(['sum', 'count']))

print("\nMissing values per column:")
print(df.isna().sum())

print("\nChecking for duplicated intervals (id, start, stop):")
duplicates = df.duplicated(subset=['id', 'start', 'stop'])
print(f"Number of duplicated intervals: {duplicates.sum()}")

print("\nChecking for zero-duration intervals:")
zero_duration = (df['stop'] - df['start']) == 0
print(f"Number of zero-duration intervals: {zero_duration.sum()}")

print("\nEvent rate by vaccination status:")
event_counts = df.groupby('vx')['event'].sum()
total_counts = df.groupby('vx').size()
event_rates = (event_counts / total_counts).to_frame('event_rate')
print(event_rates)

# ----------------------------
# Fit Cox proportional hazards model with time-varying covariate 'vx'
# Try several penalizer values for numerical stability
# ----------------------------
penalizers_to_try = [0.01, 0.1, 1, 10]
success = False

for penalizer in penalizers_to_try:
    try:
        print(f"\nTrying CoxTimeVaryingFitter with penalizer={penalizer} and formula='vx'")
        ctv = CoxTimeVaryingFitter(penalizer=penalizer)
        with warnings.catch_warnings():
            warnings.simplefilter("default")
            # Fit model using id, start, stop, event columns and formula for covariates
            ctv.fit(df, id_col='id', start_col='start', stop_col='stop', event_col='event',
                    show_progress=True, formula="vx")
        ctv.print_summary()
        success = True
        break
    except Exception as e:
        print(f"Failed with penalizer={penalizer}: {e}")

if not success:
    print("All attempts to fit CoxTimeVaryingFitter failed. Check data integrity.")


