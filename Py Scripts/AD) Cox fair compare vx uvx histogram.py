import pandas as pd
import numpy as np
from lifelines import CoxTimeVaryingFitter
import warnings
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

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

    CASE 1: Real-world data — death and dose dates read from the Czech FOI dataset 
            Optional simualted minbias dataset with real world dose schedule and constant homogen death rate at aproximate real world level.
            It can read simulated dataset created by NK) generate csv simulate deaths minimal bias.py
            Optional it can read the Cech Freedom of Information raw datfile "Vesely_106_202403141131.csv" 
            to compare simulated datasets with the real world dataset

    CASE 2: Simulated doses assigned with a flat random distribution. 
            Individuals are selected randomly, but only if their death_day > dose_day constraint is satisfied.

    CASE 3: Simulated doses assigned using a bell curve (normal distribution centered within the dose window). 
            Individuals are filtered by the constraint (death_day > dose_day), and then one is randomly selected for dosing.


Plot: added part to plot death distribution for vx uvx total

"""

# ----------------------------
# Configuration parameters
# ----------------------------
DOSE_SCHEDULE = 1  # 1, 2, or 3 as per your specification
FILE_PATH = r"C:\CzechFOI-BUCKET\TERRA\Vesely_106_202403141131.csv"
# FILE_PATH = r"C:\CzechFOI-BUCKET\TERRA\sim_MINBIAS_deathday_gr_doseday_Vesely_106_202403141131.csv"
OUTPUT_HTML = fr"C:\CzechFOI-BUCKET\Plot Results\AD) Cox fair compare vx uvx histogram\AD) sim minbias Cox fair compare vx uvx histogram case{DOSE_SCHEDULE}A.html"

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
        mean_day = (dose_days_range[0] + dose_days_range[1]) // 2
        std_dev = (dose_days_range[1] - dose_days_range[0]) / 6
        sampled_days = []
        while len(sampled_days) < total_target_doses:
            day = int(np.random.normal(loc=mean_day, scale=std_dev))
            if dose_days_range[0] <= day <= dose_days_range[1]:
                sampled_days.append(day)
    else:
        sampled_days = np.random.choice(range(dose_days_range[0], dose_days_range[1] + 1),
                                        size=total_target_doses, replace=True)

    for day in sampled_days:
        if use_bell_curve:
            candidates = np.where((vx_days == -1) & (death_days > day))[0]
            if len(candidates) == 0:
                skip_count += 1
                continue
        else:
            candidates = np.where(vx_days == -1)[0]
            if len(candidates) == 0:
                skip_count += 1
                continue

        chosen = np.random.choice(candidates)
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
# Plot death day histograms for total, vaccinated, and unvaccinated
# ----------------------------
if not df.empty:
    print("Generating histogram comparison of death distributions...")

    deaths = df[df["event"] == 1]
    deaths_total = deaths["stop"]
    deaths_vx = deaths[deaths["vx"] == 1]["stop"]
    deaths_uvx = deaths[deaths["vx"] == 0]["stop"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=deaths_total, name="Total Deaths", opacity=0.5, nbinsx=100))
    fig.add_trace(go.Histogram(x=deaths_vx, name="Vaccinated Deaths", opacity=0.5, nbinsx=100))
    fig.add_trace(go.Histogram(x=deaths_uvx, name="Unvaccinated Deaths", opacity=0.5, nbinsx=100))

    fig.update_layout(
        barmode="overlay",
        title="Death Distributions by Vaccination Status",
        xaxis_title="Day of Death",
        yaxis_title="Count",
        template="plotly_white"
    )
    fig.write_html(OUTPUT_HTML)
    print(f"Saved histogram to: {OUTPUT_HTML}")
