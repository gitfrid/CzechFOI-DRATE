import pandas as pd
import numpy as np
import plotly.graph_objs as go
from collections import defaultdict

# VX vs UVX Mortality Analysis with 7-day Baseline per Age

# This script loads individual-level vaccination and mortality data,
# calculates person-days at risk and deaths by vaccination status (vx = vaccinated, uvx = unvaccinated),
# computes rolling mortality rates and a local baseline for unvaccinated mortality,
# calculates excess mortality for vaccinated individuals, and
# plots mortality rates per age group over time using Plotly.

# === File Paths ===
# Input CSV containing vaccination dates and death dates per individual
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
# Alternative inputs commented out:
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINIMALBIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINIMALBIAS_DEATHRISK_10X_Vesely_106_202403141131.csv"

# Output HTML file path for the Plotly interactive visualization
OUTPUT_HTML = r"C:\CzechFOI-DRATE\Plot Results\Y) vx uvx persondays baslinemortality\Y vx uvx persondays baslinemortality.html"

# === Parameters ===
START_DATE = pd.Timestamp('2020-01-01')  # Reference start date for day calculation
MAX_AGE = 113                           # Maximum age to consider in analysis
REFERENCE_YEAR = 2023                   # Year used to calculate age from birth year
WINDOW_SIZE = 30                       # Window size (days) for rolling mortality rate calculation
BASELINE_TIME_RANGE = 7                # Window size for baseline UVX mortality (±3 days centered)

# === Load Data ===
# Columns representing up to 7 vaccine dose dates (Datum_1 to Datum_7)
dose_date_cols = [f'Datum_{i}' for i in range(1, 8)]
# Columns to read from CSV: birth year, death date, and dose dates
needed_cols = ['Rok_narozeni', 'DatumUmrti'] + dose_date_cols

# Read CSV with date parsing for death and dose columns
df = pd.read_csv(INPUT_CSV, usecols=needed_cols,
                 parse_dates=['DatumUmrti'] + dose_date_cols,
                 dayfirst=False, low_memory=False)

# Normalize column names to lowercase without extra spaces
df.columns = [c.lower().strip() for c in df.columns]

# Lowercase dose date columns for consistent access
dose_cols = [c.lower() for c in dose_date_cols]

# Calculate age from birth year, filter valid ages between 0 and MAX_AGE
df['age'] = REFERENCE_YEAR - pd.to_numeric(df['rok_narozeni'], errors='coerce')
df = df[df['age'].between(0, MAX_AGE)].copy()

# Determine vaccination status: 1 if any dose date present, else 0
df['is_vaxed'] = df[dose_cols].notna().any(axis=1).astype(int)

# Convert datetime columns to integer "days since START_DATE"
def to_day(d):
    """Convert datetime Series to days since START_DATE, preserving NaNs."""
    return (d - START_DATE).dt.days

# Convert death date to death_day (days since start)
df['death_day'] = to_day(df['datumumrti'])

# Convert each dose date to days since start as well
for c in dose_cols:
    df[c + '_day'] = to_day(df[c])

# Compute first dose day as the minimum of all dose days for each individual
df['first_dose_day'] = df[[c + '_day' for c in dose_cols]].min(axis=1, skipna=True)

# === Generate time and age axes for analysis ===
END = int(df['death_day'].dropna().max())  # last observed death day (max)
days = np.arange(END + 1)                   # range of days from 0 to END inclusive
ages = np.arange(0, MAX_AGE + 1)            # age range 0 to MAX_AGE inclusive

# === Aggregate person-day at risk and death counts by day and age ===
records = defaultdict(list)  # dictionary of lists to build final DataFrame

for age in ages:
    # Subset data for current age group
    sub = df[df['age'] == age]
    if sub.empty:
        # Skip ages with no data
        continue

    death_arr = sub['death_day'].values   # array of death days for age group
    is_vx = sub['is_vaxed'].values        # vaccination status array

    for d in days:
        # Determine individuals alive on day d: either no death date or death date after d
        alive = (np.isnan(death_arr) | (death_arr > d))

        # Count alive vaccinated and unvaccinated persons on day d
        vx_alive = np.sum(alive & (is_vx == 1))
        uvx_alive = np.sum(alive & (is_vx == 0))

        # Count deaths on day d among vaccinated and unvaccinated
        vx_deaths = np.sum((death_arr == d) & (is_vx == 1))
        uvx_deaths = np.sum((death_arr == d) & (is_vx == 0))

        # Append computed counts for this age and day to records
        records['day'].append(d)
        records['age'].append(age)
        records['vx_person'].append(vx_alive)
        records['uvx_person'].append(uvx_alive)
        records['vx_deaths'].append(vx_deaths)
        records['uvx_deaths'].append(uvx_deaths)

# Convert aggregated data dictionary to DataFrame
df_result = pd.DataFrame(records)

# === Calculate rolling mortality rates using WINDOW_SIZE days ===
group = df_result.groupby('age')

# Vaccinated rolling sums of deaths and person-days, and rate calculation
df_result['vx_deaths_30d'] = group['vx_deaths'].transform(
    lambda x: x.rolling(WINDOW_SIZE, center=True, min_periods=1).sum())
df_result['vx_person_30d'] = group['vx_person'].transform(
    lambda x: x.rolling(WINDOW_SIZE, center=True, min_periods=1).sum())
df_result['vx_rate'] = df_result['vx_deaths_30d'] / df_result['vx_person_30d']

# Unvaccinated rolling sums of deaths and person-days, and rate calculation
df_result['uvx_deaths_30d'] = group['uvx_deaths'].transform(
    lambda x: x.rolling(WINDOW_SIZE, center=True, min_periods=1).sum())
df_result['uvx_person_30d'] = group['uvx_person'].transform(
    lambda x: x.rolling(WINDOW_SIZE, center=True, min_periods=1).sum())
df_result['uvx_rate'] = df_result['uvx_deaths_30d'] / df_result['uvx_person_30d']

# Total (vx + uvx) rolling deaths and person-days, and total mortality rate
df_result['total_deaths_30d'] = df_result['vx_deaths_30d'] + df_result['uvx_deaths_30d']
df_result['total_person_30d'] = df_result['vx_person_30d'] + df_result['uvx_person_30d']
df_result['total_rate'] = df_result['total_deaths_30d'] / df_result['total_person_30d']

# === Calculate fine-grained baseline mortality rate for unvaccinated group ===
baseline_window = BASELINE_TIME_RANGE  # 7-day centered window (±3 days)
df_result['baseline_uvx_deaths'] = group['uvx_deaths'].transform(
    lambda x: x.rolling(baseline_window, center=True, min_periods=1).sum())
df_result['baseline_uvx_person'] = group['uvx_person'].transform(
    lambda x: x.rolling(baseline_window, center=True, min_periods=1).sum())
df_result['baseline_rate'] = df_result['baseline_uvx_deaths'] / df_result['baseline_uvx_person']

# === Compute excess mortality rate: vaccinated rate minus baseline UVX rate ===
df_result['excess_mortality'] = df_result['vx_rate'] - df_result['baseline_rate']

# === Plotting all mortality rates by age group ===
fig = go.Figure()

for age in ages:
    # Subset data for this age group
    df_a = df_result[df_result['age'] == age]
    if df_a.empty:
        # Skip empty ages
        continue

    # Plot vaccinated mortality rate (blue line)
    fig.add_trace(go.Scatter(
        x=df_a['day'], y=df_a['vx_rate'],
        mode='lines', name=f'vx rate age {age}', visible='legendonly',
        line=dict(color='blue', width=1)))

    # Plot unvaccinated mortality rate (red line)
    fig.add_trace(go.Scatter(
        x=df_a['day'], y=df_a['uvx_rate'],
        mode='lines', name=f'uvx rate age {age}', visible='legendonly',
        line=dict(color='red', width=1)))

    # Plot total mortality rate (orange line)
    fig.add_trace(go.Scatter(
        x=df_a['day'], y=df_a['total_rate'],
        mode='lines', name=f'total rate age {age}', visible='legendonly',
        line=dict(color='orange', width=1)))

    # Plot baseline UVX mortality rate (green dotted line)
    fig.add_trace(go.Scatter(
        x=df_a['day'], y=df_a['baseline_rate'],
        mode='lines', name=f'baseline 7d age {age}', visible='legendonly',
        line=dict(color='green', width=1, dash='dot')))

    # Plot excess mortality rate (black line)
    fig.add_trace(go.Scatter(
        x=df_a['day'], y=df_a['excess_mortality'],
        mode='lines', name=f'excess age {age}', visible='legendonly',
        line=dict(color='black', width=1)))

# Configure layout of the plot
fig.update_layout(
    title='VX vs UVX Mortality with 7-day Baseline per Age',
    xaxis_title='Days since 2020‑01‑01',
    yaxis_title='Mortality rate (deaths per person-day)',
    height=800,
    legend=dict(y=1, x=1, orientation='v')  # legend positioned top-right, vertical
)

# Save plot as interactive HTML file
fig.write_html(OUTPUT_HTML)

# Confirmation message with output path
print(f"Saved to: {OUTPUT_HTML}")
