import pandas as pd
import numpy as np
import plotly.graph_objs as go
from collections import defaultdict

# -----------------------------------------------------------------------------
# This script processes Czech mortality and vaccination data by age group.
# It computes daily person-days at risk and deaths for vaccinated (vx) and
# unvaccinated (uvx) individuals, calculates rolling 30-day mortality rates,
# and visualizes these rates using Plotly.
#
# Inputs:
# - CSV file with birth year, death dates, and up to 7 vaccination dose dates.
#
# Outputs:
# - Interactive Plotly HTML file showing mortality rates over time by age.
#
# Key steps:
# - Load and preprocess data (calculate age, vaccination status, day numbers)
# - Aggregate person-days and deaths per day and age group for vx and uvx
# - Compute rolling 30-day mortality rates and excess mortality (vx - uvx)
# - Plot mortality rates per age and totals with Plotly
# -----------------------------------------------------------------------------

# === File Paths ===
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
# Alternative input CSVs for simulations (commented out):
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINIMALBIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINIMALBIAS_DEATHRISK_10X_Vesely_106_202403141131.csv"

OUTPUT_HTML = r"C:\CzechFOI-DRATE\Plot Results\Y) vx uvx persondays\Y) vx uvx persondays mortality.html"

# === Parameters ===
START_DATE = pd.Timestamp('2020-01-01')  # Reference start date for day counts
MAX_AGE = 113  # Maximum age to include in analysis
REFERENCE_YEAR = 2023  # Year to calculate age from birth year
WINDOW_SIZE = 30  # Rolling window size (days) for smoothing mortality rates

# === Load Data ===
# Columns for vaccination dose dates (up to 7 doses)
dose_date_cols = [f'Datum_{i}' for i in range(1, 8)]
# Columns to read from CSV: birth year, death date, and dose dates
needed_cols = ['Rok_narozeni', 'DatumUmrti'] + dose_date_cols

# Read CSV with specified columns, parse dates for death and dose columns
df = pd.read_csv(
    INPUT_CSV,
    usecols=needed_cols,
    parse_dates=['DatumUmrti'] + dose_date_cols,
    dayfirst=False,
    low_memory=False
)

# Normalize column names to lowercase and strip spaces
df.columns = [col.strip().lower() for col in df.columns]
dose_date_cols_lower = [col.lower() for col in dose_date_cols]

# Calculate age based on reference year and filter to valid age range
df['birth_year'] = pd.to_numeric(df['rok_narozeni'], errors='coerce')
df['age'] = REFERENCE_YEAR - df['birth_year']
df = df[df['age'].between(0, MAX_AGE)].copy()

# Determine vaccination status: 1 if any dose date is present, else 0
df['is_vaxed'] = df[dose_date_cols_lower].notna().any(axis=1).astype(int)

# Helper function to convert dates to integer day numbers since START_DATE
def to_day_number(date_series):
    return (date_series - START_DATE).dt.days

# Convert death date to day number
df['death_day'] = to_day_number(df['datumumrti'])
# Convert each dose date column to day number with suffix '_day'
for col in dose_date_cols_lower:
    df[col + '_day'] = to_day_number(df[col])
# Find the earliest dose day for each person (first dose day)
df['first_dose_day'] = df[[col + '_day' for col in dose_date_cols_lower]].min(axis=1, skipna=True)

# Determine the last day to measure (max death day in data)
END_MEASURE = int(df['death_day'].dropna().max())
days = np.arange(0, END_MEASURE + 1)  # All days from 0 to END_MEASURE
ages = np.arange(0, MAX_AGE + 1)      # All ages from 0 to MAX_AGE

# === Compute Person-Days and Deaths ===
print("Computing person-days and deaths...")
# Dictionary to collect computed records for DataFrame construction
records = defaultdict(list)

# Loop over each age group
for age in ages:
    sub = df[df['age'] == age]  # Filter data for current age
    if sub.empty:
        continue  # Skip if no data for this age

    death_days = sub['death_day'].values  # Death day per person
    is_vaxed = sub['is_vaxed'].values     # Vaccination status per person

    # Loop over each day to count alive persons and deaths
    for day in days:
        # Alive if death_day is NaN or occurs after current day
        alive_mask = np.isnan(death_days) | (death_days > day)

        # Vaccinated persons alive on current day
        vx_mask = alive_mask & (is_vaxed == 1)
        # Unvaccinated persons alive on current day
        uvx_mask = alive_mask & (is_vaxed == 0)

        # Deaths occurring exactly on current day
        deaths_today = sub['death_day'] == day
        vx_deaths = np.sum(deaths_today & (is_vaxed == 1))
        uvx_deaths = np.sum(deaths_today & (is_vaxed == 0))

        # Record daily stats for current age
        records['day'].append(day)
        records['age'].append(age)
        records['vx_person'].append(np.sum(vx_mask))   # Vaccinated person-days alive
        records['uvx_person'].append(np.sum(uvx_mask)) # Unvaccinated person-days alive
        records['vx_deaths'].append(vx_deaths)         # Vaccinated deaths today
        records['uvx_deaths'].append(uvx_deaths)       # Unvaccinated deaths today

# Convert collected records into a DataFrame
df_result = pd.DataFrame(records)

# === Compute Rolling 30-day Mortality Rates per Age ===
def compute_rolling_mortality(df_result):
    """
    Calculate rolling sums of deaths and person-days over WINDOW_SIZE days per age,
    then compute mortality rates and excess mortality (vx - uvx).
    """
    # Rolling sum of vaccinated deaths per age group
    df_result['vx_deaths_30d'] = df_result.groupby('age')['vx_deaths'].transform(
        lambda x: x.rolling(WINDOW_SIZE, center=True, min_periods=1).sum()
    )
    # Rolling sum of unvaccinated deaths per age group
    df_result['uvx_deaths_30d'] = df_result.groupby('age')['uvx_deaths'].transform(
        lambda x: x.rolling(WINDOW_SIZE, center=True, min_periods=1).sum()
    )
    # Rolling sum of vaccinated person-days alive per age group
    df_result['vx_person_30d'] = df_result.groupby('age')['vx_person'].transform(
        lambda x: x.rolling(WINDOW_SIZE, center=True, min_periods=1).sum()
    )
    # Rolling sum of unvaccinated person-days alive per age group
    df_result['uvx_person_30d'] = df_result.groupby('age')['uvx_person'].transform(
        lambda x: x.rolling(WINDOW_SIZE, center=True, min_periods=1).sum()
    )
    # Calculate vaccinated mortality rate: deaths / person-days
    df_result['vx_rate'] = df_result['vx_deaths_30d'] / df_result['vx_person_30d']
    # Calculate unvaccinated mortality rate
    df_result['uvx_rate'] = df_result['uvx_deaths_30d'] / df_result['uvx_person_30d']
    # Calculate excess mortality: difference between vaccinated and unvaccinated rates
    df_result['excess_mortality'] = df_result['vx_rate'] - df_result['uvx_rate']
    return df_result

# Apply rolling mortality rate calculations
df_result = compute_rolling_mortality(df_result)

# === Add Total Aggregates (across all ages) ===
# Sum deaths and person-days across all ages for each day
df_total = df_result.groupby('day').agg({
    'vx_deaths': 'sum',
    'uvx_deaths': 'sum',
    'vx_person': 'sum',
    'uvx_person': 'sum'
}).reset_index()

# Compute rolling sums for total aggregated data
df_total['vx_deaths_30d'] = df_total['vx_deaths'].rolling(WINDOW_SIZE, center=True, min_periods=1).sum()
df_total['uvx_deaths_30d'] = df_total['uvx_deaths'].rolling(WINDOW_SIZE, center=True, min_periods=1).sum()
df_total['vx_person_30d'] = df_total['vx_person'].rolling(WINDOW_SIZE, center=True, min_periods=1).sum()
df_total['uvx_person_30d'] = df_total['uvx_person'].rolling(WINDOW_SIZE, center=True, min_periods=1).sum()

# Calculate total mortality rates and excess mortality
df_total['vx_rate'] = df_total['vx_deaths_30d'] / df_total['vx_person_30d']
df_total['uvx_rate'] = df_total['uvx_deaths_30d'] / df_total['uvx_person_30d']
df_total['excess_mortality'] = df_total['vx_rate'] - df_total['uvx_rate']

# === Plotting ===
print("Generating Plotly figure...")

fig = go.Figure()

# === Per-Age Mortality Rate Traces (vaccinated, unvaccinated, excess) ===
for age in ages:
    df_age = df_result[df_result['age'] == age]
    if df_age.empty:
        continue

    # Add vaccinated mortality rate trace (blue, thin line, initially hidden)
    fig.add_trace(go.Scatter(
        x=df_age['day'], y=df_age['vx_rate'],
        mode='lines', name=f'vx mortality age {age}',
        line=dict(color='blue', width=1), visible='legendonly'
    ))
    # Add unvaccinated mortality rate trace (red, thin line, initially hidden)
    fig.add_trace(go.Scatter(
        x=df_age['day'], y=df_age['uvx_rate'],
        mode='lines', name=f'uvx mortality age {age}',
        line=dict(color='red', width=1), visible='legendonly'
    ))
    # Add excess mortality trace (black, thin line, initially hidden)
    fig.add_trace(go.Scatter(
        x=df_age['day'], y=df_age['excess_mortality'],
        mode='lines', name=f'excess mortality age {age}',
        line=dict(color='black', width=1), visible='legendonly'
    ))

# === Add Total Mortality Rate Plot ===
# Vaccinated total mortality rate (blue, thicker line)
fig.add_trace(go.Scatter(
    x=df_total['day'], y=df_total['vx_rate'],
    mode='lines', name='Total vx mortality rate',
    line=dict(color='blue', width=4)
))
# Unvaccinated total mortality rate (red, thicker line)
fig.add_trace(go.Scatter(
    x=df_total['day'], y=df_total['uvx_rate'],
    mode='lines', name='Total uvx mortality rate',
    line=dict(color='red', width=4)
))
# Total excess mortality (black, thicker dashed line)
fig.add_trace(go.Scatter(
    x=df_total['day'], y=df_total['excess_mortality'],
    mode='lines', name='Total excess mortality (vx - uvx)',
    line=dict(color='black', width=4, dash='dash')
))

# === Layout ===
fig.update_layout(
    title='30-Day Rolling Mortality Rates: Vaccinated vs Unvaccinated',
    xaxis_title='Days Since 2020-01-01',
    yaxis_title='Mortality Rate (deaths per person-day)',
    height=800,
    legend=dict(orientation='v', x=1, y=1)
)

# === Output ===
fig.write_html(OUTPUT_HTML)
print(f"Saved plot to:\n{OUTPUT_HTML}")
