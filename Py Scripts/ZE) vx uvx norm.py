import pandas as pd
import numpy as np
import plotly.graph_objs as go

# Vaccine Death Rate Visualization Script
# ---------------------------------------

# This script processes an individual-level dataset containing vaccination dates,
# death dates, and year of birth, then calculates and visualizes:

# 1. Daily population counts for vaccinated, unvaccinated, and total.
# 2. Daily death counts for the same groups.
# 3. Normalized death rates per 100,000 individuals per day.
# 4. Rolling averages to smooth trends.
# 5. Daily counts of first and all vaccine doses.

# Output is a Plotly interactive HTML visualization with toggleable traces for all
# ages (0–113), including:
# - Raw and smoothed normalized death rates for vaccinated (vx), unvaccinated (uvx), and total.
# - Raw death counts and population trends.

# Inputs:
# - INPUT_CSV: path to the source czech-FOI (about 11 mill) individual-level dataset 

# Parameters:
# - REFERENCE_YEAR: used to calculate age from year of birth.
# - START_DATE: day zero of the timeline for all date conversions.

# Usage:
# - Adjust paths and parameters as needed and run locally with Python 3.7+.

# === File Paths ===
# INPUT_CSV = r"C:\github\CzechFOI-DRATE\TERRA\sim_HR_NOBIAS_Vesely_106_202403141131.csv" # -> simulated testdata to check script for bias
INPUT_CSV = r"C:\github\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv" # -> real czech-FOI data
OUTPUT_HTML = r"C:\github\CzechFOI-DRATE\Plot Results\ZE) vx uvx norm\ZE) vx uvx norm.html" 

# === Parameters ===
START_DATE = pd.Timestamp('2020-01-01')
MAX_AGE = 113
REFERENCE_YEAR = 2023  # Used to calculate age

# === Load CSV ===
# Define which columns are relevant (birth year, death date, and up to 7 dose dates)
dose_date_cols = [f'Datum_{i}' for i in range(1, 8)]
needed_cols = ['Rok_narozeni', 'DatumUmrti'] + dose_date_cols

print("Loading CSV...")
df = pd.read_csv(
    INPUT_CSV,
    usecols=needed_cols,
    parse_dates=['DatumUmrti'] + dose_date_cols,  # Parse dates automatically
    dayfirst=False,
    low_memory=False
)

# Normalize column names to lowercase for consistency
df.columns = [col.strip().lower() for col in df.columns]
dose_date_cols_lower = [col.lower() for col in dose_date_cols]

# Calculate age and filter for valid range
df['birth_year'] = pd.to_numeric(df['rok_narozeni'], errors='coerce')
df['age'] = REFERENCE_YEAR - df['birth_year']
df = df[df['age'].between(0, MAX_AGE)].copy()

# Determine vaccination status (any non-null dose field = vaccinated)
df['is_vaxed'] = df[dose_date_cols_lower].notna().any(axis=1).astype(int)

# Convert dates to number of days since START_DATE
def to_day_number(date_series):
    return (date_series - START_DATE).dt.days

df['death_day'] = to_day_number(df['datumumrti'])
for col in dose_date_cols_lower:
    df[col + '_day'] = to_day_number(df[col])

# Determine first dose day for each individual
df['first_dose_day'] = df[[col + '_day' for col in dose_date_cols_lower]].min(axis=1, skipna=True)

# Calculate END_MEASURE as the latest day with a recorded death
END_MEASURE = int(df['death_day'].dropna().max())
print(f"Data ranges from day 0 to day {END_MEASURE} (days since {START_DATE.date()})")

# === Computation ===
days = np.arange(0, END_MEASURE + 1)
ages = np.arange(0, MAX_AGE + 1)

# Storage for final results
results = {
    'day': [],
    'age': [],
    'pop_vx': [],
    'pop_uvx': [],
    'death_vx': [],
    'death_uvx': [],
    'death_total': [],
    'pop_total': [],
}

print("Computing daily stats by age and vaccination status...")

# Group the dataset by age
df_age_groups = [df[df['age'] == age] for age in ages]

# Loop over each age and compute daily population and deaths
for age, sub in zip(ages, df_age_groups):
    if sub.empty:
        continue

    death_days = sub['death_day'].values
    is_vaxed = sub['is_vaxed'].values

    for day in days:
        alive_mask = np.isnan(death_days) | (death_days > day)
        death_today_mask = (death_days == day)

        pop_vx = np.sum(alive_mask & (is_vaxed == 1))
        pop_uvx = np.sum(alive_mask & (is_vaxed == 0))
        pop_total = pop_vx + pop_uvx

        death_vx = np.sum(death_today_mask & (is_vaxed == 1))
        death_uvx = np.sum(death_today_mask & (is_vaxed == 0))
        death_total = death_vx + death_uvx

        results['day'].append(day)
        results['age'].append(age)
        results['pop_vx'].append(pop_vx)
        results['pop_uvx'].append(pop_uvx)
        results['pop_total'].append(pop_total)
        results['death_vx'].append(death_vx)
        results['death_uvx'].append(death_uvx)
        results['death_total'].append(death_total)

# === Normalization ===
print("Calculating normalized death rates...")

# Create dataframe from results and calculate normalized death rates per 100,000 people
result_df = pd.DataFrame(results)
result_df['death_vx_norm'] = (result_df['death_vx'] / result_df['pop_vx'].replace(0, np.nan)) * 100_000
result_df['death_uvx_norm'] = (result_df['death_uvx'] / result_df['pop_uvx'].replace(0, np.nan)) * 100_000
result_df['death_total_norm'] = (result_df['death_total'] / result_df['pop_total'].replace(0, np.nan)) * 100_000
result_df.fillna(0, inplace=True)

# === Smoothing ===
# Apply rolling mean to smooth out daily noise
window_size = 7
result_df['death_vx_norm_smooth'] = result_df.groupby('age')['death_vx_norm'].transform(
    lambda x: x.rolling(window_size, center=True, min_periods=1).mean()
)
result_df['death_uvx_norm_smooth'] = result_df.groupby('age')['death_uvx_norm'].transform(
    lambda x: x.rolling(window_size, center=True, min_periods=1).mean()
)
result_df['death_total_norm_smooth'] = result_df.groupby('age')['death_total_norm'].transform(
    lambda x: x.rolling(window_size, center=True, min_periods=1).mean()
)

# === Dose Counts ===
print("Computing dose counts per age and day...")

# Track first and all dose counts per age and day
first_dose_counts_age = {age: pd.Series(0, index=days, dtype=float) for age in ages}
all_dose_counts_age = {age: pd.Series(0, index=days, dtype=float) for age in ages}

for age, sub in zip(ages, df_age_groups):
    if sub.empty:
        continue

    # First doses
    first_counts = sub['first_dose_day'].value_counts().dropna().astype(int)
    s_first = pd.Series(0, index=days, dtype=float)
    s_first.update(first_counts)
    first_dose_counts_age[age] = s_first

    # All doses
    all_dose_days = pd.concat([sub[col + '_day'] for col in dose_date_cols_lower])
    all_counts = all_dose_days.value_counts().dropna().astype(int)
    s_all = pd.Series(0, index=days, dtype=float)
    s_all.update(all_counts)
    all_dose_counts_age[age] = s_all

# Combine into DataFrames and apply smoothing
first_dose_df = pd.DataFrame(first_dose_counts_age)
all_dose_df = pd.DataFrame(all_dose_counts_age)
first_dose_df_smooth = first_dose_df.rolling(window=window_size, center=True, min_periods=1).mean()
all_dose_df_smooth = all_dose_df.rolling(window=window_size, center=True, min_periods=1).mean()

# === Plotly Visualization ===
print("Building Plotly figure...")

fig = go.Figure()

# Define color templates
colors_vx = 'rgba(0,100,255,0.3)'
colors_uvx = 'rgba(255,0,0,0.3)'
colors_total = 'rgba(0,0,0,0.3)'

# Add a full set of traces per age group
for age in ages:
    df_age = result_df[result_df['age'] == age]
    if df_age.empty:
        continue

    # Add smoothed normalized death rates
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_vx_norm_smooth'],
                             name=f'death_vx_norm_smooth age {age}', yaxis='y1',
                             mode='lines', line=dict(width=1, color=colors_vx), visible='legendonly'))

    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_uvx_norm_smooth'],
                             name=f'death_uvx_norm_smooth age {age}', yaxis='y1',
                             mode='lines', line=dict(width=1, color=colors_uvx), visible='legendonly'))

    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_total_norm_smooth'],
                             name=f'death_total_norm_smooth age {age}', yaxis='y1',
                             mode='lines', line=dict(width=1, color=colors_total), visible='legendonly'))

    # Add raw normalized death rates
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_vx_norm'],
                             name=f'death_vx_norm age {age}', yaxis='y1',
                             mode='lines', line=dict(width=1, color=colors_vx.replace('0.3', '0.5')), visible='legendonly'))

    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_uvx_norm'],
                             name=f'death_uvx_norm age {age}', yaxis='y1',
                             mode='lines', line=dict(width=1, color=colors_uvx.replace('0.3', '0.5')), visible='legendonly'))

    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_total_norm'],
                             name=f'death_total_norm age {age}', yaxis='y1',
                             mode='lines', line=dict(width=1, color=colors_total.replace('0.3', '0.5')), visible='legendonly'))

    # Add raw death counts
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_vx'],
                             name=f'death_vx age {age}', yaxis='y2',
                             mode='lines', line=dict(width=1, color=colors_vx.replace('0.3', '0.15')), visible='legendonly'))

    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_uvx'],
                             name=f'death_uvx age {age}', yaxis='y2',
                             mode='lines', line=dict(width=1, color=colors_uvx.replace('0.3', '0.15')), visible='legendonly'))

    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_total'],
                             name=f'death_total age {age}', yaxis='y2',
                             mode='lines', line=dict(width=1, color=colors_total.replace('0.3', '0.15')), visible='legendonly'))

    # Add population counts
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['pop_vx'],
                             name=f'pop_vx age {age}', yaxis='y3',
                             mode='lines', line=dict(width=1.5, color=colors_vx.replace('0.3', '0.1')), visible='legendonly'))

    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['pop_uvx'],
                             name=f'pop_uvx age {age}', yaxis='y3',
                             mode='lines', line=dict(width=1.5, color=colors_uvx.replace('0.3', '0.1')), visible='legendonly'))

    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['pop_total'],
                             name=f'pop_total age {age}', yaxis='y3',
                             mode='lines', line=dict(width=1.5, color=colors_total.replace('0.3', '0.1')), visible='legendonly'))

fig.update_layout(
    title='Vaccinated vs Unvaccinated Deaths, Population, and Doses by Age (Timeline Based on Deaths Only)',
    xaxis=dict(title='Days since 2020-01-01'),
    yaxis=dict(title='Normalized Death Rate / 100k', side='left', autorange=True),
    yaxis2=dict(title='Raw Deaths', overlaying='y', side='right', position=0.95, autorange=True),
    yaxis3=dict(title='Population', overlaying='y', side='right', position=1.0, autorange=True, type='log'),
    yaxis4=dict(title='Dose Counts (7-day rolling)', overlaying='y', side='left', position=0.05, autorange=True),
    template='plotly_white',
    height=900,
    showlegend=True
)

# Save to HTML
print("Saving figure...")
fig.write_html(OUTPUT_HTML)
print(f"Saved Plotly HTML to: {OUTPUT_HTML}")
