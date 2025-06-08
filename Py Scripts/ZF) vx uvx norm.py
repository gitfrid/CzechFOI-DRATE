import pandas as pd
import numpy as np
import plotly.graph_objs as go

# This script reads individual-level simulated vaccination and mortality data,
# classifies the population into vaccinated and unvaccinated groups dynamicaly per day,
# computes death rates per group and age, smooths the time series, and generates
# an interactive Plotly visualization with traces for death counts, death rates,
# population, and dose counts across ages 0–113 and days since 2020-01-01.

# Features:
# - Dynamic per-day vaccination status based on dose dates
# - Death classification into vx/uvx based on current status
# - Normalization of deaths per 100,000 individuals
# - 7-day rolling average smoothing
# - Age-stratified output using Plotly with multiple Y axes

#Input:
# - Czech-FOI CSV file with columns: birth year, death date, up to 7 dose dates

# Output:
# - An interactive HTML Plotly plot visualizing the processed data

# === File Paths ===
INPUT_CSV = "C:\github\CzechFOI-DRATE\TERRA\sim_HR_NOBIAS_Vesely_106_202403141131.csv" # -> simulated Data to test script for bias
# INPUT_CSV = "C:\github\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv" # -> real Czech-FOI Data
OUTPUT_HTML = r"C:\github\CzechFOI-DRATE\Plot Results\ZF) vx uvx norm\ZF) vx uvx norm sim HR no bias.html"

START_DATE = pd.Timestamp('2020-01-01')  # Day 0 reference
MAX_AGE = 113
REFERENCE_YEAR = 2023  # Used to compute current age from birth year

# === Load and Prepare Data ===
dose_date_cols = [f'Datum_{i}' for i in range(1, 8)]  # Dose date column names
needed_cols = ['Rok_narozeni', 'DatumUmrti'] + dose_date_cols

df = pd.read_csv(
    INPUT_CSV,
    usecols=needed_cols,
    parse_dates=['DatumUmrti'] + dose_date_cols,
    dayfirst=False,
    low_memory=False
)

# Normalize column names to lowercase
df.columns = [col.strip().lower() for col in df.columns]
dose_date_cols_lower = [col.lower() for col in dose_date_cols]

# Compute current age
df['birth_year'] = pd.to_numeric(df['rok_narozeni'], errors='coerce')
df['age'] = REFERENCE_YEAR - df['birth_year']
df = df[df['age'].between(0, MAX_AGE)].copy()  # Keep only valid ages

# Convert dates to "days since START_DATE"
def to_day_number(date_series):
    return (date_series - START_DATE).dt.days

df['death_day'] = to_day_number(df['datumumrti'])
for col in dose_date_cols_lower:
    df[col + '_day'] = to_day_number(df[col])

# First dose is the minimum of all dose days
df['first_dose_day'] = df[[col + '_day' for col in dose_date_cols_lower]].min(axis=1, skipna=True)
df['has_any_dose'] = df[[col + '_day' for col in dose_date_cols_lower]].notna().any(axis=1)

# === Simulation Time Frame and Data Structures ===
END_MEASURE = int(df['death_day'].dropna().max())
days = np.arange(0, END_MEASURE + 1)
ages = np.arange(0, MAX_AGE + 1)

# Structure for storing results per day/age
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

# Pre-split dataset by age
df_age_groups = [df[df['age'] == age] for age in ages]

# === Main Loop Over Ages and Days ===
for age, sub in zip(ages, df_age_groups):
    if sub.empty:
        continue

    death_days = sub['death_day'].values
    first_dose_days = sub['first_dose_day'].values
    has_any_dose = sub['has_any_dose'].values

    for day in days:
        alive_mask = np.isnan(death_days) | (death_days > day)  # Still alive
        death_today_mask = (death_days == day)  # Died today

        # Vaccination status is determined by day >= first dose
        is_vaxed = (day >= first_dose_days) & has_any_dose
        is_uvx = ~is_vaxed

        pop_vx = np.sum(alive_mask & is_vaxed)
        pop_uvx = np.sum(alive_mask & is_uvx)
        pop_total = pop_vx + pop_uvx

        death_vx = np.sum(death_today_mask & is_vaxed)
        death_uvx = np.sum(death_today_mask & is_uvx)
        death_total = death_vx + death_uvx

        # Store results
        results['day'].append(day)
        results['age'].append(age)
        results['pop_vx'].append(pop_vx)
        results['pop_uvx'].append(pop_uvx)
        results['pop_total'].append(pop_total)
        results['death_vx'].append(death_vx)
        results['death_uvx'].append(death_uvx)
        results['death_total'].append(death_total)

# === Normalize and Smooth ===
result_df = pd.DataFrame(results)
result_df['death_vx_norm'] = (result_df['death_vx'] / result_df['pop_vx'].replace(0, np.nan)) * 100_000
result_df['death_uvx_norm'] = (result_df['death_uvx'] / result_df['pop_uvx'].replace(0, np.nan)) * 100_000
result_df['death_total_norm'] = (result_df['death_total'] / result_df['pop_total'].replace(0, np.nan)) * 100_000
result_df.fillna(0, inplace=True)  # Replace NaNs with 0

# Apply 7-day centered rolling mean per age
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

# === Count Dose Events ===
first_dose_counts_age = {age: pd.Series(0, index=days, dtype=float) for age in ages}
all_dose_counts_age = {age: pd.Series(0, index=days, dtype=float) for age in ages}

for age, sub in zip(ages, df_age_groups):
    if sub.empty:
        continue

    # First dose counts
    first_counts = sub['first_dose_day'].value_counts().dropna().astype(int)
    s_first = pd.Series(0, index=days, dtype=float)
    s_first.update(first_counts)
    first_dose_counts_age[age] = s_first

    # All dose counts
    all_dose_days = pd.concat([sub[col + '_day'] for col in dose_date_cols_lower])
    all_counts = all_dose_days.value_counts().dropna().astype(int)
    s_all = pd.Series(0, index=days, dtype=float)
    s_all.update(all_counts)
    all_dose_counts_age[age] = s_all

# Build DataFrames from dictionaries
first_dose_df = pd.DataFrame(first_dose_counts_age)
all_dose_df = pd.DataFrame(all_dose_counts_age)

# Smooth dose counts
first_dose_df_smooth = first_dose_df.rolling(window=window_size, center=True, min_periods=1).mean()
all_dose_df_smooth = all_dose_df.rolling(window=window_size, center=True, min_periods=1).mean()

# === Plotly Visualization ===
fig = go.Figure()
colors_vx = 'rgba(0,100,255,0.3)'
colors_uvx = 'rgba(255,0,0,0.3)'
colors_total = 'rgba(0,0,0,0.3)'

# Plot all traces per age group
for age in ages:
    df_age = result_df[result_df['age'] == age]
    if df_age.empty:
        continue

    # Norm smooth
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_vx_norm_smooth'],
                             name=f'death_vx_norm_smooth age {age}', yaxis='y1',
                             mode='lines', line=dict(width=1, color=colors_vx), visible='legendonly'))
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_uvx_norm_smooth'],
                             name=f'death_uvx_norm_smooth age {age}', yaxis='y1',
                             mode='lines', line=dict(width=1, color=colors_uvx), visible='legendonly'))
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_total_norm_smooth'],
                             name=f'death_total_norm_smooth age {age}', yaxis='y1',
                             mode='lines', line=dict(width=1, color=colors_total), visible='legendonly'))

    # Norm raw
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_vx_norm'],
                             name=f'death_vx_norm age {age}', yaxis='y1',
                             mode='lines', line=dict(width=1, color=colors_vx.replace('0.3', '0.5')), visible='legendonly'))
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_uvx_norm'],
                             name=f'death_uvx_norm age {age}', yaxis='y1',
                             mode='lines', line=dict(width=1, color=colors_uvx.replace('0.3', '0.5')), visible='legendonly'))
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_total_norm'],
                             name=f'death_total_norm age {age}', yaxis='y1',
                             mode='lines', line=dict(width=1, color=colors_total.replace('0.3', '0.5')), visible='legendonly'))

    # Raw death counts
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_vx'],
                             name=f'death_vx age {age}', yaxis='y2',
                             mode='lines', line=dict(width=1, color=colors_vx.replace('0.3', '0.15')), visible='legendonly'))
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_uvx'],
                             name=f'death_uvx age {age}', yaxis='y2',
                             mode='lines', line=dict(width=1, color=colors_uvx.replace('0.3', '0.15')), visible='legendonly'))
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['death_total'],
                             name=f'death_total age {age}', yaxis='y2',
                             mode='lines', line=dict(width=1, color=colors_total.replace('0.3', '0.15')), visible='legendonly'))

    # Population
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['pop_vx'],
                             name=f'pop_vx age {age}', yaxis='y3',
                             mode='lines', line=dict(width=1.5, color=colors_vx.replace('0.3', '0.1')), visible='legendonly'))
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['pop_uvx'],
                             name=f'pop_uvx age {age}', yaxis='y3',
                             mode='lines', line=dict(width=1.5, color=colors_uvx.replace('0.3', '0.1')), visible='legendonly'))
    fig.add_trace(go.Scatter(x=df_age['day'], y=df_age['pop_total'],
                             name=f'pop_total age {age}', yaxis='y3',
                             mode='lines', line=dict(width=1.5, color=colors_total.replace('0.3', '0.1')), visible='legendonly'))

    # Dose counts
    fig.add_trace(go.Scatter(x=days, y=first_dose_df_smooth[age],
                             name=f'First Dose Count (7-day rolling) age {age}', yaxis='y4',
                             mode='lines', line=dict(width=1.5, color='green'), visible='legendonly'))
    fig.add_trace(go.Scatter(x=days, y=all_dose_df_smooth[age],
                             name=f'All Doses Count (7-day rolling) age {age}', yaxis='y4',
                             mode='lines', line=dict(width=1.5, color='orange'), visible='legendonly'))

# === Export Plot ===
fig.update_layout(title='Deaths, Population, and Dose Counts by Vaccination Status and Age',
                  xaxis_title='Days since 2020-01-01',
                  yaxis=dict(title='Deaths per 100k (normalized)', side='left'),
                  yaxis2=dict(title='Raw death counts', overlaying='y', side='right'),
                  yaxis3=dict(title='Population size', overlaying='y', anchor='free', side='left', position=0.05),
                  yaxis4=dict(title='Dose counts', overlaying='y', anchor='free', side='right', position=0.95),
                  legend=dict(orientation='h', y=-0.3),
                  height=800)

fig.write_html(OUTPUT_HTML)
