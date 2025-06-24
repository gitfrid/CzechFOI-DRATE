import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from collections import defaultdict


#Script: vx_uvx_personday_excess_mortality.py
# Description:
#    This script processes individual-level vaccination and death data to calculate and visualize
#    vaccinated (VX) vs unvaccinated (UVX) mortality rates by age group over time. It:
#    - Reads dose and death dates from CSV.
#    - Computes person-days and deaths for VX and UVX individuals.
#    - Aggregates data over a time window.
#    - Calculates expected deaths and relative excess mortality.
#    - Plots mortality rates and excess mortality using Plotly.


# === File Paths ===
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_Vesely_106_SIMULATED.csv"
OUTPUT_HTML = r"C:\CzechFOI-DRATE\Plot Results\Y) vx uvx persondays baslinemortality\Y) vx uvx personday_exess_mortality_adjusted.html"

# ============================================
# 0. Setup & Parameters
# ============================================
START_DATE = pd.Timestamp('2020-01-01')        # Start reference date
REFERENCE_YEAR = 2023                          # Used to calculate age from year of birth
MAX_AGE = 113                                  # Max age included in analysis
AGG_WINDOW = 7                                 # Aggregation window in days
IMMUNITY_LAG_DAYS = 0                          # Days after dose to consider immune
MIN_PERSON_DAYS = 100                          # Minimum person-days to consider period stable

# ============================================
# 1. Load Data
# ============================================
# Define dose columns
dose_date_cols = [f'Datum_{i}' for i in range(1, 8)]

# Load CSV with selected columns and parse dates
df = pd.read_csv(
    INPUT_CSV,
    usecols=['Rok_narozeni', 'DatumUmrti'] + dose_date_cols,
    parse_dates=['DatumUmrti'] + dose_date_cols,
    dayfirst=False,
    low_memory=False
)

# Standardize column names
df.columns = [col.lower().strip() for col in df.columns]
dose_cols = [col.lower() for col in dose_date_cols]

# Calculate age from birth year
df['age'] = REFERENCE_YEAR - pd.to_numeric(df['rok_narozeni'], errors='coerce')

# Keep only valid age rows
df = df[df['age'].between(0, MAX_AGE)].copy()

# Convert dates to days since START_DATE
to_day = lambda d: (d - START_DATE).dt.days
df['death_day'] = to_day(df['datumumrti'])
for col in dose_cols:
    df[f"{col}_day"] = to_day(df[col])

# Determine first dose day per individual
df['first_dose_day'] = df[[f"{col}_day" for col in dose_cols]].min(axis=1, skipna=True)

# ============================================
# 2. Define time and age ranges
# ============================================
END_DAY = int(df['death_day'].dropna().max())  # Last day of data
days = np.arange(0, END_DAY + 1)               # Full time range
ages = np.arange(0, MAX_AGE + 1)               # Full age range

# ============================================
# 3. Daily Aggregation
# ============================================
records = defaultdict(list)  # Output storage

for age in ages:
    sub = df[df['age'] == age]  # Subset by age
    if sub.empty:
        continue

    death_arr = sub['death_day'].values
    first_dose_days = sub['first_dose_day'].values

    for d in days:
        alive = np.isnan(death_arr) | (death_arr > d)
        is_vx = (~np.isnan(first_dose_days)) & (d >= first_dose_days + IMMUNITY_LAG_DAYS)

        # Count persons and deaths by vaccination status
        vx_alive = np.sum(alive & is_vx)
        uvx_alive = np.sum(alive & (~is_vx))
        vx_deaths = np.sum((death_arr == d) & is_vx)
        uvx_deaths = np.sum((death_arr == d) & (~is_vx))

        # Store daily data
        records['day'].append(d)
        records['age'].append(age)
        records['vx_person'].append(vx_alive)
        records['uvx_person'].append(uvx_alive)
        records['vx_deaths'].append(vx_deaths)
        records['uvx_deaths'].append(uvx_deaths)

# Convert to DataFrame
df_daily = pd.DataFrame(records)

# ============================================
# 4. Aggregation and Calculations
# ============================================
pio.renderers.default = "browser"  # Use default browser for Plotly

# Aggregate over defined AGG_WINDOW
df_daily['period'] = (df_daily['day'] // AGG_WINDOW).astype(int)
agg_cols = ['vx_person', 'uvx_person', 'vx_deaths', 'uvx_deaths']
df_agg = df_daily.groupby(['period', 'age'])[agg_cols].sum().reset_index()

# Calculate total population and deaths
df_agg['total_person'] = df_agg['vx_person'] + df_agg['uvx_person']
df_agg['total_deaths'] = df_agg['vx_deaths'] + df_agg['uvx_deaths']

# Filter out unstable periods with too few person-days
mask_valid = (df_agg['vx_person'] >= MIN_PERSON_DAYS) & (df_agg['uvx_person'] >= MIN_PERSON_DAYS)
df_agg.loc[~mask_valid, ['vx_person', 'uvx_person', 'vx_deaths', 'uvx_deaths']] = np.nan
df_agg['total_person'] = df_agg['vx_person'] + df_agg['uvx_person']
df_agg['total_deaths'] = df_agg['vx_deaths'] + df_agg['uvx_deaths']

# Compute global baseline mortality rate from UVX population
valid_uvx = df_agg.dropna(subset=['uvx_person', 'uvx_deaths'])
global_uvx_deaths = valid_uvx['uvx_deaths'].sum()
global_uvx_person = valid_uvx['uvx_person'].sum()
global_baseline_rate = global_uvx_deaths / global_uvx_person if global_uvx_person > 0 else np.nan
df_agg['baseline_rate'] = global_baseline_rate

# Compute expected VX deaths and relative excess mortality
df_agg['expected_vx_deaths'] = df_agg['baseline_rate'] * df_agg['vx_person']
df_agg['rel_excess_mortality_%'] = ((df_agg['vx_deaths'] / df_agg['expected_vx_deaths']) - 1) * 100
df_agg.replace([np.inf, -np.inf], np.nan, inplace=True)

# ============================================
# 5. Plotting
# ============================================
fig = go.Figure()

for age in ages:
    df_a = df_agg[df_agg['age'] == age]
    if df_a.empty or df_a[['vx_person', 'uvx_person']].isna().all().all():
        continue

    x_days = df_a['period'] * AGG_WINDOW

    # Compute mortality rates
    vx_rate = df_a['vx_deaths'] / df_a['vx_person']
    uvx_rate = df_a['uvx_deaths'] / df_a['uvx_person']
    total_rate = df_a['total_deaths'] / df_a['total_person']

    # Compute average mortality rates
    mean_vx_rate = (df_a['vx_deaths'].sum() / df_a['vx_person'].sum()) if df_a['vx_person'].sum() > 0 else np.nan
    mean_uvx_rate = (df_a['uvx_deaths'].sum() / df_a['uvx_person'].sum()) if df_a['uvx_person'].sum() > 0 else np.nan
    mean_total_rate = (df_a['total_deaths'].sum() / df_a['total_person'].sum()) if df_a['total_person'].sum() > 0 else np.nan

    # Plot VX mortality rate
    fig.add_trace(go.Scatter(
        x=x_days,
        y=vx_rate,
        mode='lines',
        name=f'VX mortality rate age {age}',
        visible='legendonly',
        line=dict(color='blue', width=1),
        hovertemplate=f"Age: {age}<br>Mean VX rate: {mean_vx_rate:.6f}<br>Day: %{{x}}<br>Rate: %{{y:.6f}}<extra></extra>"
    ))

    # Plot UVX mortality rate
    fig.add_trace(go.Scatter(
        x=x_days,
        y=uvx_rate,
        mode='lines',
        name=f'UVX mortality rate age {age}',
        visible='legendonly',
        line=dict(color='orange', width=1),
        hovertemplate=f"Age: {age}<br>Mean UVX rate: {mean_uvx_rate:.6f}<br>Day: %{{x}}<br>Rate: %{{y:.6f}}<extra></extra>"
    ))

    # Plot total mortality rate
    fig.add_trace(go.Scatter(
        x=x_days,
        y=total_rate,
        mode='lines',
        name=f'Total mortality rate age {age}',
        visible='legendonly',
        line=dict(color='purple', width=1),
        hovertemplate=f"Age: {age}<br>Mean Total rate: {mean_total_rate:.6f}<br>Day: %{{x}}<br>Rate: %{{y:.6f}}<extra></extra>"
    ))

    # Plot baseline rate (same for all)
    fig.add_trace(go.Scatter(
        x=x_days,
        y=df_a['baseline_rate'],
        mode='lines',
        name=f'Baseline UVX mortality (global) age {age}',
        visible='legendonly',
        line=dict(color='green', width=1, dash='dot'),
        hovertemplate=f"Age: {age}<br>Global Baseline rate: {global_baseline_rate:.6f}<br>Day: %{{x}}<br>Rate: %{{y:.6f}}<extra></extra>"
    ))

    # Plot excess mortality %
    fig.add_trace(go.Scatter(
        x=x_days,
        y=df_a['rel_excess_mortality_%'],
        mode='lines',
        name=f'Excess mortality % VX age {age}',
        visible='legendonly',
        line=dict(color='red', width=2),
        hovertemplate=f"Age: {age}<br>Excess mortality %: %{{y:.2f}}<br>Day: %{{x}}<extra></extra>"
    ))

# Final layout
fig.update_layout(
    title=f'Vaccinated vs Unvaccinated Mortality Rates - Aggregation Window {AGG_WINDOW} days',
    xaxis_title='Day',
    yaxis_title='Mortality Rate or Excess Mortality %',
    legend_title='Legend',
    height=700,
    template='plotly_white'
)

# Save plot
pio.write_html(fig, OUTPUT_HTML, auto_open=False)
print(f"Plot saved to {OUTPUT_HTML}")
