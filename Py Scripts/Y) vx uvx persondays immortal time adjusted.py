import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from collections import defaultdict

# === File Paths ===
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_Vesely_106_SIMULATED.csv"
OUTPUT_HTML = r"C:\CzechFOI-DRATE\Plot Results\Y) vx uvx persondays baslinemortality\Y) vx uvx personday_exess_mortality_adjusted.html"


# ============================================
# 0. Setup & Parameters
# ============================================
START_DATE = pd.Timestamp('2020-01-01')
REFERENCE_YEAR = 2023
MAX_AGE = 113
AGG_WINDOW = 7  # <- Only one window
IMMUNITY_LAG_DAYS = 0
MIN_PERSON_DAYS = 100

# ============================================
# 1. Load Data
# ============================================
dose_date_cols = [f'Datum_{i}' for i in range(1, 8)]
df = pd.read_csv(
    INPUT_CSV,
    usecols=['Rok_narozeni', 'DatumUmrti'] + dose_date_cols,
    parse_dates=['DatumUmrti'] + dose_date_cols,
    dayfirst=False,
    low_memory=False
)

df.columns = [col.lower().strip() for col in df.columns]
dose_cols = [col.lower() for col in dose_date_cols]

df['age'] = REFERENCE_YEAR - pd.to_numeric(df['rok_narozeni'], errors='coerce')
df = df[df['age'].between(0, MAX_AGE)].copy()

to_day = lambda d: (d - START_DATE).dt.days
df['death_day'] = to_day(df['datumumrti'])
for col in dose_cols:
    df[f"{col}_day"] = to_day(df[col])
df['first_dose_day'] = df[[f"{col}_day" for col in dose_cols]].min(axis=1, skipna=True)

# ============================================
# 2. Define time and age ranges
# ============================================
END_DAY = int(df['death_day'].dropna().max())
days = np.arange(0, END_DAY + 1)
ages = np.arange(0, MAX_AGE + 1)

# ============================================
# 3. Daily Aggregation
# ============================================
records = defaultdict(list)

for age in ages:
    sub = df[df['age'] == age]
    if sub.empty:
        continue

    death_arr = sub['death_day'].values
    first_dose_days = sub['first_dose_day'].values

    for d in days:
        alive = np.isnan(death_arr) | (death_arr > d)
        is_vx = (~np.isnan(first_dose_days)) & (d >= first_dose_days + IMMUNITY_LAG_DAYS)

        vx_alive = np.sum(alive & is_vx)
        uvx_alive = np.sum(alive & (~is_vx))
        vx_deaths = np.sum((death_arr == d) & is_vx)
        uvx_deaths = np.sum((death_arr == d) & (~is_vx))

        records['day'].append(d)
        records['age'].append(age)
        records['vx_person'].append(vx_alive)
        records['uvx_person'].append(uvx_alive)
        records['vx_deaths'].append(vx_deaths)
        records['uvx_deaths'].append(uvx_deaths)

df_daily = pd.DataFrame(records)

# ============================================
# 4. Aggregation and Calculations
# ============================================
pio.renderers.default = "browser"

df_daily['period'] = (df_daily['day'] // AGG_WINDOW).astype(int)
agg_cols = ['vx_person', 'uvx_person', 'vx_deaths', 'uvx_deaths']
df_agg = df_daily.groupby(['period', 'age'])[agg_cols].sum().reset_index()

df_agg['total_person'] = df_agg['vx_person'] + df_agg['uvx_person']
df_agg['total_deaths'] = df_agg['vx_deaths'] + df_agg['uvx_deaths']

# Filter unstable periods
mask_valid = (df_agg['vx_person'] >= MIN_PERSON_DAYS) & (df_agg['uvx_person'] >= MIN_PERSON_DAYS)
df_agg.loc[~mask_valid, ['vx_person', 'uvx_person', 'vx_deaths', 'uvx_deaths']] = np.nan
df_agg['total_person'] = df_agg['vx_person'] + df_agg['uvx_person']
df_agg['total_deaths'] = df_agg['vx_deaths'] + df_agg['uvx_deaths']

# Global baseline rate from UVX
valid_uvx = df_agg.dropna(subset=['uvx_person', 'uvx_deaths'])
global_uvx_deaths = valid_uvx['uvx_deaths'].sum()
global_uvx_person = valid_uvx['uvx_person'].sum()
global_baseline_rate = global_uvx_deaths / global_uvx_person if global_uvx_person > 0 else np.nan
df_agg['baseline_rate'] = global_baseline_rate

# Expected deaths and excess mortality
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

    vx_rate = df_a['vx_deaths'] / df_a['vx_person']
    uvx_rate = df_a['uvx_deaths'] / df_a['uvx_person']
    total_rate = df_a['total_deaths'] / df_a['total_person']

    mean_vx_rate = (df_a['vx_deaths'].sum() / df_a['vx_person'].sum()) if df_a['vx_person'].sum() > 0 else np.nan
    mean_uvx_rate = (df_a['uvx_deaths'].sum() / df_a['uvx_person'].sum()) if df_a['uvx_person'].sum() > 0 else np.nan
    mean_total_rate = (df_a['total_deaths'].sum() / df_a['total_person'].sum()) if df_a['total_person'].sum() > 0 else np.nan

    fig.add_trace(go.Scatter(
        x=x_days,
        y=vx_rate,
        mode='lines',
        name=f'VX mortality rate age {age}',
        visible='legendonly',
        line=dict(color='blue', width=1),
        hovertemplate=f"Age: {age}<br>Mean VX rate: {mean_vx_rate:.6f}<br>Day: %{{x}}<br>Rate: %{{y:.6f}}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=x_days,
        y=uvx_rate,
        mode='lines',
        name=f'UVX mortality rate age {age}',
        visible='legendonly',
        line=dict(color='orange', width=1),
        hovertemplate=f"Age: {age}<br>Mean UVX rate: {mean_uvx_rate:.6f}<br>Day: %{{x}}<br>Rate: %{{y:.6f}}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=x_days,
        y=total_rate,
        mode='lines',
        name=f'Total mortality rate age {age}',
        visible='legendonly',
        line=dict(color='purple', width=1),
        hovertemplate=f"Age: {age}<br>Mean Total rate: {mean_total_rate:.6f}<br>Day: %{{x}}<br>Rate: %{{y:.6f}}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=x_days,
        y=df_a['baseline_rate'],
        mode='lines',
        name=f'Baseline UVX mortality (global) age {age}',
        visible='legendonly',
        line=dict(color='green', width=1, dash='dot'),
        hovertemplate=f"Age: {age}<br>Global Baseline rate: {global_baseline_rate:.6f}<br>Day: %{{x}}<br>Rate: %{{y:.6f}}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=x_days,
        y=df_a['rel_excess_mortality_%'],
        mode='lines',
        name=f'Excess mortality % VX age {age}',
        visible='legendonly',
        line=dict(color='red', width=2),
        hovertemplate=f"Age: {age}<br>Excess mortality %: %{{y:.2f}}<br>Day: %{{x}}<extra></extra>"
    ))

fig.update_layout(
    title=f'Vaccinated vs Unvaccinated Mortality Rates - Aggregation Window {AGG_WINDOW} days',
    xaxis_title='Day',
    yaxis_title='Mortality Rate or Excess Mortality %',
    legend_title='Legend',
    height=700,
    template='plotly_white'
)

pio.write_html(fig, OUTPUT_HTML, auto_open=False)
print(f"Plot saved to {OUTPUT_HTML}")
