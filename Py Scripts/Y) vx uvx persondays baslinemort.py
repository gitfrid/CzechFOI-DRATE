import pandas as pd
import numpy as np
import plotly.graph_objs as go
from collections import defaultdict

# === File Paths ===
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINIMALBIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINIMALBIAS_DEATHRISK_10X_Vesely_106_202403141131.csv"
OUTPUT_HTML = r"C:\CzechFOI-DRATE\Plot Results\Y) vx uvx persondays baslinemortality\Y vx uvx persondays baslinemortality.html"

# === Parameters ===
START_DATE = pd.Timestamp('2020-01-01')
MAX_AGE = 113
REFERENCE_YEAR = 2023
WINDOW_SIZE = 30  # rolling mortality window for visualization
BASELINE_TIME_RANGE = 7  # 7-day centered window for uvx baseline (±3 days)

# === Load Data ===
dose_date_cols = [f'Datum_{i}' for i in range(1, 8)]
needed_cols = ['Rok_narozeni', 'DatumUmrti'] + dose_date_cols

df = pd.read_csv(INPUT_CSV, usecols=needed_cols,
                 parse_dates=['DatumUmrti'] + dose_date_cols,
                 dayfirst=False, low_memory=False)
df.columns = [c.lower().strip() for c in df.columns]
dose_cols = [c.lower() for c in dose_date_cols]

# Calculate age and vaccination status
df['age'] = REFERENCE_YEAR - pd.to_numeric(df['rok_narozeni'], errors='coerce')
df = df[df['age'].between(0, MAX_AGE)].copy()
df['is_vaxed'] = df[dose_cols].notna().any(axis=1).astype(int)

# Convert key dates to days since START_DATE
def to_day(d): return (d - START_DATE).dt.days
df['death_day'] = to_day(df['datumumrti'])
for c in dose_cols:
    df[c + '_day'] = to_day(df[c])
df['first_dose_day'] = df[[c + '_day' for c in dose_cols]].min(axis=1, skipna=True)

# Generate time and age axes
END = int(df['death_day'].dropna().max())
days = np.arange(END + 1)
ages = np.arange(0, MAX_AGE + 1)

# === Build person-day and death counts per day/age ===
records = defaultdict(list)
for age in ages:
    sub = df[df['age'] == age]
    if sub.empty:
        continue
    death_arr = sub['death_day'].values
    is_vx = sub['is_vaxed'].values

    for d in days:
        alive = (np.isnan(death_arr) | (death_arr > d))
        vx_alive = np.sum(alive & (is_vx == 1))
        uvx_alive = np.sum(alive & (is_vx == 0))
        vx_deaths = np.sum((death_arr == d) & (is_vx == 1))
        uvx_deaths = np.sum((death_arr == d) & (is_vx == 0))

        records['day'].append(d)
        records['age'].append(age)
        records['vx_person'].append(vx_alive)
        records['uvx_person'].append(uvx_alive)
        records['vx_deaths'].append(vx_deaths)
        records['uvx_deaths'].append(uvx_deaths)

df_result = pd.DataFrame(records)

# === Rolling mortality rates ===
group = df_result.groupby('age')

df_result['vx_deaths_30d'] = group['vx_deaths'].transform(
    lambda x: x.rolling(WINDOW_SIZE, center=True, min_periods=1).sum())
df_result['vx_person_30d'] = group['vx_person'].transform(
    lambda x: x.rolling(WINDOW_SIZE, center=True, min_periods=1).sum())
df_result['vx_rate'] = df_result['vx_deaths_30d'] / df_result['vx_person_30d']

df_result['uvx_deaths_30d'] = group['uvx_deaths'].transform(
    lambda x: x.rolling(WINDOW_SIZE, center=True, min_periods=1).sum())
df_result['uvx_person_30d'] = group['uvx_person'].transform(
    lambda x: x.rolling(WINDOW_SIZE, center=True, min_periods=1).sum())
df_result['uvx_rate'] = df_result['uvx_deaths_30d'] / df_result['uvx_person_30d']

df_result['total_deaths_30d'] = df_result['vx_deaths_30d'] + df_result['uvx_deaths_30d']
df_result['total_person_30d'] = df_result['vx_person_30d'] + df_result['uvx_person_30d']
df_result['total_rate'] = df_result['total_deaths_30d'] / df_result['total_person_30d']

# === Fine-grained baseline using local 7-day range of UVX mortality ===
baseline_window = BASELINE_TIME_RANGE
df_result['baseline_uvx_deaths'] = group['uvx_deaths'].transform(
    lambda x: x.rolling(baseline_window, center=True, min_periods=1).sum())
df_result['baseline_uvx_person'] = group['uvx_person'].transform(
    lambda x: x.rolling(baseline_window, center=True, min_periods=1).sum())
df_result['baseline_rate'] = df_result['baseline_uvx_deaths'] / df_result['baseline_uvx_person']

# === Excess mortality ===
df_result['excess_mortality'] = df_result['vx_rate'] - df_result['baseline_rate']

# === Plotting ===
fig = go.Figure()
for age in ages:
    df_a = df_result[df_result['age'] == age]
    if df_a.empty:
        continue

    fig.add_trace(go.Scatter(
        x=df_a['day'], y=df_a['vx_rate'],
        mode='lines', name=f'vx rate age {age}', visible='legendonly',
        line=dict(color='blue', width=1)))
    
    fig.add_trace(go.Scatter(
        x=df_a['day'], y=df_a['uvx_rate'],
        mode='lines', name=f'uvx rate age {age}', visible='legendonly',
        line=dict(color='red', width=1)))
    
    fig.add_trace(go.Scatter(
        x=df_a['day'], y=df_a['total_rate'],
        mode='lines', name=f'total rate age {age}', visible='legendonly',
        line=dict(color='orange', width=1)))
    
    fig.add_trace(go.Scatter(
        x=df_a['day'], y=df_a['baseline_rate'],
        mode='lines', name=f'baseline 7d age {age}', visible='legendonly',
        line=dict(color='green', width=1, dash='dot')))
    
    fig.add_trace(go.Scatter(
        x=df_a['day'], y=df_a['excess_mortality'],
        mode='lines', name=f'excess age {age}', visible='legendonly',
        line=dict(color='black', width=1)))

fig.update_layout(
    title='VX vs UVX Mortality with 7-day Baseline per Age',
    xaxis_title='Days since 2020‑01‑01',
    yaxis_title='Mortality rate (deaths per person-day)',
    height=800,
    legend=dict(y=1, x=1, orientation='v')
)

fig.write_html(OUTPUT_HTML)
print(f"Saved to: {OUTPUT_HTML}")
