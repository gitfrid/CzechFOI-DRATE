import pandas as pd
import numpy as np
import plotly.graph_objs as go
from collections import defaultdict

# === File Paths ===
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINIMALBIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINIMALBIAS_DEATHRISK_10X_Vesely_106_202403141131.csv"
OUTPUT_HTML = r"C:\CzechFOI-DRATE\Plot Results\Y) vx uvx persondays\Y) vx uvx persondays mortality.html"

# === Parameters ===
START_DATE = pd.Timestamp('2020-01-01')
MAX_AGE = 113
REFERENCE_YEAR = 2023
WINDOW_SIZE = 30  # rolling window for mortality rate

# === Load Data ===
dose_date_cols = [f'Datum_{i}' for i in range(1, 8)]
needed_cols = ['Rok_narozeni', 'DatumUmrti'] + dose_date_cols

df = pd.read_csv(
    INPUT_CSV,
    usecols=needed_cols,
    parse_dates=['DatumUmrti'] + dose_date_cols,
    dayfirst=False,
    low_memory=False
)

df.columns = [col.strip().lower() for col in df.columns]
dose_date_cols_lower = [col.lower() for col in dose_date_cols]

df['birth_year'] = pd.to_numeric(df['rok_narozeni'], errors='coerce')
df['age'] = REFERENCE_YEAR - df['birth_year']
df = df[df['age'].between(0, MAX_AGE)].copy()
df['is_vaxed'] = df[dose_date_cols_lower].notna().any(axis=1).astype(int)

def to_day_number(date_series):
    return (date_series - START_DATE).dt.days

df['death_day'] = to_day_number(df['datumumrti'])
for col in dose_date_cols_lower:
    df[col + '_day'] = to_day_number(df[col])
df['first_dose_day'] = df[[col + '_day' for col in dose_date_cols_lower]].min(axis=1, skipna=True)

END_MEASURE = int(df['death_day'].dropna().max())
days = np.arange(0, END_MEASURE + 1)
ages = np.arange(0, MAX_AGE + 1)

# === Compute Person-Days and Deaths ===
print("Computing person-days and deaths...")
records = defaultdict(list)

for age in ages:
    sub = df[df['age'] == age]
    if sub.empty:
        continue

    death_days = sub['death_day'].values
    is_vaxed = sub['is_vaxed'].values

    for day in days:
        alive_mask = np.isnan(death_days) | (death_days > day)

        vx_mask = alive_mask & (is_vaxed == 1)
        uvx_mask = alive_mask & (is_vaxed == 0)

        deaths_today = sub['death_day'] == day
        vx_deaths = np.sum(deaths_today & (is_vaxed == 1))
        uvx_deaths = np.sum(deaths_today & (is_vaxed == 0))

        records['day'].append(day)
        records['age'].append(age)
        records['vx_person'].append(np.sum(vx_mask))
        records['uvx_person'].append(np.sum(uvx_mask))
        records['vx_deaths'].append(vx_deaths)
        records['uvx_deaths'].append(uvx_deaths)

df_result = pd.DataFrame(records)

# === Compute Rolling 30-day Mortality Rates per Age ===
def compute_rolling_mortality(df_result):
    df_result['vx_deaths_30d'] = df_result.groupby('age')['vx_deaths'].transform(
        lambda x: x.rolling(WINDOW_SIZE, center=True, min_periods=1).sum()
    )
    df_result['uvx_deaths_30d'] = df_result.groupby('age')['uvx_deaths'].transform(
        lambda x: x.rolling(WINDOW_SIZE, center=True, min_periods=1).sum()
    )
    df_result['vx_person_30d'] = df_result.groupby('age')['vx_person'].transform(
        lambda x: x.rolling(WINDOW_SIZE, center=True, min_periods=1).sum()
    )
    df_result['uvx_person_30d'] = df_result.groupby('age')['uvx_person'].transform(
        lambda x: x.rolling(WINDOW_SIZE, center=True, min_periods=1).sum()
    )
    df_result['vx_rate'] = df_result['vx_deaths_30d'] / df_result['vx_person_30d']
    df_result['uvx_rate'] = df_result['uvx_deaths_30d'] / df_result['uvx_person_30d']
    df_result['excess_mortality'] = df_result['vx_rate'] - df_result['uvx_rate']
    return df_result

df_result = compute_rolling_mortality(df_result)

# === Add Total Aggregates (across all ages) ===
df_total = df_result.groupby('day').agg({
    'vx_deaths': 'sum',
    'uvx_deaths': 'sum',
    'vx_person': 'sum',
    'uvx_person': 'sum'
}).reset_index()

df_total['vx_deaths_30d'] = df_total['vx_deaths'].rolling(WINDOW_SIZE, center=True, min_periods=1).sum()
df_total['uvx_deaths_30d'] = df_total['uvx_deaths'].rolling(WINDOW_SIZE, center=True, min_periods=1).sum()
df_total['vx_person_30d'] = df_total['vx_person'].rolling(WINDOW_SIZE, center=True, min_periods=1).sum()
df_total['uvx_person_30d'] = df_total['uvx_person'].rolling(WINDOW_SIZE, center=True, min_periods=1).sum()
df_total['vx_rate'] = df_total['vx_deaths_30d'] / df_total['vx_person_30d']
df_total['uvx_rate'] = df_total['uvx_deaths_30d'] / df_total['uvx_person_30d']
df_total['excess_mortality'] = df_total['vx_rate'] - df_total['uvx_rate']

# === Plotting ===
print("Generating Plotly figure...")

fig = go.Figure()

# === Per-Age Mortality Rate Traces (vx, uvx) ===
for age in ages:
    df_age = df_result[df_result['age'] == age]
    if df_age.empty:
        continue
    # vx mortality
    fig.add_trace(go.Scatter(
        x=df_age['day'], y=df_age['vx_rate'],
        mode='lines', name=f'vx mortality age {age}',
        line=dict(color='blue', width=1), visible='legendonly'
    ))
    # uvx mortality
    fig.add_trace(go.Scatter(
        x=df_age['day'], y=df_age['uvx_rate'],
        mode='lines', name=f'uvx mortality age {age}',
        line=dict(color='red', width=1), visible='legendonly'
    ))
    # excess mortality
    fig.add_trace(go.Scatter(
        x=df_age['day'], y=df_age['excess_mortality'],
        mode='lines', name=f'excess mortality age {age}',
        line=dict(color='black', width=1), visible='legendonly'
    ))

# === Add Total Mortality Rate Plot ===
fig.add_trace(go.Scatter(
    x=df_total['day'], y=df_total['vx_rate'],
    mode='lines', name='Total vx mortality rate',
    line=dict(color='blue', width=4)
))
fig.add_trace(go.Scatter(
    x=df_total['day'], y=df_total['uvx_rate'],
    mode='lines', name='Total uvx mortality rate',
    line=dict(color='red', width=4)
))
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
