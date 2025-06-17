import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dowhy import CausalModel

# --- Parameters ---
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_SELCTION_BIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
OUTPUT_HTML = r"C:\CzechFOI-DRATE\Plot Results\S) diff death dose agebin\S) sim selection bias vx uvx raw diff population doses causal estimate.html"
START_DATE = pd.Timestamp('2020-01-01')
MAX_AGE = 113
REFERENCE_YEAR = 2023

# --- Load and preprocess data ---
dose_date_cols = [f'Datum_{i}' for i in range(1, 8)]
usecols = ['Rok_narozeni', 'DatumUmrti'] + dose_date_cols
df = pd.read_csv(INPUT_CSV, usecols=usecols, parse_dates=['DatumUmrti'] + dose_date_cols)
df.columns = [c.strip().lower() for c in df.columns]
dose_date_cols_lower = [c.lower() for c in dose_date_cols]
df['birth_year'] = pd.to_numeric(df['rok_narozeni'], errors='coerce')
df['age'] = REFERENCE_YEAR - df['birth_year']
df = df[df['age'].between(0, MAX_AGE)].copy()

def to_day_num(series):
    return (series - START_DATE).dt.days

df['death_day'] = to_day_num(df['datumumrti'])
for col in dose_date_cols_lower:
    df[col + '_day'] = to_day_num(df[col])
df['first_dose_day'] = df[[col + '_day' for col in dose_date_cols_lower]].min(axis=1, skipna=True)

END_MEASURE = int(df['death_day'].dropna().max())
days = np.arange(0, END_MEASURE + 1)
ages = np.arange(0, MAX_AGE + 1)

# --- Aggregate daily results per age ---
result_rows = []
for age in ages:
    df_age = df[df['age'] == age]
    if df_age.empty:
        for day in days:
            result_rows.append({'age': age, 'day': day, 'pop_vx': 0, 'pop_uvx': 0, 'death_vx': 0, 'death_uvx': 0,
                                'first_dose_count': 0, 'all_dose_count': 0})
        continue

    death_days = df_age['death_day'].values
    first_dose_days = df_age['first_dose_day'].values
    dose_days_all = pd.concat([df_age[col + '_day'] for col in dose_date_cols_lower])

    for day in days:
        alive_mask = np.isnan(death_days) | (death_days > day)
        death_today_mask = (death_days == day)
        is_vx = (day >= first_dose_days)
        is_uvx = ~is_vx

        result_rows.append({
            'age': age,
            'day': day,
            'pop_vx': np.sum(alive_mask & is_vx),
            'pop_uvx': np.sum(alive_mask & is_uvx),
            'death_vx': np.sum(death_today_mask & is_vx),
            'death_uvx': np.sum(death_today_mask & is_uvx),
            'first_dose_count': np.sum(first_dose_days == day),
            'all_dose_count': np.sum(dose_days_all == day),
        })

result_df = pd.DataFrame(result_rows)
result_df['uvx_minus_vx_death'] = result_df['death_uvx'] - result_df['death_vx']

# --- DoWhy Causal Estimates per Age ---
causal_estimates = {}
ci_lowers = {}
ci_uppers = {}

for age in ages:
    age_df = result_df[result_df['age'] == age].copy()
    if age_df['all_dose_count'].sum() == 0:
        causal_estimates[age] = 0
        ci_lowers[age] = 0
        ci_uppers[age] = 0
        continue

    model = CausalModel(
        data=age_df,
        treatment='all_dose_count',
        outcome='uvx_minus_vx_death',
        common_causes=[],
    )
    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=True
    )

    causal_estimates[age] = estimate.value
    cis = estimate.get_confidence_intervals()
    if cis is not None and cis.size > 0 and len(cis[0]) == 2:
        ci_lowers[age] = cis[0][0]
        ci_uppers[age] = cis[0][1]
    else:
        ci_lowers[age] = np.nan
        ci_uppers[age] = np.nan

# --- Plot 1: Raw Differences ---
fig1 = make_subplots(specs=[[{"secondary_y": True}]])
for age in ages:
    df_age = result_df[result_df['age'] == age]
    if df_age.empty:
        continue

    fig1.add_trace(go.Scatter(
        x=df_age['day'],
        y=df_age['death_uvx'] - df_age['death_vx'],  # <- updated line
        mode='lines',
        name=f'Death diff uvx-vx age {age}',         # <- updated label
        visible='legendonly'
    ), secondary_y=False)

    fig1.add_trace(go.Scatter(
        x=df_age['day'],
        y=df_age['first_dose_count'],
        mode='lines',
        name=f'First Dose Count age {age}',
        visible='legendonly',
        line=dict(dash='dot')
    ), secondary_y=True)

    fig1.add_trace(go.Scatter(
        x=df_age['day'],
        y=df_age['all_dose_count'],
        mode='lines',
        name=f'All Dose Count age {age}',
        visible='legendonly',
        line=dict(dash='dash')
    ), secondary_y=True)

fig1.update_layout(
    title="Raw Difference (uvx - vx) Deaths and Dose Counts Over Time by Age",
    xaxis_title="Days since 2020-01-01",
    yaxis_title="Death Difference (uvx - vx)",
    template="plotly_white",
    height=700
)
fig1.update_yaxes(title_text="Dose Counts", secondary_y=True)

# --- Plot 2: DoWhy Causal Estimates with Confidence Intervals (Error Bars) ---
ages_list = list(causal_estimates.keys())
estimates_list = [causal_estimates[a] for a in ages_list]
ci_lower_list = [causal_estimates[a] - ci_lowers[a] if not np.isnan(ci_lowers[a]) else 0 for a in ages_list]
ci_upper_list = [ci_uppers[a] - causal_estimates[a] if not np.isnan(ci_uppers[a]) else 0 for a in ages_list]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=ages_list,
    y=estimates_list,
    mode='markers',
    marker=dict(size=6, color='darkblue'),
    name="Causal Estimate",
    error_y=dict(
        type='data',
        symmetric=False,
        array=ci_upper_list,
        arrayminus=ci_lower_list,
        thickness=1.5,
        width=6,
        color='lightblue'
    )
))
fig2.update_layout(
    title="DoWhy Causal Estimates by Age (Death Diff / All Doses)",
    xaxis_title="Age",
    yaxis_title="Causal Estimate with 95% CI",
    template="plotly_white",
    height=500
)

# --- Combine into Single Plot ---
fig_combined = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        "Raw Difference (uvx - vx) Deaths and Dose Counts Over Time",
        "Causal Estimates by Age"
    ),
    specs=[[{"secondary_y": True}], [{}]],
    vertical_spacing=0.15
)

for trace in fig1.data:
    fig_combined.add_trace(trace, row=1, col=1, secondary_y="Dose Count" in trace.name)

for trace in fig2.data:
    fig_combined.add_trace(trace, row=2, col=1)

fig_combined.update_layout(
    height=1100,
    showlegend=True,
    template="plotly_white"
)
fig_combined.update_yaxes(title_text="Death Difference (uvx - vx)", row=1, col=1, secondary_y=False)
fig_combined.update_yaxes(title_text="Dose Counts", row=1, col=1, secondary_y=True)

# --- Save ---
fig_combined.write_html(OUTPUT_HTML)
print(f"Saved combined plot to {OUTPUT_HTML}")
