import pandas as pd
import numpy as np
import plotly.graph_objects as go

INPUT_CSV = r"C:\CzechFOI-BUCKET\TERRA\AG70_sim_MINBIAS_bucket_Vesely_106_202403141131.csv"
OUTPUT_HTML = r"C:\CzechFOI-BUCKET\Plot Results\AA) record level mort\AA) AG70 sim MINBIAS record_level_mort_vx uvx.html"

# Load data - CSV already aggregated by age, week_since_first_dose, calendar_month
df = pd.read_csv(INPUT_CSV)

print("Columns in data:", df.columns.tolist())

# Convert calendar_month (e.g. '2020-01') to monthly period as int64 for grouping
df['month'] = pd.to_datetime(df['calendar_month'], format='%Y-%m').dt.to_period('M').astype('int64')

# Define vaccination status based on week_since_first_dose
# week_since_first_dose >=0 => vaccinated; else unvaccinated
df['vax_stat'] = np.where(df['week_since_first_dose'] >= 0, 'Vaccinated (1st dose)', 'Unvaccinated (aggregate)')

# Split vaccinated and unvaccinated for baseline calculation
df_vx = df[df['vax_stat'] == 'Vaccinated (1st dose)'].copy()
df_uvx = df[df['vax_stat'] == 'Unvaccinated (aggregate)'].copy()

# Calculate baseline mortality rate from unvaccinated group by month and age
baseline = df_uvx.groupby(['month', 'age'], observed=True).agg(
    deaths_sum = ('deaths', 'sum'),
    person_days_sum = ('person_days', 'sum')
).reset_index()

baseline['mort_rate_baseline'] = np.where(
    baseline['person_days_sum'] > 0,
    baseline['deaths_sum'] / baseline['person_days_sum'],
    0
)

baseline = baseline[['month', 'age', 'mort_rate_baseline']]

# Merge baseline mortality back into vaccinated and unvaccinated datasets
df_vx = df_vx.merge(baseline, on=['month', 'age'], how='left')
df_uvx = df_uvx.merge(baseline, on=['month', 'age'], how='left')

# Calculate expected deaths based on baseline mortality
df_vx['expected_dead'] = df_vx['person_days'] * df_vx['mort_rate_baseline']
df_uvx['expected_dead'] = df_uvx['person_days'] * df_uvx['mort_rate_baseline']

# Aggregate by week_since_first_dose to prepare for plotting
agg_vx = df_vx.groupby('week_since_first_dose').agg({
    'deaths': 'sum',
    'expected_dead': 'sum',
    'person_days': 'sum'
}).reset_index()
agg_vx['vax_stat'] = 'Vaccinated (1st dose)'

agg_uvx = df_uvx.groupby('week_since_first_dose').agg({
    'deaths': 'sum',
    'expected_dead': 'sum',
    'person_days': 'sum'
}).reset_index()
agg_uvx['vax_stat'] = 'Unvaccinated (aggregate)'

# Combine vaccinated and unvaccinated aggregates
agg = pd.concat([agg_vx, agg_uvx], ignore_index=True)

# Calculate excess mortality percentage
agg['excess_mortality_pct'] = np.where(
    agg['expected_dead'] > 0,
    (agg['deaths'] / agg['expected_dead'] - 1) * 100,
    np.nan
)

# Filter out weeks with very low person_days for stable estimates
MIN_PERSON_WEEKS = 1e4
agg.loc[agg['person_days'] < MIN_PERSON_WEEKS, 'excess_mortality_pct'] = np.nan

# Colors for plot groups
colors = {'Vaccinated (1st dose)': 'blue', 'Unvaccinated (aggregate)': 'red'}

# Create Plotly figure
fig = go.Figure()

for group in agg['vax_stat'].unique():
    df_sub = agg[agg['vax_stat'] == group]
    fig.add_trace(go.Scatter(
        x=df_sub['week_since_first_dose'],
        y=df_sub['excess_mortality_pct'],
        mode='lines+markers',
        name=group,
        line=dict(color=colors.get(group, 'black'))
    ))

# Layout and axis update to highlight unvaccinated at week -1
fig.update_layout(
    title="Excess Mortality by Weeks After Vaccination (Vaccinated vs Unvaccinated)",
    xaxis_title="Weeks Since First Dose (or aggregate unvaccinated at -1)",
    yaxis_title="Excess Mortality (%)",
    yaxis=dict(ticksuffix="%"),
    xaxis=dict(range=[-5, 105], dtick=10),
    legend_title="Group",
    template="simple_white"
)

# Add annotation for unvaccinated aggregate point (week -1)
unvax_point = agg[agg['vax_stat'] == 'Unvaccinated (aggregate)']
if not unvax_point.empty:
    fig.add_annotation(
        x=-1,
        y=unvax_point['excess_mortality_pct'].values[0],
        text="Unvaccinated aggregate point",
        showarrow=True,
        arrowhead=1,
        ax=-40,
        ay=-40
    )

# Save plot as HTML
fig.write_html(OUTPUT_HTML)
print(f"Plot saved to {OUTPUT_HTML}")
