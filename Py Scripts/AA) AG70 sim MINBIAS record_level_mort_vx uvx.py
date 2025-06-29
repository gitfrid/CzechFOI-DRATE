import pandas as pd
import numpy as np
import plotly.graph_objects as go

"""
This script loads a pre-aggregated bucket CSV file with mortality data by age group, vaccination status,
and calendar month. It calculates baseline mortality rates from the unvaccinated population, estimates 
expected deaths for both vaccinated and unvaccinated groups, and computes excess mortality as a percentage 
above the baseline.

Input bucket file structure:
- age: integer age of the individuals in the group (here fixed at 70)
- week_since_first_dose: integer, weeks elapsed since first vaccine dose (>=0 vaccinated, <0 unvaccinated aggregate)
- calendar_month: string representing calendar month in 'YYYY-MM' format
- deaths: count of deaths in the bucket for that age/week/month
- person_days: count of person-days observed in that bucket

The output is an interactive Plotly HTML line chart showing excess mortality (%) over weeks since vaccination
for vaccinated and unvaccinated groups, with unvaccinated aggregated at week -1.
"""

INPUT_CSV = r"C:\CzechFOI-BUCKET\TERRA\AG70_sim_MINBIAS_bucket_Vesely_106_202403141131.csv"
OUTPUT_HTML = r"C:\CzechFOI-BUCKET\Plot Results\AA) record level mort\AA) AG70 sim MINBIAS record_level_mort_vx uvx.html"

# Load aggregated bucket data from CSV
df = pd.read_csv(INPUT_CSV)

print("Columns in data:", df.columns.tolist())  # Verify loaded columns

# Convert calendar_month string (e.g. '2020-01') to a monthly period integer (int64) for grouping purposes
df['month'] = pd.to_datetime(df['calendar_month'], format='%Y-%m').dt.to_period('M').astype('int64')

# Define vaccination status based on weeks since first dose:
# week_since_first_dose >= 0 means vaccinated, otherwise unvaccinated aggregate
df['vax_stat'] = np.where(df['week_since_first_dose'] >= 0, 'Vaccinated (1st dose)', 'Unvaccinated (aggregate)')

# Separate vaccinated and unvaccinated data for baseline mortality calculation
df_vx = df[df['vax_stat'] == 'Vaccinated (1st dose)'].copy()
df_uvx = df[df['vax_stat'] == 'Unvaccinated (aggregate)'].copy()

# Calculate baseline mortality rates from unvaccinated group by month and age
baseline = df_uvx.groupby(['month', 'age'], observed=True).agg(
    deaths_sum = ('deaths', 'sum'),           # sum of deaths in bucket
    person_days_sum = ('person_days', 'sum')  # sum of person-days in bucket
).reset_index()

# Compute baseline mortality rate, guarding against division by zero
baseline['mort_rate_baseline'] = np.where(
    baseline['person_days_sum'] > 0,
    baseline['deaths_sum'] / baseline['person_days_sum'],
    0
)

# Keep only relevant columns for merging
baseline = baseline[['month', 'age', 'mort_rate_baseline']]

# Merge baseline mortality rate back into vaccinated and unvaccinated datasets by month and age
df_vx = df_vx.merge(baseline, on=['month', 'age'], how='left')
df_uvx = df_uvx.merge(baseline, on=['month', 'age'], how='left')

# Calculate expected deaths in each bucket based on baseline mortality rate and person-days observed
df_vx['expected_dead'] = df_vx['person_days'] * df_vx['mort_rate_baseline']
df_uvx['expected_dead'] = df_uvx['person_days'] * df_uvx['mort_rate_baseline']

# Aggregate vaccinated data by week since first dose for plotting
agg_vx = df_vx.groupby('week_since_first_dose').agg({
    'deaths': 'sum',
    'expected_dead': 'sum',
    'person_days': 'sum'
}).reset_index()
agg_vx['vax_stat'] = 'Vaccinated (1st dose)'

# Aggregate unvaccinated data similarly
agg_uvx = df_uvx.groupby('week_since_first_dose').agg({
    'deaths': 'sum',
    'expected_dead': 'sum',
    'person_days': 'sum'
}).reset_index()
agg_uvx['vax_stat'] = 'Unvaccinated (aggregate)'

# Combine vaccinated and unvaccinated aggregates for unified plotting
agg = pd.concat([agg_vx, agg_uvx], ignore_index=True)

# Calculate excess mortality percentage compared to baseline expected deaths
agg['excess_mortality_pct'] = np.where(
    agg['expected_dead'] > 0,
    (agg['deaths'] / agg['expected_dead'] - 1) * 100,
    np.nan  # undefined if no expected deaths
)

# Filter out weeks with very low person_days to avoid unstable mortality estimates
MIN_PERSON_WEEKS = 1e4
agg.loc[agg['person_days'] < MIN_PERSON_WEEKS, 'excess_mortality_pct'] = np.nan

# Define plot colors for vaccination status groups
colors = {'Vaccinated (1st dose)': 'blue', 'Unvaccinated (aggregate)': 'red'}

# Create Plotly figure for excess mortality over weeks since first dose
fig = go.Figure()

# Add line and marker traces for each vaccination group
for group in agg['vax_stat'].unique():
    df_sub = agg[agg['vax_stat'] == group]
    fig.add_trace(go.Scatter(
        x=df_sub['week_since_first_dose'],           # weeks since first dose
        y=df_sub['excess_mortality_pct'],             # excess mortality percentage
        mode='lines+markers',
        name=group,
        line=dict(color=colors.get(group, 'black'))
    ))

# Configure layout with titles, axis labels, and formatting
fig.update_layout(
    title="Excess Mortality by Weeks After Vaccination (Vaccinated vs Unvaccinated)",
    xaxis_title="Weeks Since First Dose (or aggregate unvaccinated at -1)",
    yaxis_title="Excess Mortality (%)",
    yaxis=dict(ticksuffix="%"),
    xaxis=dict(range=[-5, 105], dtick=10),
    legend_title="Group",
    template="simple_white"
)

# Add annotation pointing to the unvaccinated aggregate point at week -1 for clarity
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

# Save interactive plot to HTML file
fig.write_html(OUTPUT_HTML)
print(f"Plot saved to {OUTPUT_HTML}")
