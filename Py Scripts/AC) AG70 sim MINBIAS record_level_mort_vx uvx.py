import pandas as pd
import numpy as np
import plotly.graph_objects as go

"""
This script loads a pre-aggregated bucket CSV file with mortality data by age group, vaccination status,
and calendar month. It calculates Combined Baseline mortality rates from the vaccinated and unvaccinated population, estimates 
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

# Path to input CSV with pre-aggregated mortality data buckets
INPUT_CSV = r"C:\CzechFOI-BUCKET\TERRA\AG70_sim_MINBIAS_bucket_Vesely_106_202403141131.csv"

# Path to output HTML file with interactive plot
OUTPUT_HTML = r"C:\CzechFOI-BUCKET\Plot Results\AC) record level mort\AC) AG70 sim minbias record_level_mort_combined_baseline.html"

# Load aggregated bucket data from CSV into DataFrame
df = pd.read_csv(INPUT_CSV)

print("Columns in data:", df.columns.tolist())  # Show available columns to verify data loaded correctly

# Convert calendar_month string to a numeric month period (integer) for grouping and merging
df['month'] = pd.to_datetime(df['calendar_month'], format='%Y-%m').dt.to_period('M').astype('int64')

# Create vaccination status label:
# weeks_since_first_dose >= 0 means vaccinated; < 0 means unvaccinated aggregate group
df['vax_stat'] = np.where(df['week_since_first_dose'] >= 0, 'Vaccinated (1st dose)', 'Unvaccinated (aggregate)')

# Split data into vaccinated and unvaccinated subsets
df_vx = df[df['vax_stat'] == 'Vaccinated (1st dose)'].copy()
df_uvx = df[df['vax_stat'] == 'Unvaccinated (aggregate)'].copy()

# Combine vaccinated and unvaccinated data for baseline mortality rate calculation
df_combined = pd.concat([df_vx, df_uvx], ignore_index=True)

# Calculate baseline mortality rate by month and age using combined population deaths and person-days
baseline = df_combined.groupby(['month', 'age'], observed=True).agg(
    deaths_sum=('deaths', 'sum'),
    person_days_sum=('person_days', 'sum')
).reset_index()

# Calculate baseline mortality rate = deaths / person-days, handling zero person-days case
baseline['mort_rate_baseline'] = np.where(
    baseline['person_days_sum'] > 0,
    baseline['deaths_sum'] / baseline['person_days_sum'],
    0
)

# Keep only relevant columns in baseline dataframe
baseline = baseline[['month', 'age', 'mort_rate_baseline']]

# Merge baseline mortality rate back into vaccinated and unvaccinated data frames by month and age
df_vx = df_vx.merge(baseline, on=['month', 'age'], how='left')
df_uvx = df_uvx.merge(baseline, on=['month', 'age'], how='left')

# Calculate expected deaths in each bucket using baseline mortality rate * person-days
df_vx['expected_dead'] = df_vx['person_days'] * df_vx['mort_rate_baseline']
df_uvx['expected_dead'] = df_uvx['person_days'] * df_uvx['mort_rate_baseline']

# Aggregate deaths, expected deaths, and person-days by weeks since first dose for vaccinated group
agg_vx = df_vx.groupby('week_since_first_dose').agg({
    'deaths': 'sum',
    'expected_dead': 'sum',
    'person_days': 'sum'
}).reset_index()
agg_vx['vax_stat'] = 'Vaccinated (1st dose)'  # Label for plot legend

# Aggregate deaths, expected deaths, and person-days by weeks since first dose for unvaccinated group
agg_uvx = df_uvx.groupby('week_since_first_dose').agg({
    'deaths': 'sum',
    'expected_dead': 'sum',
    'person_days': 'sum'
}).reset_index()
agg_uvx['vax_stat'] = 'Unvaccinated (aggregate)'  # Label for plot legend

# Combine aggregated vaccinated and unvaccinated data into one dataframe
agg = pd.concat([agg_vx, agg_uvx], ignore_index=True)

# Calculate excess mortality percentage = ((observed deaths / expected deaths) - 1) * 100
agg['excess_mortality_pct'] = np.where(
    agg['expected_dead'] > 0,
    (agg['deaths'] / agg['expected_dead'] - 1) * 100,
    np.nan  # Use NaN if expected deaths is zero to avoid division by zero
)

# Define minimum person-days threshold for reliable estimates (mask low data weeks)
MIN_PERSON_WEEKS = 10000
agg.loc[agg['person_days'] < MIN_PERSON_WEEKS, 'excess_mortality_pct'] = np.nan

# Define colors for vaccinated and unvaccinated traces
colors = {
    'Vaccinated (1st dose)': 'blue',
    'Unvaccinated (aggregate)': 'red'
}

# Create Plotly figure object
fig = go.Figure()

# Add traces for each vaccination status group
for group in agg['vax_stat'].unique():
    df_sub = agg[agg['vax_stat'] == group]
    fig.add_trace(go.Scatter(
        x=df_sub['week_since_first_dose'],
        y=df_sub['excess_mortality_pct'],
        mode='lines+markers',
        name=group,
        line=dict(color=colors.get(group, 'black'))
    ))

# Update layout with titles, axis labels, formatting, and template
fig.update_layout(
    title="Excess Mortality by Weeks After Vaccination (Using Combined Baseline)",
    xaxis_title="Weeks Since First Dose (or aggregate unvaccinated at -1)",
    yaxis_title="Excess Mortality (%)",
    yaxis=dict(ticksuffix="%"),  # Append % to y-axis ticks
    xaxis=dict(range=[-5, 105], dtick=10),  # Show x-axis from -5 to 105 weeks with step 10
    legend_title="Group",
    template="simple_white"
)

# Add annotation for unvaccinated aggregate point at week -1 if available
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
