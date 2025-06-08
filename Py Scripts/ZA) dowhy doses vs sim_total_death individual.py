# This script analyzes the causal effect of vaccine dose count on death risk 
# per age bin using the DoWhy library. It loads individual-level Czech vaccination 
# and mortality data, computes the estimated Average Treatment Effect (ATE) for 
# each age bin (0–114), and visualizes the ATE, observed death rates, and 
# mean dose counts in a Plotly interactive HTML file.

# Output:
# - An HTML plot showing:
#   - Estimated ATE per age bin (ΔDeath Risk per Dose)
#   - Observed death rate
#   - Mean number of vaccine doses

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dowhy import CausalModel

# -------------------
# Config
# -------------------
REFERENCE_DATE = pd.Timestamp("2020-01-01")  # Reference day 0
AGE_BIN_WIDTH = 1  # Age binning resolution
# Input data - homogen population with simulated constant random death rate, test the code for bias  
# INPUT_CSV = r"C:\github\CzechFOI-DRATE\TERRA\sim_SELCTION_BIAS_Vesely_106_202403141131.csv"
# Input data - real death rate 
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"  
OUTPUT_HTML = r"C:\github\CzechFOI-DRATE\Plot Results\ZA) dowhy doses vs total_death individual\ZA) doses_vs_total_deaths_dowhy_individual.html"
CURRENT_YEAR = 2023  # Used to calculate age

# -------------------
# Load data
# -------------------
# Define date columns for up to 7 vaccine doses
dose_date_cols = [f'Datum_{i}' for i in range(1, 8)]
date_cols = ['DatumUmrti'] + dose_date_cols

# Read relevant columns and parse dates
df = pd.read_csv(
    INPUT_CSV,
    usecols=['Rok_narozeni'] + date_cols,
    parse_dates=date_cols,
    dayfirst=False,
    low_memory=False
)

# -------------------
# Preprocessing
# -------------------
# Compute the first dose and map to day offset
first_dose = df[dose_date_cols].min(axis=1)
df['first_dose_day'] = (first_dose - REFERENCE_DATE).dt.days

# Compute death day as offset
df['death_day'] = (df['DatumUmrti'] - REFERENCE_DATE).dt.days

# Binary vaccinated flag
df['vaccinated'] = ~first_dose.isna()

# Binary death flag
df['death'] = (~df['DatumUmrti'].isna()).astype(int)

# Calculate age from birth year
df['age'] = CURRENT_YEAR - df['Rok_narozeni']

# Bin ages into 1-year intervals: "0-0", "1-1", ..., "114-114"
bins = np.arange(0, 115 + AGE_BIN_WIDTH, AGE_BIN_WIDTH)
labels = [f"{i}-{i}" for i in range(0, 115)]
df['age_bin'] = pd.cut(df['age'], bins=bins, right=False, labels=labels)
df = df[df['age_bin'].notna()]  # Drop rows with unknown age
df['age_bin'] = df['age_bin'].astype('category')

# Count how many dose dates are available (non-null)
df['dose_count'] = df[dose_date_cols].notna().sum(axis=1)

# -------------------
# Causal effect estimation per age bin
# -------------------
results = []

for age_bin in df['age_bin'].cat.categories:
    sub = df[df['age_bin'] == age_bin].copy()

    # Skip bins with too few samples
    if len(sub) < 50:
        print(f"Skipping age bin {age_bin} due to small sample size ({len(sub)})")
        continue

    # Create causal model: dose_count -> death
    model = CausalModel(
        data=sub,
        treatment='dose_count',
        outcome='death',
        common_causes=[],  # Add confounders if available
        treatment_is_binary=False
    )

    identified_model = model.identify_effect()

    # Estimate treatment effect using linear regression
    estimate = model.estimate_effect(
        identified_model,
        method_name="backdoor.linear_regression"
    )

    # Perform significance test
    test_result = estimate.test_stat_significance()
    p_val = float(test_result['p_value'])  # Ensure scalar float
    significant = p_val < 0.05

    # Collect metrics
    death_rate = sub['death'].mean()
    mean_doses = sub['dose_count'].mean()
    N = len(sub)

    # Print summary
    print(
        f"Age bin {age_bin}: ATE = {estimate.value:.5f}, "
        f"Death Rate = {death_rate:.5f}, Mean Doses = {mean_doses:.2f}, N={N}, "
        f"p-value = {p_val:.4f}, Significant = {significant}"
    )

    results.append({
        'age_bin': age_bin,
        'age': int(age_bin.split('-')[0]),
        'ate': estimate.value,
        'death_rate': death_rate,
        'mean_doses': mean_doses,
        'p_value': p_val,
        'significant': significant,
        'N_total': N
    })

# Convert to DataFrame for plotting
df_causal = pd.DataFrame(results)

# -------------------
# Plot results
# -------------------
fig = go.Figure()

# Plot estimated ATE, colored by significance
fig.add_trace(go.Scatter(
    x=df_causal['age'],
    y=df_causal['ate'],
    mode='markers+lines',
    name='Estimated ATE',
    marker=dict(
        color=np.where(df_causal['significant'], 'green', 'red'),
        size=8,
        symbol='circle'
    ),
    hovertemplate=(
        'Age=%{x}<br>'
        'ATE (ΔDeathRisk per Dose)=%{y:.6f}<br>'
        'p-value=%{customdata[0]:.4f}<br>'
        'Significant=%{customdata[1]}'
    ),
    customdata=df_causal[['p_value', 'significant']]
))

# Plot observed death rate on secondary y-axis
fig.add_trace(go.Scatter(
    x=df_causal['age'],
    y=df_causal['death_rate'],
    mode='lines+markers',
    name='Observed Death Rate',
    yaxis='y2',
    marker=dict(color='black'),
    hovertemplate='Age=%{x}<br>Death Rate=%{y:.6f}'
))

# Plot mean dose count on tertiary y-axis
fig.add_trace(go.Scatter(
    x=df_causal['age'],
    y=df_causal['mean_doses'],
    mode='lines+markers',
    name='Mean Dose Count',
    yaxis='y3',
    marker=dict(color='blue'),
    hovertemplate='Age=%{x}<br>Mean Dose Count=%{y:.2f}'
))

# Configure layout with 3 y-axes
fig.update_layout(
    title="Estimated Causal Effect of Dose Count on real Death Risk by Age Bin (with Significance)",
    xaxis_title='Age',
    yaxis=dict(
        title='Estimated ATE (ΔDeath Risk per Dose)',
        side='left',
        showgrid=True,
        zeroline=True,
    ),
    yaxis2=dict(
        title='Observed Death Rate',
        overlaying='y',
        side='right',
        position=0.95
    ),
    yaxis3=dict(
        title='Mean Dose Count',
        anchor='free',
        overlaying='y',
        side='right',
        position=1,
        showgrid=False
    ),
    template='plotly_white',
    height=700
)

# Save output to HTML
fig.write_html(OUTPUT_HTML)
print(f"Plot saved to: {OUTPUT_HTML}")
