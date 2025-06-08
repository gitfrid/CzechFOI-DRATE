import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dowhy import CausalModel
from pandas.api.types import CategoricalDtype

# ------------------------------------------------------------------------------
# DoWhy Causal Analysis: Estimating Average Treatment Effect (ATE) of Vaccine
# Dose Count on Death Outcome, Grouped by Age Bin and Vaccine Code.
#
# Data source: Czech simulated/real data with birth year, vaccine doses (dates
# and codes), and death dates. Performs causal inference per (age_bin, vax_code)
# pair and visualizes results using Plotly.
# ------------------------------------------------------------------------------

# -------------------
# Config
# -------------------
REFERENCE_DATE = pd.Timestamp("2020-01-01")  # Reference start date for calculating relative days
AGE_BIN_WIDTH = 1  # Width of each age bin
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_Vesely_106_202403141131.csv"  # Simulated test data
#INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"    # Real Czech FOI data
OUTPUT_HTML = r"C:\github\CzechFOI-DRATE\Plot Results\ZC) dowhy vaxcode doses vs totaldeath individual\ZC) sim dowhy_vaxcode_doses_vs_deaths.html"
CURRENT_YEAR = 2023  # Used for age calculation

# -------------------
# Load data
# -------------------
# Columns for dose dates and vaccine codes
dose_date_cols = [f'Datum_{i}' for i in range(1, 8)]
vax_code_cols = [f'OckovaciLatkaKod_{i}' for i in range(1, 8)]
all_cols = ['Rok_narozeni', 'DatumUmrti'] + dose_date_cols + vax_code_cols

# Read the CSV, parsing all relevant dates
df = pd.read_csv(
    INPUT_CSV,
    usecols=all_cols,
    parse_dates=['DatumUmrti'] + dose_date_cols,
    dayfirst=False,
    low_memory=False
)

# -------------------
# Preprocessing
# -------------------

# Compute first dose date and convert to days since reference
first_dose = df[dose_date_cols].min(axis=1)
df['first_dose_day'] = (first_dose - REFERENCE_DATE).dt.days

# Compute death day and death indicator
df['death_day'] = (df['DatumUmrti'] - REFERENCE_DATE).dt.days
df['death'] = (~df['DatumUmrti'].isna()).astype(int)

# Compute age from birth year
df['age'] = CURRENT_YEAR - df['Rok_narozeni']

# Bin age into categorical labels (e.g., "70-70", "71-71", ...)
bins = np.arange(0, 115 + AGE_BIN_WIDTH, AGE_BIN_WIDTH)
labels = [f"{i}-{i}" for i in range(0, 115)]
cat_type = CategoricalDtype(categories=labels, ordered=True)
df['age_bin'] = pd.cut(df['age'], bins=bins, right=False, labels=labels)
df = df[df['age_bin'].notna()]  # Remove rows with invalid age bin
df['age_bin'] = df['age_bin'].astype(cat_type)

# Count number of non-null doses per individual
df['dose_count'] = df[dose_date_cols].notna().sum(axis=1)

# Assign first available vaccine code (bfill across columns)
df['vax_code'] = df[vax_code_cols].bfill(axis=1).iloc[:, 0]
df['vax_code'] = df['vax_code'].fillna('None')  # Unvaccinated marked as 'None'

# -------------------
# Causal effect estimation per (age_bin, vax_code)
# -------------------
results = []

# Group by age bin and vaccine code
grouped = df.groupby(['age_bin', 'vax_code'])

for (age_bin, vax_code), sub in grouped:
    # Skip small groups
    if len(sub) < 50:
        print(f"Skipping age bin {age_bin}, vax_code {vax_code} due to small sample size ({len(sub)})")
        continue

    try:
        # Create causal model
        model = CausalModel(
            data=sub,
            treatment='dose_count',
            outcome='death',
            common_causes=[],  # Extend if confounders are available
            treatment_is_binary=False
        )

        # Identify causal effect
        identified_model = model.identify_effect()

        # Estimate effect using linear regression
        estimate = model.estimate_effect(
            identified_model,
            method_name="backdoor.linear_regression"
        )

        # Test statistical significance
        test_result = estimate.test_stat_significance()
        p_val = float(test_result['p_value'])
        significant = p_val < 0.05

        # Store result metrics
        death_rate = sub['death'].mean()
        mean_doses = sub['dose_count'].mean()
        age = int(age_bin.split('-')[0])
        N = len(sub)

        results.append({
            'age_bin': age_bin,
            'age': age,
            'vax_code': vax_code,
            'ate': estimate.value,
            'death_rate': death_rate,
            'mean_doses': mean_doses,
            'p_value': p_val,
            'significant': significant,
            'N_total': N
        })

        print(f"{age_bin} | {vax_code} | ATE={estimate.value:.5f}, p={p_val:.4f}, N={N}")

    except Exception as e:
        # Handle errors gracefully
        print(f"Failed for age_bin {age_bin}, vax_code {vax_code}: {str(e)}")

# Convert results to DataFrame
df_causal = pd.DataFrame(results)

# -------------------
# Plot results
# -------------------
fig = go.Figure()

# Plot ATE (Average Treatment Effect) for each vaccine code
for vax_code, sub in df_causal.groupby('vax_code'):
    fig.add_trace(go.Scatter(
        x=sub['age'],
        y=sub['ate'],
        mode='markers+lines',
        name=f'ATE - {vax_code}',
        marker=dict(
            size=8,
            symbol='circle',
            color=np.where(sub['significant'], 'green', 'gray')  # Green if p < 0.05
        ),
        customdata=sub[['p_value', 'significant', 'N_total']],
        hovertemplate=(
            'Age=%{x}<br>'
            'ATE=%{y:.6f}<br>'
            'p-value=%{customdata[0]:.4f}<br>'
            'Significant=%{customdata[1]}<br>'
            'N=%{customdata[2]}<br>'
            f'Vax Code={vax_code}'
        )
    ))

# Overlay death rate (black line) - not stratified by vax_code
fig.add_trace(go.Scatter(
    x=df_causal['age'],
    y=df_causal['death_rate'],
    mode='lines',
    name='Death Rate',
    yaxis='y2',
    line=dict(color='black', width=1.5),
    hovertemplate='Age=%{x}<br>Death Rate=%{y:.6f}'
))

# Overlay mean dose count (blue line) - not stratified by vax_code
fig.add_trace(go.Scatter(
    x=df_causal['age'],
    y=df_causal['mean_doses'],
    mode='lines',
    name='Mean Dose Count',
    yaxis='y3',
    line=dict(color='blue', width=1.5),
    hovertemplate='Age=%{x}<br>Mean Doses=%{y:.2f}'
))

# Configure plot layout with 3 Y-axes
fig.update_layout(
    title="DoWhy ATE of Dose Count on Death by Age Bin and Vaccine Code",
    xaxis_title='Age',
    yaxis=dict(
        title='Estimated ATE (ΔDeath Risk per Dose)',
        side='left',
        showgrid=True
    ),
    yaxis2=dict(
        title='Death Rate',
        overlaying='y',
        side='right',
        position=0.95
    ),
    yaxis3=dict(
        title='Mean Dose Count',
        overlaying='y',
        anchor='free',
        side='right',
        position=1.0,
        showgrid=False
    ),
    template='plotly_white',
    height=750
)

# Export to interactive HTML
fig.write_html(OUTPUT_HTML)
print(f"Plot saved to: {OUTPUT_HTML}")
