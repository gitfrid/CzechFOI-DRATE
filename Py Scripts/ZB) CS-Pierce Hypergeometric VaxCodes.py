# Hypergeometric (C.S. Peirce) Vaccine Effectiveness Analysis with Confidence Intervals
# =======================================================================

#This script:
# - Loads individual-level vaccination and death data.
# - Classifies individuals by vaccination status and vaccine code.
# - Computes observed vaccine effectiveness using the hypergeometric test.
# - Calculates Wilson confidence intervals for death rates stratified by age bin and vaccine code.
# - Visualizes the results in a two-row Plotly HTML figure:
#     1. Overall difference in death rates (vaccinated vs. unvaccinated) with null distribution.
#     2. Per-age-bin death rates by vaccine code, with 95% Wilson confidence intervals.

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.proportion import proportion_confint
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from pandas.api.types import CategoricalDtype

# Configurations
REFERENCE_DATE = pd.Timestamp("2020-01-01")
# INPUT_CSV = r"C:\github\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv" # -> real czech FOI data 
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_Vesely_106_202403141131.csv" # -> real real czech FOI data - with homogen constant simulated death rate
OUTPUT_HTML = r"C:\github\CzechFOI-DRATE\Plot Results\ZB) CS-Pierce Hypergeometric VaxCodes\ZB) sim hypergeom_vaccine_effectiveness_with_CI_and_vaxcode_age.html"
OUTPUT_CSV = r"C:\github\CzechFOI-DRATE\Plot Results\ZB) CS-Pierce Hypergeometric VaxCodes\ZB) sim hypergeom_results_by_age_and_vaxcode.csv"
AGE_BIN_WIDTH = 1

# --- Load and prepare data ---

# Define column names for up to 7 dose dates and vaccine codes
dose_date_cols = [f'Datum_{i}' for i in range(1, 8)]
vax_code_cols = [f'OckovaciLatkaKod_{i}' for i in range(1, 8)]

# Read relevant columns with date parsing
df = pd.read_csv(
    INPUT_CSV,
    usecols=['Rok_narozeni', 'DatumUmrti'] + dose_date_cols + vax_code_cols,
    parse_dates=['DatumUmrti'] + dose_date_cols,
    low_memory=False
)

# Determine if person was vaccinated (at least one non-NaN dose date)
first_dose = df[dose_date_cols].min(axis=1)
df['vaccinated'] = ~first_dose.isna()

# Determine if person died (non-NaN death date)
df['death'] = ~df['DatumUmrti'].isna()

# Compute current age as of year 2023
current_year = 2023
df['age'] = current_year - df['Rok_narozeni']

# Bin ages into 1-year intervals from 0–114
bins = list(range(0, 115, AGE_BIN_WIDTH))
age_labels = [f"{i}-{i}" for i in range(0, 114)]
cat_type = CategoricalDtype(categories=age_labels, ordered=True)
df['age_bin'] = pd.cut(df['age'], bins=bins, right=False, labels=age_labels, include_lowest=True)
df['age_bin'] = df['age_bin'].astype(cat_type)


# Determine first vaccine code (corresponding to the first non-NaN dose)
def get_first_vax_code(row):
    for i in range(7):
        if pd.notna(row[dose_date_cols[i]]):
            return row[vax_code_cols[i]]
    return np.nan

df['first_vax_code'] = df.apply(get_first_vax_code, axis=1).astype('category')

# Assign vaccine group: real code if vaccinated, '0' if unvaccinated
df['vax_code_group'] = df['first_vax_code'].cat.add_categories('0')
df.loc[~df['vaccinated'], 'vax_code_group'] = '0'

# --- (C.S. Pierce) Hypergeometric test on all-age totals ---

def hypergeom_test(N_v, N_p, D, k):
    # N_v: vaccinated count
    # N_p: unvaccinated count
    # D: total deaths
    # k: vaccinated deaths
    M = N_v + N_p  # total population
    rv = hypergeom(M, D, N_v)
    p_lower = rv.cdf(k)
    p_upper = rv.sf(k - 1)
    p_val = 2 * min(p_lower, p_upper)  # two-tailed
    ci_v = proportion_confint(k, N_v, alpha=0.05, method='wilson')
    ci_p = proportion_confint(D - k, N_p, alpha=0.05, method='wilson')
    return p_val, ci_v, ci_p, rv

# Compute totals
N_v = df['vaccinated'].sum()
N_p = len(df) - N_v
D = df['death'].sum()
k = df.loc[df['vaccinated'], 'death'].sum()

# Calculate proportions
p_vax = k / N_v if N_v > 0 else 0
p_plac = (D - k) / N_p if N_p > 0 else 0

# Run hypergeometric test
p_val, ci_v, ci_p, rv = hypergeom_test(N_v, N_p, D, k)

# --- Null distribution for visualizing test statistic ---

# Simulate all possible k values and compute rate differences
rate_diffs = np.arange(0, D + 1) / N_v - (D - np.arange(0, D + 1)) / N_p
pmf = rv.pmf(np.arange(0, D + 1))

# Histogram binning for rate differences
bins_edge = np.linspace(rate_diffs.min(), rate_diffs.max(), 100)
bin_idx = np.digitize(rate_diffs, bins_edge) - 1
hist_counts = np.zeros(len(bins_edge) - 1)

# Accumulate probabilities into histogram bins
for i, prob in zip(bin_idx, pmf):
    if 0 <= i < len(hist_counts):
        hist_counts[i] += prob

# X-axis centers and hover text
bin_centers = (bins_edge[:-1] + bins_edge[1:]) / 2
ohover = [f"Diff≈{bc:.5f}<br>Prob≈{hc:.3g}" for bc, hc in zip(bin_centers, hist_counts)]

# --- Stratified death rate CI per age bin and vax code ---

results = []
age_groups = sorted(df['age_bin'].dropna().unique())
vax_codes = sorted(df['vax_code_group'].dropna().unique())

# Loop over each (age bin, vaccine code) pair
for age in age_groups:
    for vc in vax_codes:
        sub = df[(df['age_bin'] == age) & (df['vax_code_group'] == vc)]
        n = len(sub)
        deaths = sub['death'].sum()
        if n == 0:
            continue
        rate = deaths / n
        ci_low, ci_high = proportion_confint(deaths, n, alpha=0.05, method='wilson')
        results.append({
            'age_bin': age,
            'vax_code': vc,
            'count': n,
            'deaths': deaths,
            'death_rate': rate,
            'ci_low': ci_low,
            'ci_high': ci_high
        })

df_results = pd.DataFrame(results)

# Save results to CSV
df_results.to_csv(OUTPUT_CSV, index=False)
print(f"Results saved to {OUTPUT_CSV}")

# --- Plotting ---

fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=(
        'Overall Vaccine Effect (Exact Test, Null Distribution)',
        'Death Rates by Age Bin Stratified by Vaccine Code (95% Wilson CI)'
    ),
    vertical_spacing=0.15
)

# Row 1: Overall null distribution and observed difference
fig.add_trace(
    go.Bar(
        x=bin_centers, y=hist_counts,
        marker=dict(color='lightgrey'),
        hovertext=ohover,
        name='Overall Null Distribution'
    ), row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=[p_vax - p_plac], y=[hist_counts.max()],
        mode='markers',
        marker=dict(color='red', size=10),
        hovertext=f"p_vax={p_vax:.5f}<br>p_plac={p_plac:.5f}<br>p-value={p_val:.3g}",
        hoverinfo='text',
        name='Observed Difference'
    ), row=1, col=1
)

# Row 2: Death rates by age bin & vaccine code
import plotly.express as px

# Assign colors to each vaccine code group
unique_vax_codes = sorted(df_results['vax_code'].unique())
colors = px.colors.qualitative.Plotly
vax_code_colors = {vc: colors[i % len(colors)] for i, vc in enumerate(unique_vax_codes)}
if '0' in vax_code_colors:
    vax_code_colors['0'] = 'gray'  # gray for unvaccinated

# Extract the numeric start of the bin for proper sorting
# x_age_bins = sorted(df_results['age_bin'].unique(), key=lambda x: int(str(x).split('-')[0]))
x_age_bins = df['age_bin'].cat.categories

# Add death rate line with CI for each vax code group
for vc in unique_vax_codes:
    df_vc = df_results[df_results['vax_code'] == vc]
    y_rates = []
    y_err_low = []
    y_err_high = []
    for age in x_age_bins:
        row = df_vc[df_vc['age_bin'] == age]
        if not row.empty:
            y_rates.append(row['death_rate'].values[0])
            y_err_low.append(row['death_rate'].values[0] - row['ci_low'].values[0])
            y_err_high.append(row['ci_high'].values[0] - row['death_rate'].values[0])
        else:
            y_rates.append(np.nan)
            y_err_low.append(0)
            y_err_high.append(0)
    fig.add_trace(
        go.Scatter(
            x=x_age_bins,
            y=y_rates,
            mode='lines+markers',
            name=f'Vax Code {vc}',
            line=dict(color=vax_code_colors.get(vc, 'black')),
            error_y=dict(
                type='data',
                symmetric=False,
                array=y_err_high,
                arrayminus=y_err_low,
                thickness=1.5,
                width=5
            ),
            connectgaps=False,
            hovertemplate=(
                'Age %{x}<br>Vax Code %{customdata[0]}<br>Death Rate %{y:.5f}<br>' +
                '95% CI [%{customdata[1]:.5f}, %{customdata[2]:.5f}]<extra></extra>'
            ),
            customdata=np.stack((
                [vc]*len(x_age_bins),
                np.array(y_rates) - np.array(y_err_low),
                np.array(y_rates) + np.array(y_err_high)
            ), axis=-1)
        ), row=2, col=1
    )

# Layout and axis titles
fig.update_layout(
    height=900,
    template='plotly_white',
    legend=dict(orientation='h', yanchor='bottom', y=-0.25),
    xaxis2=dict(type='category'),
    title_text="Hypergeometric (C.S. Peirce) Vaccine Effectiveness Analysis by Age and Vaccine Code"
)

fig.update_xaxes(title_text="Age Bin", row=2, col=1)
fig.update_yaxes(title_text="Death Rate", row=2, col=1)
fig.update_xaxes(title_text="Difference in Death Rates (Vaccinated - Unvaccinated)", row=1, col=1)
fig.update_yaxes(title_text="Probability", row=1, col=1)

# Save plot to HTML
fig.write_html(OUTPUT_HTML)
print(f"Plot saved to {OUTPUT_HTML}")