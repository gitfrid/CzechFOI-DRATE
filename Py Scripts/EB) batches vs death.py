import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from pandas.api.types import CategoricalDtype

# --- Configuration ---
REFERENCE_DATE = pd.Timestamp("2020-01-01")
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
OUTPUT_HTML_LINE = r"C:\CzechFOI-DRATE\Plot Results\EB) batches vs death\E1) lineplot_batches_vs_death.html"
OUTPUT_HTML_HEATMAP = r"C:\CzechFOI-DRATE\Plot Results\EB) batches vs death\E2) heatmap_batches_vs_death.html"
AGE_BIN_WIDTH = 1
current_year = 2023

# --- Load and prepare data ---
dose_date_cols = [f'Datum_{i}' for i in range(1, 8)]
batch_code_cols = [f'Sarze_{i}' for i in range(1, 8)]

df = pd.read_csv(
    INPUT_CSV,
    usecols=['Rok_narozeni', 'DatumUmrti'] + dose_date_cols + batch_code_cols,
    parse_dates=['DatumUmrti'] + dose_date_cols,
    low_memory=False
)

df['death'] = ~df['DatumUmrti'].isna()
df['age'] = current_year - df['Rok_narozeni']

bins = list(range(0, 115, AGE_BIN_WIDTH))
age_labels = [f"{i}-{i+AGE_BIN_WIDTH-1}" for i in range(0, 114, AGE_BIN_WIDTH)]
cat_type = CategoricalDtype(categories=age_labels, ordered=True)
df['age_bin'] = pd.cut(df['age'], bins=bins, right=False, labels=age_labels, include_lowest=True)
df['age_bin'] = df['age_bin'].astype(cat_type)

# Flatten doses and batches into long format
records = []
for i in range(7):
    date_col = dose_date_cols[i]
    batch_col = batch_code_cols[i]
    valid = df[batch_col].notna()
    temp = df.loc[valid, ['age_bin', 'death', batch_col]].copy()
    temp = temp.rename(columns={batch_col: 'batch'})
    records.append(temp)

long_df = pd.concat(records, ignore_index=True)

# Group by batch and age_bin
grouped = long_df.groupby(['batch', 'age_bin'], observed=True)
summary = grouped['death'].agg(['count', 'sum']).reset_index()
summary.rename(columns={'count': 'n', 'sum': 'deaths'}, inplace=True)
summary['death_rate'] = 100 * summary['deaths'] / summary['n']  # percent

# Filter to only batches with enough data
min_count_per_batch = 20
batch_counts = summary.groupby('batch')['n'].sum()
valid_batches = batch_counts[batch_counts >= min_count_per_batch].index
summary = summary[summary['batch'].isin(valid_batches)]

# --- Fix for heatmap ---
all_batches = sorted(summary['batch'].unique())
all_ages = age_labels

full_index = pd.MultiIndex.from_product([all_batches, all_ages], names=['batch', 'age_bin'])
summary_full = summary.set_index(['batch', 'age_bin']).reindex(full_index).reset_index()

summary_full['n'] = summary_full['n'].fillna(0)
summary_full['deaths'] = summary_full['deaths'].fillna(0)

summary_full['death_rate'] = summary_full.apply(
    lambda row: 100 * row['deaths'] / row['n'] if row['n'] > 0 else np.nan,
    axis=1
)

heatmap_data = summary_full.pivot(index='age_bin', columns='batch', values='death_rate')
heatmap_data = heatmap_data.loc[all_ages, all_batches]
heatmap_data = heatmap_data.astype(float)

# --- Debug print ---
print("Heatmap data preview:")
print(heatmap_data.head())
print("Total NaNs:", heatmap_data.isna().sum().sum())

# --- Line Plot ---
fig_line = go.Figure()

for batch in all_batches:
    data = summary[summary['batch'] == batch]
    fig_line.add_trace(go.Scatter(
        x=data['age_bin'],
        y=data['death_rate'],
        mode='lines+markers',
        name=batch,
        hovertemplate=(
            f"<b>Batch:</b> {batch}<br>" +
            "Age: %{x}<br>" +
            "Death rate: %{y:.2f}%<br>" +
            "Deaths: %{customdata[0]} / %{customdata[1]}<extra></extra>"
        ),
        customdata=np.stack((data['deaths'], data['n']), axis=-1)
    ))

fig_line.update_layout(
    title="Death Rate per Batch Code by Age Bin (1-year)",
    xaxis_title="Age Bin",
    yaxis_title="Death Rate (%)",
    xaxis=dict(type='category', categoryorder='array', categoryarray=all_ages),
    template="plotly_white",
    height=800
)

fig_line.write_html(OUTPUT_HTML_LINE)
print(f"Saved line plot to {OUTPUT_HTML_LINE}")

# --- Heatmap ---
max_rate = summary_full['death_rate'].max()
if pd.isna(max_rate) or max_rate == 0:
    max_rate = 0.1

fig_heatmap = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=all_batches,
    y=all_ages,
    colorscale='Reds',
    zmin=0,
    zmax=max_rate,
    colorbar=dict(title="Death Rate (%)")
))

fig_heatmap.update_layout(
    title="Heatmap of Death Rate (%) by Age Bin and Batch",
    xaxis=dict(title="Batch"),
    yaxis=dict(title="Age Bin", autorange='reversed', type='category'),
    height=1000,
    template="plotly_white"
)

fig_heatmap.write_html(OUTPUT_HTML_HEATMAP)
print(f"Saved heatmap to {OUTPUT_HTML_HEATMAP}")
