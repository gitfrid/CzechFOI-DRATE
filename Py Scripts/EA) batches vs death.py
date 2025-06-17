import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os

# --- Configuration ---
REFERENCE_DATE = pd.Timestamp("2020-01-01")
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
OUTPUT_DIR = r"C:\CzechFOI-DRATE\Plot Results\EA) batches vs death\hist_chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

current_year = 2023
DOSE_RANGE = range(-30, 61)  # days relative to dose
BATCHES_PER_FILE = 100

# --- Load and preprocess data ---
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

# Flatten doses
dose_records = []
for i in range(7):
    date_col = dose_date_cols[i]
    batch_col = batch_code_cols[i]
    mask = df[batch_col].notna() & df[date_col].notna()
    temp = df.loc[mask, ['DatumUmrti', date_col, batch_col]].copy()
    temp['dose_day'] = (temp[date_col] - REFERENCE_DATE).dt.days
    temp['death_day'] = (temp['DatumUmrti'] - REFERENCE_DATE).dt.days
    temp['batch'] = temp[batch_col]
    dose_records.append(temp[['batch', 'dose_day', 'death_day']])

long_df = pd.concat(dose_records, ignore_index=True)

# --- Determine all batch names ---
all_batches = long_df['batch'].value_counts().index.tolist()
batch_chunks = [all_batches[i:i + BATCHES_PER_FILE] for i in range(0, len(all_batches), BATCHES_PER_FILE)]

# --- Generate one HTML per chunk ---
for chunk_idx, batch_list in enumerate(batch_chunks):
    chunk_df = long_df[long_df['batch'].isin(batch_list)].copy()

    n_batches = len(batch_list)
    cols = 5
    rows = int(np.ceil(n_batches / cols))

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=batch_list,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=min(0.04, 1 / (rows + 1)),  # prevent spacing error
        horizontal_spacing=0.05,
        specs=[[{"secondary_y": True} for _ in range(cols)] for _ in range(rows)]
    )

    for i, batch in enumerate(batch_list):
        row = i // cols + 1
        col = i % cols + 1

        batch_df = chunk_df[chunk_df['batch'] == batch].dropna(subset=['dose_day'])

        dose_days = batch_df['dose_day'].tolist()

        # Vectorized death offset calculation
        offsets = batch_df['death_day'] - batch_df['dose_day']
        valid_offsets = offsets[(offsets >= min(DOSE_RANGE)) & (offsets <= max(DOSE_RANGE)) & offsets.notna()]
        death_offsets = valid_offsets.tolist()

        # Doses histogram
        fig.add_trace(go.Histogram(
            x=dose_days,
            xbins=dict(start=min(DOSE_RANGE), end=max(DOSE_RANGE) + 1, size=1),
            name='Doses' if i == 0 else None,
            marker_color='blue',
            opacity=0.6,
            showlegend=(i == 0)
        ), row=row, col=col, secondary_y=False)

        # Death offsets histogram
        fig.add_trace(go.Histogram(
            x=death_offsets,
            xbins=dict(start=min(DOSE_RANGE), end=max(DOSE_RANGE) + 1, size=1),
            name='Deaths' if i == 0 else None,
            marker_color='red',
            opacity=0.6,
            showlegend=(i == 0)
        ), row=row, col=col, secondary_y=True)

        fig.update_xaxes(title_text="Days Since Dose", row=row, col=col)
        fig.update_yaxes(title_text="Doses", row=row, col=col, secondary_y=False)
        fig.update_yaxes(title_text="Deaths", row=row, col=col, secondary_y=True)

    fig.update_layout(
        height=rows * 300,
        width=1800,
        title_text=f"Histograms of Doses and Deaths per Batch (Batches {chunk_idx * BATCHES_PER_FILE + 1}–{chunk_idx * BATCHES_PER_FILE + n_batches})",
        barmode='overlay',
        template='plotly_white'
    )

    out_file = os.path.join(
        OUTPUT_DIR,
        f"batch_hist_{chunk_idx * BATCHES_PER_FILE + 1}_to_{chunk_idx * BATCHES_PER_FILE + n_batches}.html"
    )
    fig.write_html(out_file)
    print(f"✅ Saved: {out_file}")
