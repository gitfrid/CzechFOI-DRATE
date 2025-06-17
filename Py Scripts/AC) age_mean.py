# This script processes data from pivot CSV files located in the TERRA folder, 
# which were generated from a Czech Freedom of Information request (Vesely_106_202403141131.csv). 
# The pivot CSV files were created using the DB Browser for SQLite.

import pandas as pd
import plotly.graph_objects as go

# --- Config ---
REFERENCE_DATE = pd.Timestamp("2020-01-01")
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_SELCTION_BIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
OUTPUT_HTML = r"C:\CzechFOI-DRATE\Plot Results\AC) age mean pop\AC) age_mean_pop.html"
start_delay = 0
ROLLING_WINDOW = 7

# --- Load and preprocess ---
dose_date_cols = [f"Datum_{i}" for i in range(1, 8)]
df = pd.read_csv(INPUT_CSV, parse_dates=dose_date_cols + ["DatumUmrti"], low_memory=False)

df["age"] = 2023 - df["Rok_narozeni"]
df = df[df["age"].between(0, 113)].copy()

# Convert dates to days since reference
for col in dose_date_cols:
    df[col] = (df[col] - REFERENCE_DATE).dt.days
df["t_death"] = (df["DatumUmrti"] - REFERENCE_DATE).dt.days

# --- Deaths by age and day ---
df_deaths = df.dropna(subset=["t_death"]).copy()
df_deaths["day"] = df_deaths["t_death"].astype(int)
df_deaths["age_int"] = df_deaths["age"].astype(int)

# Pivot to create table: rows=day, cols=age, values=counts
death_df = df_deaths.groupby(["day", "age_int"]).size().unstack(fill_value=0).sort_index()

# Total deaths per day
daily_total_deaths = death_df.sum(axis=1)

# Age-weighted deaths per day
daily_weighted_sum = (death_df * death_df.columns).sum(axis=1)

# Mean age of death
mean_age_of_death = daily_weighted_sum / daily_total_deaths

# --- Vaccine doses by age and day ---
# Melt all dose dates into a single table
df_doses = df.melt(
    id_vars=["age"],
    value_vars=dose_date_cols,
    value_name="dose_day"
).dropna(subset=["dose_day"]).copy()

df_doses["day"] = df_doses["dose_day"].astype(int)
df_doses["age_int"] = df_doses["age"].astype(int)

# Pivot to create table: rows=day, cols=age, values=counts
dose_df = df_doses.groupby(["day", "age_int"]).size().unstack(fill_value=0).sort_index()

# Total doses per day
daily_total_doses = dose_df.sum(axis=1)

# --- Rolling Means ---
rolling_weighted_deaths = daily_weighted_sum.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
rolling_total_doses = daily_total_doses.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
rolling_mean_age = mean_age_of_death.rolling(window=ROLLING_WINDOW, min_periods=1).mean()

# First day when total doses exceed 100
first_vaccine_day = daily_total_doses[daily_total_doses > 100].index.min()

# --- Compute mean age before and after vaccination start ---
mean_age_before_vac = mean_age_of_death[mean_age_of_death.index < first_vaccine_day].mean()
mean_age_after_vac = mean_age_of_death[mean_age_of_death.index >= first_vaccine_day].mean()

# --- Plotting ---
fig = go.Figure()

# Raw traces
fig.add_trace(go.Scatter(
    x=daily_total_deaths.index,
    y=daily_total_deaths.values,
    mode='lines',
    name='Total deaths per day',
    line=dict(color='blue')
))

fig.add_trace(go.Scatter(
    x=daily_weighted_sum.index,
    y=daily_weighted_sum.values,
    mode='lines',
    name='Age-weighted deaths per day',
    line=dict(color='red')
))

fig.add_trace(go.Scatter(
    x=daily_total_doses.index,
    y=daily_total_doses.values,
    mode='lines',
    name='Total vaccine doses per day',
    line=dict(color='green')
))

fig.add_trace(go.Scatter(
    x=mean_age_of_death.index,
    y=mean_age_of_death.values,
    mode='lines',
    name='Mean age of death',
    line=dict(color='grey'),
    yaxis='y2'
))

# Rolling averages
fig.add_trace(go.Scatter(
    x=rolling_weighted_deaths.index,
    y=rolling_weighted_deaths.values,
    mode='lines',
    name='Weighted deaths (7-day avg)',
    line=dict(color='red', width=0.7)
))

fig.add_trace(go.Scatter(
    x=rolling_total_doses.index,
    y=rolling_total_doses.values,
    mode='lines',
    name='Vac doses (7-day avg)',
    line=dict(color='green', width=0.7)
))

fig.add_trace(go.Scatter(
    x=rolling_mean_age.index,
    y=rolling_mean_age.values,
    mode='lines',
    name='Mean age of death (7-day avg)',
    line=dict(color='black', width=0.7),
    yaxis='y2'
))

# Layout and markers
fig.update_layout(
    title='Daily Deaths, Age-Weighted Deaths, Vaccine Doses, and Mean Age of Death (7-Day Averages)',
    xaxis_title='Day',
    yaxis=dict(
        title='Count / Weighted Sum',
    ),
    yaxis2=dict(
        title='Mean Age of Death',
        overlaying='y',
        side='right',
        showgrid=False
    ),
    legend_title='Metric',
    template='plotly_white',
    shapes=[
        # Vertical line at start of vaccination
        dict(
            type='line',
            x0=first_vaccine_day,
            x1=first_vaccine_day,
            y0=0,
            y1=1,
            yref='paper',
            line=dict(color='purple', width=2, dash='dot')
        ),
        # Horizontal mean age before vaccination
        dict(
            type='line',
            x0=mean_age_of_death.index.min(),
            x1=first_vaccine_day,
            y0=mean_age_before_vac,
            y1=mean_age_before_vac,
            line=dict(color="orange", width=2, dash="dash"),
            yref="y2"
        ),
        # Horizontal mean age after vaccination
        dict(
            type='line',
            x0=first_vaccine_day,
            x1=mean_age_of_death.index.max(),
            y0=mean_age_after_vac,
            y1=mean_age_after_vac,
            line=dict(color="darkorange", width=2, dash="dot"),
            yref="y2"
        )
    ],
    annotations=[
        dict(
            x=first_vaccine_day,
            y=1.02,
            xref='x',
            yref='paper',
            text='Start of Vac (>100 doses)',
            showarrow=False,
            font=dict(color='purple'),
            align='center'
        ),
        dict(
            x=mean_age_of_death.index.min(),
            y=mean_age_before_vac,
            text=f"Mean age before vac: {mean_age_before_vac:.1f}",
            showarrow=False,
            yref="y2",
            xanchor="left",
            font=dict(color="orange")
        ),
        dict(
            x=mean_age_of_death.index.max(),
            y=mean_age_after_vac,
            text=f"Mean age after vac: {mean_age_after_vac:.1f}",
            showarrow=False,
            yref="y2",
            xanchor="right",
            font=dict(color="darkorange")
        )
    ]
)

# Save plot
fig.write_html(OUTPUT_HTML)
print(f"Plot saved as {OUTPUT_HTML}")
