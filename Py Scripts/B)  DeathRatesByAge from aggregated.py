import pandas as pd
import numpy as np
import plotly.graph_objs as go

#This script reads vaccine-related CSV data files where rows = days (0–1533)
#and columns = ages (0–113), each file corresponding to a metric:

# - deaths_total.csv
# - deaths_vx.csv
# - deaths_uvx.csv
# - population_total.csv
# - population_vx.csv
# - population_uvx.csv
# - doses_first.csv
# - doses_all.csv

# It plots:
# - Death rates (total, vx, uvx): raw, normalized per 100k (based on remaining population), and smoothed
# - Population (total, vx, uvx): sum over ages (right Y-axis)
# - Doses (first, all): raw and smoothed, sum over ages (third Y-axis)
# All traces are saved into a single interactive Plotly HTML file.

# === Config ===
ROLLING_WINDOW = 7
DAYS_RANGE = (0, 1533)
AGES = list(range(114))
MIN_POPULATION = 30  # Threshold below which normalized values are masked

INPUT_DIR = r"C:\github\CzechFOI-DRATE\TERRA"
OUTPUT_HTML = r"C:\github\CzechFOI-DRATE\Plot Results\B) DeathRates byage from aggregated csvfiles\B) DeathRates PerAgeGroup.html"

# === Load Data ===
def load_csv(name):
    df = pd.read_csv(fr"{INPUT_DIR}\{name}")
    df = df.iloc[DAYS_RANGE[0]:DAYS_RANGE[1]+1, :]
    df.rename(columns={df.columns[0]: "DAY"}, inplace=True)
    age_columns = df.columns[1:].astype(int)
    df.columns = ["DAY"] + list(age_columns)
    return df

deaths_total = load_csv("deaths_total.csv")
deaths_vx = load_csv("deaths_vx.csv")
deaths_uvx = load_csv("deaths_uvx.csv")
pop_total = load_csv("population_total.csv")
pop_vx = load_csv("population_vx.csv")
pop_uvx = load_csv("population_uvx.csv")
doses_first = load_csv("first_doses.csv")
doses_all = load_csv("all_doses.csv")

days = deaths_total["DAY"]
fig = go.Figure()

def smooth(s):
    return s.rolling(ROLLING_WINDOW, center=True, min_periods=1).mean()

for age in AGES:
    # Daily populations
    daily_pop_total = pop_total[age]
    daily_pop_vx = pop_vx[age]
    daily_pop_uvx = pop_uvx[age]

    # Cumulative deaths
    cum_deaths_total = deaths_total[age].cumsum()
    cum_deaths_vx = deaths_vx[age].cumsum()
    cum_deaths_uvx = deaths_uvx[age].cumsum()

    # Remaining population
    remaining_total = (daily_pop_total - cum_deaths_total).clip(lower=1)
    remaining_vx = (daily_pop_vx - cum_deaths_vx).clip(lower=1)
    remaining_uvx = (daily_pop_uvx - cum_deaths_uvx).clip(lower=1)

    # Raw deaths
    y_raw_total = deaths_total[age]
    y_raw_vx = deaths_vx[age]
    y_raw_uvx = deaths_uvx[age]

    # Normalized death rates with masking
    y_norm_total = y_raw_total / remaining_total * 100_000
    y_norm_total[remaining_total < MIN_POPULATION] = np.nan

    y_norm_vx = y_raw_vx / remaining_vx * 100_000
    y_norm_vx[remaining_vx < MIN_POPULATION] = np.nan

    y_norm_uvx = y_raw_uvx / remaining_uvx * 100_000
    y_norm_uvx[remaining_uvx < MIN_POPULATION] = np.nan

    # Traces
    fig.add_trace(go.Scatter(x=days, y=y_raw_total, name=f"Age {age} Total (raw)",
                             visible="legendonly", line=dict(width=0.8)))
    fig.add_trace(go.Scatter(x=days, y=y_raw_vx, name=f"Age {age} Vx (raw)",
                             visible="legendonly", line=dict(width=0.8)))
    fig.add_trace(go.Scatter(x=days, y=y_raw_uvx, name=f"Age {age} Uvx (raw)",
                             visible="legendonly", line=dict(width=0.8)))

    fig.add_trace(go.Scatter(x=days, y=y_norm_total, name=f"Age {age} Total (/100k)",
                             visible="legendonly", line=dict(width=0.8)))
    fig.add_trace(go.Scatter(x=days, y=y_norm_vx, name=f"Age {age} Vx (/100k)",
                             visible="legendonly", line=dict(width=0.8)))
    fig.add_trace(go.Scatter(x=days, y=y_norm_uvx, name=f"Age {age} Uvx (/100k)",
                             visible="legendonly", line=dict(width=0.8)))

    fig.add_trace(go.Scatter(x=days, y=smooth(y_norm_total), name=f"Age {age} Total (/100k, smoothed)",
                             visible="legendonly", line=dict(width=0.8)))
    fig.add_trace(go.Scatter(x=days, y=smooth(y_norm_vx), name=f"Age {age} Vx (/100k, smoothed)",
                             visible="legendonly", line=dict(width=0.8)))
    fig.add_trace(go.Scatter(x=days, y=smooth(y_norm_uvx), name=f"Age {age} Uvx (/100k, smoothed)",
                             visible="legendonly", line=dict(width=0.8)))

    # Smoothed dose counts
    fig.add_trace(go.Scatter(x=days, y=smooth(doses_all[age]), mode='lines',
        name=f'Age {age} - All Doses (smoothed)', visible='legendonly',
        yaxis='y2', line=dict(width=0.8)))
    fig.add_trace(go.Scatter(x=days, y=smooth(doses_first[age]), mode='lines',
        name=f'Age {age} - First Doses (smoothed)', visible='legendonly',
        yaxis='y2', line=dict(width=0.8)))

    # Smoothed remaining population
    fig.add_trace(go.Scatter(x=days, y=smooth(remaining_total), mode='lines',
        name=f'Age {age} - Total Population Remaining', visible='legendonly',
        yaxis='y3', line=dict(color='green', width=0.8)))
    fig.add_trace(go.Scatter(x=days, y=smooth(remaining_vx), mode='lines',
        name=f'Age {age} - Vx Population Remaining', visible='legendonly',
        yaxis='y3', line=dict(color='blue', width=0.8)))
    fig.add_trace(go.Scatter(x=days, y=smooth(remaining_uvx), mode='lines',
        name=f'Age {age} - Uvx Population Remaining', visible='legendonly',
        yaxis='y3', line=dict(color='red', width=0.8)))

    # Mean normalized lines from first vax day onward
    vax_start_day = (doses_all.iloc[:, 1:].sum(axis=1) > 0).idxmax()
    vax_start_idx = vax_start_day

    mean_total = y_norm_total.iloc[vax_start_idx:].mean(skipna=True)
    mean_vx = y_norm_vx.iloc[vax_start_idx:].mean(skipna=True)
    mean_uvx = y_norm_uvx.iloc[vax_start_idx:].mean(skipna=True)

    fig.add_trace(go.Scatter(x=days, y=[mean_total] * len(days),
                             name=f"Age {age} Mean Total (/100k)", visible='legendonly',
                             line=dict(dash='dash', color='black', width=1)))
    fig.add_trace(go.Scatter(x=days, y=[mean_vx] * len(days),
                             name=f"Age {age} Mean Vx (/100k)", visible='legendonly',
                             line=dict(dash='dot', color='blue', width=1)))
    fig.add_trace(go.Scatter(x=days, y=[mean_uvx] * len(days),
                             name=f"Age {age} Mean Uvx (/100k)", visible='legendonly',
                             line=dict(dash='dot', color='red', width=1)))

# === Layout ===
fig.update_layout(
    title="Death Rates per Age Group (Raw, Normalized, Smoothed) + Doses + Remaining Population",
    xaxis=dict(title="Days since start"),
    yaxis=dict(title="Deaths (/100k or raw)"),
    yaxis2=dict(title="Doses (smoothed)", overlaying="y", side="right"),
    yaxis3=dict(title="Remaining Population", anchor="free", overlaying="y", side="right", position=0.85),
    height=900,
    width=2000,
    margin=dict(r=400),
    legend=dict(
        x=1.05,
        y=1,
        orientation="v",
        traceorder="normal",
        itemsizing="constant",
        font=dict(size=9),
        bordercolor="black",
        borderwidth=1
    )
)

# === Save Plot ===
fig.write_html(OUTPUT_HTML)
print(f"Saved interactive plot to {OUTPUT_HTML}")
