import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

# ===============================================================================
# Title: Dynamic Death Risk Visualization by Age and Vaccination Status (Plotly HTML)
#     This script processes a Czech vaccination and death dataset to analyze
#     age-specific death risks for vaccinated (vx) vs unvaccinated (uvx) groups.
#    
#    The pipeline:
#    - Loads individual-level data with dose dates and death date.
#    - Computes age and dynamic daily vaccination status.
#    - Calculates daily alive population and death counts per age group.
#    - Computes death risk per 100k for total, vx, and uvx groups.
#    - Applies smoothing via rolling average.
#    - Computes mean risk per age from first-dose-onward period.
#    - Produces a Plotly HTML with:
#        1. Smoothed death risk time series (hidden by default).
#        2. Mean death risk bar charts per age group.
#==============================================================================

# === File Paths ===
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv" # -> czech-FOI real data
INPUT_CSV = r"C:\github\CzechFOI-DRATE\TERRA\sim_SELCTION_BIAS_Vesely_106_202403141131.csv" # -> simulated testdata homogen random constant death rates over time per ag  
OUTPUT_HTML = r"C:\github\CzechFOI-DRATE\Plot Results\E) death risk by age\E) sim selection bias vx_uvx_death_risk_by_age.html"

# === Parameters ===
START_DATE = pd.Timestamp("2020-01-01")  # reference day zero
MAX_AGE = 113                            # maximum age to include
REFERENCE_YEAR = 2023                    # used to compute age
ROLLING_WINDOW = 7                       # smoothing window size

# === Load and Clean Data ===
df = pd.read_csv(
    INPUT_CSV,
    parse_dates=[f"Datum_{i}" for i in range(1, 8)] + ["DatumUmrti"],
    dayfirst=False,
    low_memory=False,
)

# Normalize column names
df.columns = [col.strip().lower() for col in df.columns]
dose_cols = [f"datum_{i}" for i in range(1, 8)]

# Compute age
df["birth_year"] = pd.to_numeric(df["rok_narozeni"], errors="coerce")
df["age"] = REFERENCE_YEAR - df["birth_year"]
df = df[df["age"].between(0, MAX_AGE)]

# Days since START_DATE
to_day = lambda s: (s - START_DATE).dt.days
df["t_death"] = to_day(df["datumumrti"])
for col in dose_cols:
    df[col + "_day"] = to_day(df[col])
df["first_dose_day"] = df[[c + "_day" for c in dose_cols]].min(axis=1, skipna=True)

# === End Measure and Start Measure per Age ===
END_MEASURE = int(df["t_death"].dropna().max())
print(f"Using END_MEASURE = {END_MEASURE}")

first_dose_per_age = df.groupby("age")["first_dose_day"].min().clip(lower=0).astype(int)
START_MEASURE = first_dose_per_age.to_dict()
print(f"Using START_MEASURE = {START_MEASURE}")

# Preallocate result arrays
days = np.arange(0, END_MEASURE + 1)
ages = np.arange(0, MAX_AGE + 1)
n_days = len(days)
n_ages = len(ages)

death_risk_total = np.zeros((n_ages, n_days))
death_risk_vx = np.zeros((n_ages, n_days))
death_risk_uvx = np.zeros((n_ages, n_days))

# === Calculate Population and Deaths per Age and Day with DYNAMIC VAX STATUS ===
for i, age in enumerate(ages):
    sub = df[df["age"] == age]
    if sub.empty:
        continue

    dday = sub["t_death"].values
    first_dose = sub["first_dose_day"].values

    # Broadcast matrices
    day_matrix = np.broadcast_to(days[None, :], (len(sub), n_days))
    death_day_matrix = np.broadcast_to(dday[:, None], (len(sub), n_days))
    first_dose_matrix = np.broadcast_to(first_dose[:, None], (len(sub), n_days))

    # Alive mask: before death (or no death)
    alive_mask = np.isnan(dday[:, None]) | (day_matrix < death_day_matrix)

    # Died on the day
    died_mask = day_matrix == death_day_matrix

    # Dynamic vaccination status
    vx_mask = day_matrix >= first_dose_matrix  # vaccinated after first dose
    uvx_mask = ~vx_mask                        # unvaccinated before first dose (or never dosed)

    # Combine with alive/dead status
    pop_vx = np.sum(alive_mask & vx_mask, axis=0)
    pop_uvx = np.sum(alive_mask & uvx_mask, axis=0)
    death_vx = np.sum(died_mask & vx_mask, axis=0)
    death_uvx = np.sum(died_mask & uvx_mask, axis=0)

    total_pop = pop_vx + pop_uvx
    total_deaths = death_vx + death_uvx

    # Death risk per 100,000
    with np.errstate(divide='ignore', invalid='ignore'):
        death_risk_total[i] = np.where(total_pop > 0, total_deaths / total_pop * 1e5, 0)
        death_risk_vx[i] = np.where(pop_vx > 0, death_vx / pop_vx * 1e5, 0)
        death_risk_uvx[i] = np.where(pop_uvx > 0, death_uvx / pop_uvx * 1e5, 0)

# === Flatten and Convert to DataFrame ===
df_stats = pd.DataFrame({
    "age": np.repeat(ages, n_days),
    "day": np.tile(days, n_ages),
    "risk_total": death_risk_total.flatten(),
    "risk_vx": death_risk_vx.flatten(),
    "risk_uvx": death_risk_uvx.flatten()
})

# Apply smoothing (rolling average)
df_stats[["risk_total_s", "risk_vx_s", "risk_uvx_s"]] = (
    df_stats.groupby("age")[["risk_total", "risk_vx", "risk_uvx"]]
    .transform(lambda g: g.rolling(ROLLING_WINDOW, center=True, min_periods=1).mean())
)

# === Compute Mean Risk per Age (from age-specific first dose day onward) ===
mean_records = []
for age in ages:
    start_day = START_MEASURE.get(age, 0)
    subset = df_stats[(df_stats["age"] == age) & (df_stats["day"] >= start_day) & (df_stats["day"] <= END_MEASURE)]
    if subset.empty:
        continue
    mean_records.append({
        "age": age,
        "mean_total": subset["risk_total_s"].mean(),
        "mean_vx": subset["risk_vx_s"].mean(),
        "mean_uvx": subset["risk_uvx_s"].mean()
    })

mean_risk = pd.DataFrame(mean_records)

# === Plotly Visualization ===
fig = sp.make_subplots(
    rows=2, cols=1,
    subplot_titles=("Smoothed Death Risk by Age Over Time", "Mean Death Risk by Age (Post-Vaccination Start)")
)

# Time series per age (hidden by default)
for age in mean_risk["age"]:
    adf = df_stats[df_stats["age"] == age]

    fig.add_trace(
        go.Scatter(x=adf["day"], y=adf["risk_total_s"],
                   mode="lines", name=f"Age {age} • Total",
                   line=dict(color="black", width=1), visible="legendonly"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=adf["day"], y=adf["risk_vx_s"],
                   mode="lines", name=f"Age {age} • Vx",
                   line=dict(color="blue", width=1), visible="legendonly"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=adf["day"], y=adf["risk_uvx_s"],
                   mode="lines", name=f"Age {age} • UVX",
                   line=dict(color="red", width=1), visible="legendonly"),
        row=1, col=1
    )

# Mean bar chart by age
fig.add_trace(
    go.Bar(x=mean_risk["age"], y=mean_risk["mean_total"],
           name="Mean Total", marker_color="black"),
    row=2, col=1
)
fig.add_trace(
    go.Bar(x=mean_risk["age"], y=mean_risk["mean_vx"],
           name="Mean Vx", marker_color="blue"),
    row=2, col=1
)
fig.add_trace(
    go.Bar(x=mean_risk["age"], y=mean_risk["mean_uvx"],
           name="Mean UVX", marker_color="red"),
    row=2, col=1
)

# Final layout
fig.update_layout(
    height=1000, width=900,
    title_text="Death Risk Trends and Age-specific Averages (Post-Vax Start, Dynamic Vx Status)",
    xaxis=dict(title="Day Since 2020‑01‑01"),
    xaxis2=dict(title="Age"),
    yaxis=dict(title="Death Risk per 100k/day"),
    yaxis2=dict(title="Mean Risk per 100k/day"),
    template="plotly_white",
    showlegend=True
)

# Save as interactive HTML
fig.write_html(OUTPUT_HTML)
print(f"Saved interactive Plotly HTML: {OUTPUT_HTML}")
