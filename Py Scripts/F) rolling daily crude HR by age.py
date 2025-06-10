import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp

# Rolling Daily Crude Hazard Rate Visualization by Age
# ----------------------------------------------------

# This script calculates and visualizes the rolling crude hazard rates (HR) of vaccinated (Vx)
# and unvaccinated (UVX) individuals using Czech vaccination/death data. It:
#  - Loads individual-level CSV data with birth, death, and vaccination dates.
#  - Computes age, day-of-event (death and doses) relative to Jan 1, 2020.
#  - Classifies individuals dynamically into Vx and UVX groups over time.
#  - Calculates population-at-risk and deaths per group per age/day.
#  - Computes crude HRs using a centered rolling window (default ±30 days).
#  - Smooths HRs for visual clarity and plots:
#     - Time-series of smoothed and raw daily crude HR by age.
#     - Mean crude HR per age (after vaccination start).

# Output:
#    1. Rolling HR traces for each age.
#    2. Mean crude HRs by age group post-vaccination.


# === File Paths ===
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_HR_NOBIAS_Vesely_106_202403141131.csv" # -> simulated homogenized constant death rate
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"  # -> Czech FOI real data
OUTPUT_HTML = r"C:\github\CzechFOI-DRATE\Plot Results\F) rolling daily crude HR by age\F) rolling daily crude HR by age.html"

# === Parameters ===
START_DATE = pd.Timestamp("2020-01-01")  # reference day zero for time axis
MAX_AGE = 113                            # maximum age included in analysis
REFERENCE_YEAR = 2023                    # used to calculate age from birth year
ROLLING_WINDOW = 61                      # rolling window size (±30 days)

# === Load and Clean Data ===
df = pd.read_csv(
    INPUT_CSV,
    parse_dates=[f"Datum_{i}" for i in range(1, 8)] + ["DatumUmrti"],
    dayfirst=False,
    low_memory=False,
)

# Standardize column names
df.columns = [col.strip().lower() for col in df.columns]
dose_cols = [f"datum_{i}" for i in range(1, 8)]

# Compute age from birth year
df["birth_year"] = pd.to_numeric(df["rok_narozeni"], errors="coerce")
df["age"] = REFERENCE_YEAR - df["birth_year"]
df = df[df["age"].between(0, MAX_AGE)]

# Convert dates to days since START_DATE
to_day = lambda s: (s - START_DATE).dt.days
df["t_death"] = to_day(df["datumumrti"])
for col in dose_cols:
    df[col + "_day"] = to_day(df[col])
df["first_dose_day"] = df[[c + "_day" for c in dose_cols]].min(axis=1, skipna=True)

# Determine final measurement day and first dose start per age
END_MEASURE = int(df["t_death"].dropna().max())
print(f"Using END_MEASURE = {END_MEASURE}")

first_dose_per_age = df.groupby("age")["first_dose_day"].min().clip(lower=0).astype(int)
START_MEASURE = first_dose_per_age.to_dict()
print(f"Using START_MEASURE = {START_MEASURE}")

# Define grid for age and day
days = np.arange(0, END_MEASURE + 1)
ages = np.arange(0, MAX_AGE + 1)
n_days = len(days)
n_ages = len(ages)

# Preallocate arrays for population and deaths
pop_vx = np.zeros((n_ages, n_days), dtype=float)
pop_uvx = np.zeros((n_ages, n_days), dtype=float)
death_vx = np.zeros((n_ages, n_days), dtype=float)
death_uvx = np.zeros((n_ages, n_days), dtype=float)

# === Calculate Population and Deaths per Age and Day with DYNAMIC VAX STATUS ===
for i, age in enumerate(ages):
    sub = df[df["age"] == age]
    if sub.empty:
        continue

    # Extract relevant columns
    dday = sub["t_death"].values
    first_dose = sub["first_dose_day"].values

    # Create broadcast matrices for vectorized comparison
    day_matrix = np.broadcast_to(days[None, :], (len(sub), n_days))
    death_day_matrix = np.broadcast_to(dday[:, None], (len(sub), n_days))
    first_dose_matrix = np.broadcast_to(first_dose[:, None], (len(sub), n_days))

    # Alive if no death yet or died after current day
    alive_mask = np.isnan(dday[:, None]) | (day_matrix < death_day_matrix)
    # Died exactly on that day
    died_mask = day_matrix == death_day_matrix

    # Vaccination status mask
    vx_mask = day_matrix >= first_dose_matrix
    uvx_mask = ~vx_mask

    # Count living population and deaths per group
    pop_vx[i] = np.sum(alive_mask & vx_mask, axis=0)
    pop_uvx[i] = np.sum(alive_mask & uvx_mask, axis=0)
    death_vx[i] = np.sum(died_mask & vx_mask, axis=0)
    death_uvx[i] = np.sum(died_mask & uvx_mask, axis=0)

# === Compute crude Hazard Ratio per day using rolling windows ±30 days ===
half_window = ROLLING_WINDOW // 2

crude_hr_total = np.zeros((n_ages, n_days))
crude_hr_vx = np.zeros((n_ages, n_days))
crude_hr_uvx = np.zeros((n_ages, n_days))

for i in range(n_ages):
    for day_idx in range(n_days):
        start_idx = max(0, day_idx - half_window)
        end_idx = min(n_days, day_idx + half_window + 1)  # end exclusive

        # Sum deaths and person-days within rolling window
        d_vx = death_vx[i, start_idx:end_idx].sum()
        d_uvx = death_uvx[i, start_idx:end_idx].sum()
        p_vx = pop_vx[i, start_idx:end_idx].sum()
        p_uvx = pop_uvx[i, start_idx:end_idx].sum()
        p_total = p_vx + p_uvx
        d_total = d_vx + d_uvx

        # Compute hazard rates
        hr_vx = d_vx / p_vx if p_vx > 0 else np.nan
        hr_uvx = d_uvx / p_uvx if p_uvx > 0 else np.nan
        hr_total = d_total / p_total if p_total > 0 else np.nan

        # Compute crude HR: vaccinated HR / unvaccinated HR
        if hr_uvx and hr_uvx > 0 and hr_vx is not np.nan:
            crude_hr = hr_vx / hr_uvx
        else:
            crude_hr = np.nan

        # Store results
        crude_hr_total[i, day_idx] = hr_total
        crude_hr_vx[i, day_idx] = hr_vx
        crude_hr_uvx[i, day_idx] = hr_uvx

# === Flatten into DataFrame for Plotting ===
df_crude = pd.DataFrame({
    "age": np.repeat(ages, n_days),
    "day": np.tile(days, n_ages),
    "crude_hr_total": crude_hr_total.flatten(),
    "crude_hr_vx": crude_hr_vx.flatten(),
    "crude_hr_uvx": crude_hr_uvx.flatten()
})

# === Smooth crude HR using rolling average (for plot clarity) ===
df_crude[["crude_hr_total_s", "crude_hr_vx_s", "crude_hr_uvx_s"]] = (
    df_crude.groupby("age")[["crude_hr_total", "crude_hr_vx", "crude_hr_uvx"]]
    .transform(lambda g: g.rolling(ROLLING_WINDOW, center=True, min_periods=1).mean())
)

# === Compute mean HR after first-dose start per age ===
mean_records = []
for age in ages:
    start_day = START_MEASURE.get(age, 0)
    subset = df_crude[(df_crude["age"] == age) & (df_crude["day"] >= start_day) & (df_crude["day"] <= END_MEASURE)]
    if subset.empty:
        continue
    mean_records.append({
        "age": age,
        "mean_total": subset["crude_hr_total_s"].mean(),
        "mean_vx": subset["crude_hr_vx_s"].mean(),
        "mean_uvx": subset["crude_hr_uvx_s"].mean()
    })

mean_crude_hr = pd.DataFrame(mean_records)

# === Plotly Visualization ===
fig = sp.make_subplots(
    rows=2, cols=1,
    subplot_titles=("Smoothed crude Hazard Rates by Age Over Time", "Mean crude Hazard Rates by Age (Post-Vaccination Start)"),
    specs=[[{"secondary_y": True}], [{}]]
)

# === Line traces for smoothed daily HRs ===
for age in mean_crude_hr["age"]:
    adf = df_crude[df_crude["age"] == age].copy()

    # Compute relative HR (Vx / UVX) daily (non-smoothed)
    adf["rel_hr_vx_uvx"] = np.where(
        (adf["crude_hr_uvx"] > 0) & (~np.isnan(adf["crude_hr_vx"])),
        adf["crude_hr_vx"] / adf["crude_hr_uvx"],
        np.nan
    )

    # Keep full range from day 0 (no trimming here)
    adf = adf[(adf["day"] >= 0) & (adf["day"] <= END_MEASURE)]

    fig.add_trace(
        go.Scatter(x=adf["day"], y=adf["rel_hr_vx_uvx"],
                   mode="lines", name=f"Age {age} • Daily Vx/UVX HR",
                   line=dict(color="green", width=1, dash="dot"), visible="legendonly"),
        row=1, col=1, secondary_y=True
    )

    fig.add_trace(
        go.Scatter(x=adf["day"], y=adf["crude_hr_total_s"],
                   mode="lines", name=f"Age {age} • Total HR (Smoothed)",
                   line=dict(color="black", width=1), visible="legendonly"),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=adf["day"], y=adf["crude_hr_vx_s"],
                   mode="lines", name=f"Age {age} • Vx HR (Smoothed)",
                   line=dict(color="blue", width=1), visible="legendonly"),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=adf["day"], y=adf["crude_hr_uvx_s"],
                   mode="lines", name=f"Age {age} • UVX HR (Smoothed)",
                   line=dict(color="red", width=1), visible="legendonly"),
        row=1, col=1, secondary_y=False
    )

# === Bar traces for mean HRs post-vax start ===
fig.add_trace(
    go.Bar(x=mean_crude_hr["age"], y=mean_crude_hr["mean_total"],
           name="Mean Total HR", marker_color="black"),
    row=2, col=1
)
fig.add_trace(
    go.Bar(x=mean_crude_hr["age"], y=mean_crude_hr["mean_vx"],
           name="Mean Vx HR", marker_color="blue"),
    row=2, col=1
)
fig.add_trace(
    go.Bar(x=mean_crude_hr["age"], y=mean_crude_hr["mean_uvx"],
           name="Mean UVX HR", marker_color="red"),
    row=2, col=1
)

# === Update layout ===
fig.update_layout(
    title="Rolling Daily Crude Hazard Rate (HR) by Age",
    height=900,
    width=1200,
    legend=dict(
        orientation="v",
        x=1.02,
        y=1,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1
    ),
    margin=dict(l=80, r=200, t=80, b=60)
)
fig.update_xaxes(title_text="Days since 2020-01-01", row=1, col=1)
fig.update_xaxes(title_text="Age", row=2, col=1)
fig.update_yaxes(title_text="Smoothed Crude HR", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="Daily Vx/UVX HR", row=1, col=1, secondary_y=True)
fig.update_yaxes(title_text="Mean Crude HR", row=2, col=1)

# === Save output ===
fig.write_html(OUTPUT_HTML)
print(f"Output saved to: {OUTPUT_HTML}")
