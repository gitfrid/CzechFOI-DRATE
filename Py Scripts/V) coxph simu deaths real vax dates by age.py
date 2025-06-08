import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
import plotly.graph_objs as go

# -----------------------------------------------------------------------------
# Script: Simulated Cox PH Hazard Ratio by Age Bin using Real Vaccination Dates
#
# This script loads a real-world dataset of deaths and vaccination dates,
# then for each age bin:
#   - Estimates person-time and real death hazards
#   - Simulates individual-level death times using an exponential distribution
#   - Randomly assigns vaccination status using real vaccine timing data
#   - Transforms the data into a time-varying format suitable for Cox modeling
#   - Fits a Cox proportional hazards model per age bin
#   - Plots hazard ratios and 95% confidence intervals by age
#
# NOTE: All real deaths and real vaccine dates are used only for timing
#       — no real outcomes are used in fitting (simulation only).
# -----------------------------------------------------------------------------

# --- Config ---
REFERENCE_DATE = pd.Timestamp("2020-01-01")
INPUT_CSV = r"C:\github\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
OUTPUT_HTML = r"C:\github\CzechFOI-DRATE\Plot Results\V) coxph simu deaths real vax dates by age\V) coxph simu deaths real vax dates by age.html"
start_delay = 0  # optional lag after vaccination before effect begins
AGE_BIN_WIDTH = 1

# --- Load and preprocess ---
dose_date_cols = [f"Datum_{i}" for i in range(1, 8)]  # columns with dose dates
df = pd.read_csv(INPUT_CSV, parse_dates=dose_date_cols + ["DatumUmrti"])

df["age"] = 2023 - df["Rok_narozeni"]
df = df[df["age"].between(0, 113)].copy()  # restrict to age 0–113
df["first_dose_date"] = df[dose_date_cols].min(axis=1)  # earliest dose date per person

# Calculate time (days) from reference for death and vaccination
df["t_death"] = (df["DatumUmrti"] - REFERENCE_DATE).dt.days
df["t_vacc"] = (df["first_dose_date"] - REFERENCE_DATE).dt.days

# --- Age binning ---
max_age = 114
age_bin_edges = list(range(0, max_age + AGE_BIN_WIDTH, AGE_BIN_WIDTH))
age_bin_labels = [f"{start}-{min(start + AGE_BIN_WIDTH - 1, max_age - 1)}" for start in age_bin_edges[:-1]]
df["age_bin"] = pd.cut(df["age"], bins=age_bin_edges, right=False, labels=age_bin_labels)

# --- Define END_MEASURE (latest observable death time) ---
END_MEASURE = int(df["t_death"].dropna().max())
print(f"Using END_MEASURE = {END_MEASURE}")

# --- Estimate real hazard rates per age bin ---
age_stats = []
for age_bin in age_bin_labels:
    sub = df[df["age_bin"] == age_bin]
    if len(sub) < 20:
        continue

    # Compute total person-time and number of deaths
    person_time = 0
    deaths = 0
    for row in sub.itertuples():
        t_end = row.t_death if pd.notna(row.t_death) else END_MEASURE
        person_time += max(t_end, 0.5)  # avoid zero time
        deaths += int(pd.notna(row.t_death))

    if person_time == 0:
        continue

    hazard = deaths / person_time
    age_stats.append({"age_bin": age_bin, "hazard": hazard, "N": len(sub)})

stats_df = pd.DataFrame(age_stats)

# --- Simulation ---
hr_results = []
np.random.seed(42)

for row in stats_df.itertuples():
    age_bin = row.age_bin
    hazard = row.hazard
    N = row.N

    real_age_group = df[df["age_bin"] == age_bin]
    t_vacc_pool = real_age_group["t_vacc"].dropna()  # use real vax dates

    if len(t_vacc_pool) < 10:
        print(f"Skipping {age_bin}: too few real vaccinated individuals.")
        continue

    # Randomly assign vaccination status
    has_vacc = np.random.rand(N) < 0.5
    n_vacc = has_vacc.sum()
    t_vacc = np.full(N, np.nan)

    # Sample real vax dates for simulated vaccinated people
    if n_vacc > len(t_vacc_pool):
        print(f"Skipping {age_bin}: insufficient real t_vacc samples for {n_vacc} simulated vaccinated.")
        continue

    t_vacc[has_vacc] = t_vacc_pool.sample(n=n_vacc, replace=False).to_numpy()

    # Simulate death times from exponential distribution
    death_times = np.random.exponential(scale=1 / hazard, size=N)
    death_times = np.minimum(death_times, END_MEASURE)  # truncate at END_MEASURE
    observed = death_times < END_MEASURE

    # --- Erzeuge long-Format für Cox mit zeitabhängiger Impfung ---
    rows = []
    for i in range(N):
        pid = i
        dtime = float(death_times[i])
        event = int(observed[i])
        tv = t_vacc[i]

        if np.isnan(tv) or tv + start_delay >= dtime:
            # Unvaccinated or vaccinated after death: entire follow-up unvaccinated
            rows.append({
                "id": pid,
                "start": 0.0,
                "stop": dtime if dtime > 0 else 0.5,
                "event": event,
                "vaccinated": 0,
            })
        else:
            # Vaccinated during follow-up, split into two periods
            t_eff = float(tv) + start_delay
            if t_eff < dtime:
                rows.append({
                    "id": pid,
                    "start": 0.0,
                    "stop": t_eff,
                    "event": 0,
                    "vaccinated": 0,
                })
                rows.append({
                    "id": pid,
                    "start": t_eff,
                    "stop": dtime if dtime > t_eff else t_eff + 0.5,
                    "event": event,
                    "vaccinated": 1,
                })
            else:
                rows.append({
                    "id": pid,
                    "start": 0.0,
                    "stop": dtime if dtime > 0 else 0.5,
                    "event": event,
                    "vaccinated": 0,
                })

    df_long = pd.DataFrame(rows)

    # Fix any 0-duration windows with events (to avoid fitting errors)
    df_long.loc[
        (df_long["start"] == df_long["stop"]) & (df_long["start"] == 0) & (df_long["event"] == 1),
        "stop"
    ] = 0.5

    if len(df_long) < 50:
        continue

    # Require some variance in vaccinated status among deaths
    if df_long[df_long["event"] == 1]["vaccinated"].var() < 1e-5:
        print(f"Skipping {age_bin}: no variability in vaccinated event occurrences.")
        continue

    print(f"\n--- Age Bin {age_bin} ---")
    print(df_long.groupby("vaccinated")["event"].value_counts().unstack(fill_value=0))

    try:
        # Fit Cox time-varying model
        cph = CoxTimeVaryingFitter()
        cph.fit(df_long, id_col="id", start_col="start", stop_col="stop", event_col="event", formula="vaccinated")
        hr = np.exp(cph.params_["vaccinated"])  # hazard ratio
        ci = cph.confidence_intervals_.loc["vaccinated"]
        ci_lower = np.exp(np.clip(ci.iloc[0], -100, 100))
        ci_upper = np.exp(np.clip(ci.iloc[1], -100, 100))

        print(f"Age Bin {age_bin}: HR = {hr:.3f} (95% CI: {ci_lower:.3f} – {ci_upper:.3f})")
        hr_results.append({
            "age_bin": age_bin,
            "HR": hr,
            "CI_lower": ci_lower,
            "CI_upper": ci_upper
        })
    except Exception as e:
        print(f"Skipping {age_bin}: {e}")

# --- Plotting ---
if not hr_results:
    print("No results to plot.")
    exit()

df_hr = pd.DataFrame(hr_results)

fig = go.Figure([
    go.Scatter(
        x=df_hr["age_bin"],
        y=df_hr["HR"],
        mode="markers+lines",
        name="HR",
        error_y=dict(
            type="data",
            symmetric=False,
            array=df_hr["CI_upper"] - df_hr["HR"],
            arrayminus=df_hr["HR"] - df_hr["CI_lower"]
        )
    )
])

# Add horizontal reference line at HR = 1
fig.add_shape(
    type="line",
    xref="paper", yref="y",
    x0=0, x1=1,
    y0=1, y1=1,
    line=dict(color="red", width=0.8, dash="dash"),
)

fig.update_layout(
    title=f"Simulated Cox PH per {AGE_BIN_WIDTH}-Year Age Bin (real vax dates, simulated deaths, real person-time)",
    xaxis_title="Age Bin",
    yaxis_title="Hazard Ratio (log scale)",
    yaxis_type="log"
)

fig.write_html(OUTPUT_HTML)
print(f"Saved plot to: {OUTPUT_HTML}")
