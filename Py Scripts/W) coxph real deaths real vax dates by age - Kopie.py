import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter
import plotly.graph_objs as go
from pandas.api.types import CategoricalDtype

# Cox Proportional Hazards Analysis of Vaccination Effect by Age Bin

# This script performs survival analysis using a Cox Time-Varying Proportional Hazards model
# to estimate the effect of vaccination on mortality, stratified by age group.

# Workflow:
#  1. Load individual-level czech-FOI (real or simulated) data with vaccination and death dates.
#  2. Preprocess and align time variables to a common reference date.
#  3. Bin individuals into age groups.
#  4. For each age group, reshape the data into a long format suitable for time-varying Cox modeling.
#  5. Estimate hazard ratios (HRs) comparing vaccinated vs. unvaccinated time at risk.
#  6. Plot HRs with 95% confidence intervals across age bins using Plotly.

# Assumes input data includes dates of up to 7 doses and death date per person.


# --- Config ---
REFERENCE_DATE = pd.Timestamp("2020-01-01")  # Reference date for calculating time in days
# INPUT_CSV = r"C:\github\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv" # -> real czech-FOI data 
INPUT_CSV = r"C:\github\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"  # -> simulated data constant homogen death rate to test/check code for bias
OUTPUT_HTML = r"C:\github\CzechFOI-DRATE\Plot Results\W) coxph real deaths real vax dates by age\W) coxph no bias sim deaths sim vax dates by age.html"
start_delay = 0  # Optional delay after vaccination before it takes effect
AGE_BIN_WIDTH = 1  # Age bin width for grouping

# --- Load and preprocess ---
dose_date_cols = [f"Datum_{i}" for i in range(1, 8)]  # Columns for dose dates
df = pd.read_csv(INPUT_CSV, parse_dates=dose_date_cols + ["DatumUmrti"])  # Load data with parsed date columns
df["age"] = 2023 - df["Rok_narozeni"]  # Estimate age from birth year
df = df[df["age"].between(0, 113)].copy()  # Keep only valid age range

# Determine first vaccine dose date for each individual
df["first_dose_date"] = df[dose_date_cols].min(axis=1)

# Calculate time from reference to death/vaccination
df["t_death"] = (df["DatumUmrti"] - REFERENCE_DATE).dt.days
df["t_vacc"] = (df["first_dose_date"] - REFERENCE_DATE).dt.days

# --- Age binning ---
max_age = 114
age_bin_edges = list(range(0, max_age + AGE_BIN_WIDTH, AGE_BIN_WIDTH))
age_bin_labels = [f"{start}-{min(start + AGE_BIN_WIDTH - 1, max_age - 1)}" for start in age_bin_edges[:-1]]
# cat_type = CategoricalDtype(categories=age_bin_labels, ordered=True)

# Bin ages into discrete groups
# df["age_bin"] = pd.cut(df["age"], bins=age_bin_edges, right=False, labels=age_bin_labels).astype(cat_type)
df["age_bin"] = pd.cut(df["age"], bins=age_bin_edges, right=False, labels=age_bin_labels)

# Define max follow-up duration
END_MEASURE = int(df["t_death"].dropna().max())
print(f"Using END_MEASURE = {END_MEASURE}")

# --- Run Cox PH model per age bin ---
hr_results = []

for age_bin in ["15-15","70-70", "71-71", "72-72"]: # age_bin_labels
    sub = df[df["age_bin"] == age_bin].copy()  # Subset data for this age bin
    if len(sub) < 20:
        continue  # Skip small groups

    rows = []
    for idx, row in sub.iterrows():
        pid = idx
        t_vacc = row["t_vacc"]
        t_death = row["t_death"]
        t_end = t_death if not np.isnan(t_death) else END_MEASURE
        t_end = max(t_end, 0.5)  # Avoid zero duration
        event = int(not np.isnan(t_death))  # 1 if death occurred, else 0

        if np.isnan(t_vacc) or t_vacc + start_delay >= t_end:
            # Never vaccinated or vaccinated after death -> unvaccinated follow-up
            rows.append({
                "id": pid,
                "start": 0.0,
                "stop": t_end,
                "event": event,
                "vaccinated": 0
            })
        else:
            t_eff = float(t_vacc) + start_delay
            if t_eff < t_end:
                # Vaccinated during follow-up -> split into unvaccinated and vaccinated periods
                rows.append({
                    "id": pid,
                    "start": 0.0,
                    "stop": t_eff,
                    "event": 0,
                    "vaccinated": 0
                })
                rows.append({
                    "id": pid,
                    "start": t_eff,
                    "stop": t_end,
                    "event": event,
                    "vaccinated": 1
                })
            else:
                # Vaccinated at or after end -> treat as unvaccinated
                rows.append({
                    "id": pid,
                    "start": 0.0,
                    "stop": t_end,
                    "event": event,
                    "vaccinated": 0
                })

    df_long = pd.DataFrame(rows)

    # Lifelines can't handle 0-length intervals with event=1; set minimal duration
    df_long.loc[
        (df_long["start"] == df_long["stop"]) & (df_long["start"] == 0) & (df_long["event"] == 1),
        "stop"
    ] = 0.5

    if len(df_long) < 50:
        continue  # Not enough data to model

    # Skip bins where all deaths are in the same vaccine group (no variance)
    if df_long[df_long["event"] == 1]["vaccinated"].var() < 1e-5:
        print(f"Skipping {age_bin}: no variability in vaccinated events.")
        continue

    try:
        # Fit Cox model using time-varying covariate
        cph = CoxTimeVaryingFitter()
        cph.fit(df_long, id_col="id", start_col="start", stop_col="stop", event_col="event", formula="vaccinated")

        hr = np.exp(cph.params_["vaccinated"])  # Convert log-HR to HR
        ci = cph.confidence_intervals_.loc["vaccinated"]
        ci_lower = np.exp(np.clip(ci.iloc[0], -100, 100))  # Avoid extreme values
        ci_upper = np.exp(np.clip(ci.iloc[1], -100, 100))

        print(f"Age Bin {age_bin}: HR = {hr:.3f} (95% CI: {ci_lower:.3f} – {ci_upper:.3f})")

        # Save results
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
#df_hr["age_bin"] = df_hr["age_bin"].astype(cat_type)
#df_hr.sort_values("age_bin", inplace=True)

# Create Plotly figure with HR and confidence intervals
fig = go.Figure([
    go.Scatter(
        x=df_hr["age_bin"].astype(str),
        y=df_hr["HR"],
        mode="markers+lines",
        name="Hazard Ratio",
        error_y=dict(
            type="data",
            symmetric=False,
            array=df_hr["CI_upper"] - df_hr["HR"],
            arrayminus=df_hr["HR"] - df_hr["CI_lower"]
        )
    )
])

# Add reference line at HR = 1 (no effect)
fig.add_shape(
    type="line",
    xref="paper", yref="y",
    x0=0, x1=1,
    y0=1, y1=1,
    line=dict(color="red", width=0.8, dash="dash")
)

# Configure layout and axis
fig.update_layout(
    title="Cox PH Hazard Ratio by Age Bin (Real Deaths, Real Vaccination Dates)",
    xaxis=dict(
        title="Age Bin",
        tickangle=-45,
        tickmode="array",
        tickvals=df_hr["age_bin"].astype(str),
        ticktext=df_hr["age_bin"].astype(str),
        tickfont=dict(size=10)
    ),
    yaxis=dict(
        title="Hazard Ratio (log scale)",
        type="log"
    ),
    margin=dict(t=80, b=150)
)

# Save figure to HTML
fig.write_html(OUTPUT_HTML)
print(f"Saved plot to: {OUTPUT_HTML}")
