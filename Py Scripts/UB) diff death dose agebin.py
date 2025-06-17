import numpy as np
import pandas as pd
from dowhy import CausalModel
import plotly.graph_objects as go
from pandas.api.types import CategoricalDtype

# --- Config ---
REFERENCE_DATE = pd.Timestamp("2020-01-01")
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_SELCTION_BIAS_Vesely_106_202403141131.csv"
OUTPUT_HTML_SCATTER = r"C:\CzechFOI-DRATE\Plot Results\UB) diff death dose agebin\sim selection bias dowhy_scatter_doses_vs_diff.html"
OUTPUT_HTML_TIMESERIES = r"C:\CzechFOI-DRATE\Plot Results\UB) diff death dose agebin\sim selection bias dowhy_timeseries_causal_per_agebin.html"
start_delay = 14
AGE_BIN_WIDTH = 1

# --- Load and preprocess ---
dose_date_cols = [f"Datum_{i}" for i in range(1, 8)]
df = pd.read_csv(INPUT_CSV, parse_dates=dose_date_cols + ["DatumUmrti"], low_memory=False)

# Age calculation and filtering
df["age"] = 2023 - df["Rok_narozeni"]
df = df[df["age"].between(0, 113)].copy()

# Days since REFERENCE_DATE
for col in dose_date_cols:
    df[col] = (df[col] - REFERENCE_DATE).dt.days
df["t_death"] = (df["DatumUmrti"] - REFERENCE_DATE).dt.days

# --- Age bins ---
max_age = 114
age_bin_edges = list(range(0, max_age + AGE_BIN_WIDTH, AGE_BIN_WIDTH))
age_bin_labels = [f"{start}-{min(start + AGE_BIN_WIDTH - 1, max_age - 1)}" for start in age_bin_edges[:-1]]
cat_type = CategoricalDtype(categories=age_bin_labels, ordered=True)
df["age_bin"] = pd.cut(df["age"], bins=age_bin_edges, right=False, labels=age_bin_labels).astype(cat_type)

# --- Deaths classification ---
df_deaths = df.dropna(subset=["t_death"]).copy()
df_deaths["vaccinated"] = df_deaths[dose_date_cols].le(df_deaths["t_death"] - start_delay, axis=0).any(axis=1).astype(int)
df_deaths["vx_status"] = df_deaths["vaccinated"].map({1: "deaths_vx", 0: "deaths_uvx"})
df_deaths["day"] = df_deaths["t_death"].astype(int)

deaths_agg = df_deaths.groupby(["age_bin", "day", "vx_status"]).size().unstack(fill_value=0).reset_index()
deaths_agg["deaths_vx"] = deaths_agg.get("deaths_vx", 0)
deaths_agg["deaths_uvx"] = deaths_agg.get("deaths_uvx", 0)
deaths_agg["deaths_diff"] = deaths_agg["deaths_uvx"] - deaths_agg["deaths_vx"]

# --- Vaccination per day ---
df["first_effective_vx_day"] = df[dose_date_cols].min(axis=1) + start_delay
df_vacc = df.dropna(subset=["first_effective_vx_day"]).copy()
df_vacc["day"] = df_vacc["first_effective_vx_day"].astype(int)

doses_agg = df_vacc.groupby(["age_bin", "day"]).size().reset_index(name="doses_given")

# --- Merge ---
agg = pd.merge(deaths_agg, doses_agg, how="outer", on=["age_bin", "day"])
agg = agg.fillna({col: 0 for col in agg.select_dtypes(include=["number"]).columns})

agg["day_numeric"] = agg["day"]
agg["age_bin"] = agg["age_bin"].astype(str)
agg["month"] = agg["day"] // 30

# --- Causal effect per month per age bin ---
results = []
for (age_bin, month), group in agg.groupby(["age_bin", "month"]):
    if group["doses_given"].sum() == 0 or group["deaths_diff"].sum() == 0:
        continue

    try:
        model = CausalModel(
            data=group,
            treatment="doses_given",
            outcome="deaths_diff",
            common_causes=[],
            treatment_type="continuous",
            outcome_type="continuous"
        )
        # common_causes=["day_numeric"],
        estimand = model.identify_effect()
        estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
        results.append({"age_bin": age_bin, "month": month, "effect": estimate.value})
    except Exception as e:
        print(f"Error for {age_bin}, month {month}: {e}")

result_df = pd.DataFrame(results)

# --- Plot: Line plot of causal effect per month for each age bin, x-axis in days ---
fig2 = go.Figure()

for age_bin, group in result_df.groupby("age_bin"):
    fig2.add_trace(go.Scatter(
        x=group["month"] * 30,  # show days on x-axis, approx month start day
        y=group["effect"],
        mode="lines+markers",
        name=f"Age {age_bin}",
        hovertemplate="Day: %{x}<br>Effect: %{y:.5f}<extra></extra>"
    ))

fig2.update_layout(
    title="DoWhy Causal Effect per Month per Age Bin (X-axis in Days)",
    xaxis_title="Day (approximate month start since Jan 1, 2020)",
    yaxis_title="Estimated Causal Effect (Δ Deaths per Dose)",
    template="plotly_white",
    height=700,
    width=1200
)

fig2.write_html(OUTPUT_HTML_TIMESERIES)
print(f"Saved time-series causal effect plot: {OUTPUT_HTML_TIMESERIES}")
