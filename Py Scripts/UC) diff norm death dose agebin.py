import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api.types import CategoricalDtype
from scipy.stats import pearsonr

# --- Config ---
REFERENCE_DATE = pd.Timestamp("2020-01-01")
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
OUTPUT_HTML = r"C:\CzechFOI-DRATE\Plot Results\UC) diff norm death dose agebin\norm rolling_corr_doses_vs_diff.html"
start_delay = 0
AGE_BIN_WIDTH = 1
ROLLING_WINDOW = 60
CORR_WINDOW = 30

# --- Load and preprocess ---
dose_date_cols = [f"Datum_{i}" for i in range(1, 8)]
df = pd.read_csv(INPUT_CSV, parse_dates=dose_date_cols + ["DatumUmrti"], low_memory=False)

df["age"] = 2023 - df["Rok_narozeni"]
df = df[df["age"].between(0, 113)].copy()

for col in dose_date_cols:
    df[col] = (df[col] - REFERENCE_DATE).dt.days
df["t_death"] = (df["DatumUmrti"] - REFERENCE_DATE).dt.days

# --- Age bins ---
max_age = 114
age_bin_edges = list(range(0, max_age + AGE_BIN_WIDTH, AGE_BIN_WIDTH))
age_bin_labels = [f"{start}-{min(start + AGE_BIN_WIDTH - 1, max_age - 1)}" for start in age_bin_edges[:-1]]
cat_type = CategoricalDtype(categories=age_bin_labels, ordered=True)
df["age_bin"] = pd.cut(df["age"], bins=age_bin_edges, right=False, labels=age_bin_labels).astype(cat_type)

# --- Death classification ---
df_deaths = df.dropna(subset=["t_death"]).copy()
df_deaths["vaccinated"] = df_deaths[dose_date_cols].le(df_deaths["t_death"] - start_delay, axis=0).any(axis=1).astype(int)
df_deaths["vx_status"] = df_deaths["vaccinated"].map({1: "deaths_vx", 0: "deaths_uvx"})
df_deaths["day"] = df_deaths["t_death"].astype(int)

deaths_agg = df_deaths.groupby(["age_bin", "day", "vx_status"], observed=True).size().unstack(fill_value=0).reset_index()
deaths_agg["deaths_vx"] = deaths_agg.get("deaths_vx", 0)
deaths_agg["deaths_uvx"] = deaths_agg.get("deaths_uvx", 0)

# --- Vaccinated population over time ---
df["first_vax_day"] = df[dose_date_cols].min(axis=1)
df["first_vax_day"] = df["first_vax_day"].fillna(1e9).astype(int)
df["death_day"] = df["t_death"].fillna(1e9).astype(int)

# Mark vaccination status over time for surviving people
days = np.arange(0, int(df[["first_vax_day", "death_day"]].replace(1e9, np.nan).max().max()) + 1)
pop_data = []

for age_bin, group in df.groupby("age_bin", observed=True):
    vx_start = group["first_vax_day"].values
    death_day = group["death_day"].values
    for day in days:
        alive = (day < death_day)
        pop_vx = ((vx_start <= day) & alive).sum()
        pop_uvx = ((vx_start > day) & alive).sum()
        pop_data.append((age_bin, day, pop_vx, pop_uvx))

pop_df = pd.DataFrame(pop_data, columns=["age_bin", "day", "pop_vx", "pop_uvx"])

# --- Merge deaths + pop ---
agg = pd.merge(deaths_agg, pop_df, on=["age_bin", "day"], how="outer")
agg = agg.fillna({col: 0 for col in agg.select_dtypes(include=["number"]).columns})

# Avoid divide-by-zero
agg["deaths_vx_norm"] = np.where(agg["pop_vx"] > 0, agg["deaths_vx"] / agg["pop_vx"] * 100_000, np.nan)
agg["deaths_uvx_norm"] = np.where(agg["pop_uvx"] > 0, agg["deaths_uvx"] / agg["pop_uvx"] * 100_000, np.nan)
agg["deaths_diff_norm"] = agg["deaths_uvx_norm"] - agg["deaths_vx_norm"]

# --- Doses ---
df_doses = df.melt(
    id_vars=["age_bin"],
    value_vars=dose_date_cols,
    var_name="dose_number",
    value_name="dose_day"
).dropna()
df_doses["day"] = df_doses["dose_day"].astype(int) + start_delay
doses_agg = df_doses.groupby(["age_bin", "day"], observed=True).size().reset_index(name="doses_given")

# --- Merge doses ---
agg = pd.merge(agg, doses_agg, on=["age_bin", "day"], how="outer")
agg = agg.fillna({col: 0 for col in agg.select_dtypes(include=["number"]).columns})
agg["age_bin"] = agg["age_bin"].astype(str)

# --- Plotting ---
fig = go.Figure()

for age_bin, group in agg.groupby("age_bin"):
    group = group.sort_values("day").reset_index(drop=True)
    
    group["deaths_diff_norm_smooth"] = group["deaths_diff_norm"].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    group["doses_given_smooth"] = group["doses_given"].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    
    deaths_smooth = group["deaths_diff_norm_smooth"].values
    doses_smooth = group["doses_given_smooth"].values
    days = group["day"].values

    corr_vals = []
    pvals = []
    corr_days = []

    for i in range(len(group) - CORR_WINDOW + 1):
        window_deaths = deaths_smooth[i:i + CORR_WINDOW]
        window_doses = doses_smooth[i:i + CORR_WINDOW]
        if np.std(window_deaths) > 0 and np.std(window_doses) > 0:
            r, p = pearsonr(window_doses, window_deaths)
        else:
            r, p = np.nan, np.nan
        corr_vals.append(r)
        pvals.append(p)
        corr_days.append(days[i + CORR_WINDOW // 2])

    fig.add_trace(go.Scatter(
        x=corr_days,
        y=corr_vals,
        mode="lines",
        name=f"{age_bin} rolling r",
        line=dict(width=0.8),
        hovertemplate=f"Day: %{{x}}<br>r: %{{y:.3f}}<extra>{age_bin}</extra>"
    ))
    fig.add_trace(go.Scatter(
        x=corr_days,
        y=(np.array(pvals) < 0.05).astype(int),
        mode='lines',
        name=f"Significant {age_bin} (p<0.05)",
        line=dict(dash='dash', width=1),
        hovertemplate=f"Day: %{{x}}<br>p < 0.05: %{{y}}<extra>{age_bin}</extra>"
    ))
    fig.add_trace(go.Scatter(
        x=corr_days,
        y=pvals,
        mode='lines+markers',
        name=f"P-Values {age_bin}",
        line=dict(dash='dot', width=1, color='gray'),
        text=[f"p = {p:.4f}" if not np.isnan(p) else "p = NaN" for p in pvals],
        hoverinfo='text',
        hovertemplate="Day: %{x}<br>%{text}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=corr_days,
        y=[0.05]*len(corr_days),
        mode='lines',
        name=f'p = 0.05 threshold',
        line=dict(color='red', width=1, dash='dash'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=group["day"],
        y=group["deaths_diff_norm_smooth"],
        mode="lines",
        name=f"{age_bin} deaths_diff_norm per 100k (rolling mean)",
        line=dict(width=1, color="darkred"),
        yaxis="y2",
        hovertemplate=f"Day: %{{x}}<br>Deaths Diff Norm: %{{y:.3f}}<extra>{age_bin}</extra>",
        visible='legendonly'
    ))
    fig.add_trace(go.Scatter(
        x=group["day"],
        y=group["doses_given_smooth"],
        mode="lines",
        name=f"{age_bin} doses_given (rolling mean)",
        line=dict(width=1, color="darkblue"),
        yaxis="y3",
        hovertemplate=f"Day: %{{x}}<br>Doses Given (rolling mean): %{{y:.3f}}<extra>{age_bin}</extra>",
        visible='legendonly'
    ))

fig.update_layout(
    title=f"Rolling Correlation of Smoothed normalized Deaths Diff -> (deaths_uvx - deaths_vx) per 100k live vx and uvx pop per age and day",
    legend=dict(x=1.1, y=1, xanchor='left', yanchor='top'),
    xaxis_title="Day (since Jan 1, 2020)",
    yaxis=dict(title="Pearson r", side="left", range=[-1.05, 1.05]),
    yaxis2=dict(
        title="Rolling Mean Deaths Diff Norm (per 100k)",
        overlaying="y",
        side="right",
        anchor="free",
        position=0.96,
        showgrid=False
    ),
    yaxis3=dict(
        title="Rolling Mean Doses Given",
        overlaying="y",
        side="right",
        anchor="free",
        position=1,
        showgrid=False
    ),
    template="plotly_white",
    height=1000,
    width=1800,
)

fig.write_html(OUTPUT_HTML)
print(f"Saved rolling correlation plot: {OUTPUT_HTML}")
