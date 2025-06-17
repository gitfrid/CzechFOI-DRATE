import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pandas.api.types import CategoricalDtype
from scipy.stats import pearsonr

# --- Config ---
REFERENCE_DATE = pd.Timestamp("2020-01-01")
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_SELCTION_BIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
OUTPUT_HTML = r"C:\CzechFOI-DRATE\Plot Results\UA) diff death dose agebin\sim selection bias rolling_corr_doses_vs_diff.html"
start_delay = 0
AGE_BIN_WIDTH = 1
ROLLING_WINDOW = 60  # window for rolling mean smoothing
CORR_WINDOW = 30     # window for rolling correlation calculation

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

# --- Deaths classification ---
df_deaths = df.dropna(subset=["t_death"]).copy()
df_deaths["vaccinated"] = df_deaths[dose_date_cols].le(df_deaths["t_death"] - start_delay, axis=0).any(axis=1).astype(int)
df_deaths["vx_status"] = df_deaths["vaccinated"].map({1: "deaths_vx", 0: "deaths_uvx"})
df_deaths["day"] = df_deaths["t_death"].astype(int)

deaths_agg = df_deaths.groupby(["age_bin", "day", "vx_status"], observed=True).size().unstack(fill_value=0).reset_index()
deaths_agg["deaths_vx"] = deaths_agg.get("deaths_vx", 0)
deaths_agg["deaths_uvx"] = deaths_agg.get("deaths_uvx", 0)
deaths_agg["deaths_diff"] = deaths_agg["deaths_uvx"] - deaths_agg["deaths_vx"]

# --- All Doses Given (all doses) ---
df_doses = df.melt(
    id_vars=["age_bin"],
    value_vars=dose_date_cols,
    var_name="dose_number",
    value_name="dose_day"
)
df_doses = df_doses.dropna(subset=["dose_day"]).copy()
df_doses["day"] = df_doses["dose_day"].astype(int) + start_delay

doses_agg = df_doses.groupby(["age_bin", "day"], observed=True).size().reset_index(name="doses_given")

# --- Merge ---
agg = pd.merge(deaths_agg, doses_agg, how="outer", on=["age_bin", "day"])
agg = agg.fillna({col: 0 for col in agg.select_dtypes(include=["number"]).columns})
agg["age_bin"] = agg["age_bin"].astype(str)

fig = go.Figure()

for age_bin, group in agg.groupby("age_bin"):
    group = group.sort_values("day").reset_index(drop=True)
    
    # Compute rolling means WITHOUT centering to avoid phase shift
    group["deaths_diff_smooth"] = group["deaths_diff"].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    group["doses_given_smooth"] = group["doses_given"].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    
    # Prepare arrays for correlation calculation on rolling means
    deaths_smooth = group["deaths_diff_smooth"].values
    doses_smooth = group["doses_given_smooth"].values
    days = group["day"].values
    
    corr_vals = []
    pvals = []
    corr_days = []
    
    # Compute rolling correlation on smoothed data
    for i in range(len(group) - CORR_WINDOW + 1):
        window_deaths = deaths_smooth[i:i + CORR_WINDOW]
        window_doses = doses_smooth[i:i + CORR_WINDOW]
        
        # Only compute correlation if variance exists
        if np.std(window_deaths) > 0 and np.std(window_doses) > 0:
            r, p = pearsonr(window_doses, window_deaths)
        else:
            r, p = np.nan, np.nan
        
        corr_vals.append(r)
        pvals.append(p)
        # Use center day of correlation window for x-axis
        corr_days.append(days[i + CORR_WINDOW // 2])
    
    # Add rolling correlation trace
    fig.add_trace(go.Scatter(
        x=corr_days,
        y=corr_vals,
        mode="lines",
        name=f"{age_bin} rolling r",
        line=dict(width=0.8),
        hovertemplate=f"Day: %{{x}}<br>r: %{{y:.3f}}<extra>{age_bin}</extra>"
    ))
    
    # Add p-value < 0.05 mask
    sig_mask = (np.array(pvals) < 0.05).astype(int)
    fig.add_trace(go.Scatter(
        x=corr_days,
        y=sig_mask,
        mode='lines',
        name=f"Significant {age_bin} (p<0.05)",
        line=dict(dash='dash', width=1),
        hovertemplate=f"Day: %{{x}}<br>p < 0.05: %{{y}}<extra>{age_bin}</extra>"
    ))
    
    # Add p-values trace
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
    
    # Add p=0.05 threshold line
    fig.add_trace(go.Scatter(
        x=corr_days,
        y=[0.05]*len(corr_days),
        mode='lines',
        name=f'p = 0.05 threshold',
        line=dict(color='red', width=1, dash='dash'),
        showlegend=False
    ))
    
    # Plot rolling mean deaths_diff on yaxis2
    fig.add_trace(go.Scatter(
        x=group["day"],
        y=group["deaths_diff_smooth"],
        mode="lines",
        name=f"{age_bin} deaths_diff (rolling mean)",
        line=dict(width=1, color="darkred"),
        yaxis="y2",
        hovertemplate=f"Day: %{{x}}<br>Deaths Diff (rolling mean): %{{y:.3f}}<extra>{age_bin}</extra>",
        visible='legendonly'
    ))
    
    # Plot rolling mean doses_given on yaxis3
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

# --- Layout ---
fig.update_layout(
    title=f"Rolling Correlation of Smoothed Deaths Diff and Doses Given by Age Bin\n"
          f"(Rolling Mean Window={ROLLING_WINDOW}, Correlation Window={CORR_WINDOW})",
    legend=dict(x=1.1, y=1, xanchor='left', yanchor='top'),
    xaxis_title="Day (since Jan 1, 2020)",
    yaxis=dict(title="Pearson r", side="left", range=[-1.05, 1.05]),
    yaxis2=dict(
        title="Rolling Mean Deaths Diff (uvx-vx)",
        overlaying="y",
        side="right",
        anchor="free",
        position=0.99,
        showgrid=False
    ),
    yaxis3=dict(
        title="Rolling Mean Doses Given",
        overlaying="y",
        side="right",
        anchor="free",
        position=0.98,
        showgrid=False
    ),
    template="plotly_white",
    height=1000,
    width=1800,
)

fig.write_html(OUTPUT_HTML)
print(f"Saved rolling correlation plot: {OUTPUT_HTML}")
