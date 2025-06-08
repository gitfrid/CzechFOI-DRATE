import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.subplots as sp
from lifelines import KaplanMeierFitter, CoxPHFitter


# Bias Study: Static vs. Time-Dependent Grouping in Survival Analysis

# This script simulates death and vaccination events in a synthetic population to demonstrate
# the bias introduced by static classification of vaccination status (immortal time bias).
# It computes and compares death rates, Kaplan-Meier survival curves, and Cox model predictions
# for vaccinated (VX) and unvaccinated (UVX) groups using both static and time-dependent grouping.

# Generates an interactive Plotly HTML visualization showing:
# - Death rates (static and time-dependent)
# - Kaplan-Meier survival and hazard curves
# - Cox model predicted survival and hazard rates

# Simulation parameters
np.random.seed(0)
N = 400_000  # number of individuals
MAX_DAYS = 1095  # observation period (3 years)
DEATH_PROB_PER_DAY = 0.001  # constant daily death probability
OUTPUT_HTML = r"C:\github\CzechFOI-DRATE\Plot Results\J) Bias study ratio vx uvx\J) Bias study ratio over time  TimeDependend.html"

# Simulate death day for each individual (or alive = NaN)
death_day = np.where(np.random.rand(N) < DEATH_PROB_PER_DAY * MAX_DAYS,
                     np.random.randint(0, MAX_DAYS, N),
                     np.nan)

# Vaccination rollout: people get vaccinated on a specific day between 30 and 60
vaccinated_day = np.random.randint(30, 60, size=N)
got_vaccine = np.random.rand(N) < 0.5  # 50% vaccinated

# Set vaccinated_day to NaN for unvaccinated individuals
vaccinated_day = np.where(got_vaccine, vaccinated_day, np.nan)

# Create main DataFrame
df = pd.DataFrame({
    "id": np.arange(N),
    "death_day": death_day,
    "vaccinated_day": vaccinated_day
})

# -------- Option A: Static classification (immortal time bias) --------
df["static_group"] = np.where(df["vaccinated_day"].notna(), "VX", "UVX")

# -------- Option B: Time-dependent classification --------
df_long = pd.DataFrame({
    "id": np.repeat(df["id"].values, MAX_DAYS),
    "day": np.tile(np.arange(MAX_DAYS), N)
})
df_long = df_long.merge(df[["id", "death_day", "vaccinated_day"]], on="id", how="left")
df_long = df_long[(df_long["death_day"].isna()) | (df_long["day"] <= df_long["death_day"])]
df_long["group_td"] = np.where(df_long["vaccinated_day"].notna() & (df_long["day"] >= df_long["vaccinated_day"]),
                               "VX", "UVX")
df_long["death"] = (df_long["day"] == df_long["death_day"]).astype(int)

# ---------- Compute death rates ----------
records_static = []
for day in range(MAX_DAYS):
    for group in ["VX", "UVX"]:
        alive = (df["death_day"].isna()) | (df["death_day"] >= day)
        in_group = df["static_group"] == group
        deaths = ((df["death_day"] == day) & in_group).sum()
        at_risk = (alive & in_group).sum()
        rate = deaths / at_risk if at_risk > 0 else 0
        records_static.append({"day": day, "group": group, "death_rate": rate, "type": "Static"})

# Compute time-dependent death rates
death_rates_td = df_long.groupby(["day", "group_td"])["death"].sum().unstack(fill_value=0)
counts_td = df_long.groupby(["day", "group_td"]).size().unstack(fill_value=0)
rates_td = (death_rates_td / counts_td).reset_index().melt(id_vars="day", var_name="group", value_name="death_rate")
rates_td["type"] = "Time-dependent"

# Combine both approaches for plotting
df_static = pd.DataFrame(records_static)
df_combined = pd.concat([df_static, rates_td], ignore_index=True)

# ---------- Kaplan-Meier Survival Curves (Static Group) ----------
df_km = df.copy()
df_km["event"] = df_km["death_day"].notna().astype(int)
df_km["duration"] = np.where(df_km["event"] == 1, df_km["death_day"], MAX_DAYS)

kmf = KaplanMeierFitter()

km_curves = {}
km_derivatives = {}
for label in ["VX", "UVX"]:
    kmf.fit(
        durations=df_km[df_km["static_group"] == label]["duration"],
        event_observed=df_km[df_km["static_group"] == label]["event"],
        label=label,
    )
    times = kmf.survival_function_.index.values
    survival = kmf.survival_function_[label].values
    km_curves[label] = (times, survival)
    # 1st derivative (negative slope) - approximate death hazard rate (constant should be ~0.001)
    deriv = -np.gradient(survival, times)
    km_derivatives[label] = (times, deriv)

# ---------- Cox Proportional Hazards Model ----------
df_cox = df.copy()
df_cox["event"] = df_cox["death_day"].notna().astype(int)
df_cox["duration"] = np.where(df_cox["event"] == 1, df_cox["death_day"], MAX_DAYS)
df_cox["vaccinated"] = df_cox["vaccinated_day"].notna().astype(int)

cph = CoxPHFitter()
cph.fit(df_cox[["duration", "event", "vaccinated"]], duration_col="duration", event_col="event")

# Predict survival functions for vaccinated and unvaccinated
times = np.arange(0, MAX_DAYS)
surv_vx = cph.predict_survival_function(pd.DataFrame({"vaccinated": [1]}), times=times).values.flatten()
surv_uvx = cph.predict_survival_function(pd.DataFrame({"vaccinated": [0]}), times=times).values.flatten()

# 1st derivative of Cox survival curves (negative slope)
cox_deriv_vx = -np.gradient(surv_vx, times)
cox_deriv_uvx = -np.gradient(surv_uvx, times)

# ---------- Create Plotly figure ----------
fig = sp.make_subplots(rows=1, cols=1)

# Plot death rates: static and time-dependent
for typ in df_combined["type"].unique():
    for group in ["VX", "UVX"]:
        df_plot = df_combined[(df_combined["type"] == typ) & (df_combined["group"] == group)]
        fig.add_trace(
            go.Scatter(
                x=df_plot["day"],
                y=df_plot["death_rate"],
                mode="lines",
                name=f"{group} Death Rate ({typ})",
                line=dict(dash="solid" if typ == "Static" else "dot"),
            )
        )

# Plot Kaplan-Meier survival curves
for group in ["VX", "UVX"]:
    x, y = km_curves[group]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=f"{group} KM Survival",
            line=dict(width=3, dash="dash"),
        )
    )
    # Plot KM 1st derivative
    x_d, y_d = km_derivatives[group]
    fig.add_trace(
        go.Scatter(
            x=x_d,
            y=y_d,
            mode="lines",
            name=f"{group} KM 1st Derivative",
            line=dict(width=2, dash="dot"),
        )
    )

# Plot Cox model predicted survival curves
fig.add_trace(
    go.Scatter(
        x=times,
        y=surv_vx,
        mode="lines",
        name="VX Cox Predicted Survival",
        line=dict(color="green", dash="dot", width=3),
    )
)
fig.add_trace(
    go.Scatter(
        x=times,
        y=surv_uvx,
        mode="lines",
        name="UVX Cox Predicted Survival",
        line=dict(color="red", dash="dot", width=3),
    )
)

# Plot Cox 1st derivative curves
fig.add_trace(
    go.Scatter(
        x=times,
        y=cox_deriv_vx,
        mode="lines",
        name="VX Cox 1st Derivative",
        line=dict(color="green", dash="dash", width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=times,
        y=cox_deriv_uvx,
        mode="lines",
        name="UVX Cox 1st Derivative",
        line=dict(color="red", dash="dash", width=2),
    )
)

# Update layout: autoscale y-axis with some margin, legend on right in vertical column
fig.update_layout(
    title="Death Rates, Kaplan-Meier, Cox Model Survival Curves and their 1st Derivatives",
    xaxis_title="Day",
    yaxis_title="Rate / Survival Probability",
    height=700,
    legend=dict(
        x=1.02,
        y=1,
        traceorder="normal",
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0)",
        bordercolor="Black",
        borderwidth=1,
        orientation="v"
    ),
    margin=dict(l=50, r=250, t=100, b=50),
)

# Autoscale y axis: manually set range with a bit of margin around data min/max
all_y = []
for trace in fig.data:
    all_y.extend(trace.y if isinstance(trace.y, (list, np.ndarray)) else [])
if all_y:
    y_min = min(all_y)
    y_max = max(all_y)
    y_range = y_max - y_min
    fig.update_yaxes(range=[max(0, y_min - 0.1 * y_range), y_max + 0.1 * y_range])

fig.write_html(OUTPUT_HTML)
print(f"Saved interactive plot to {OUTPUT_HTML}")
