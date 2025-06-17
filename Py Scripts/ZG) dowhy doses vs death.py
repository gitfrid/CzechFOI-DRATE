import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dowhy import CausalModel

# Config
REFERENCE_DATE = pd.Timestamp("2020-01-01")
AGE_BIN_WIDTH = 1
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
#INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
OUTPUT_HTML = r"C:\CzechFOI-DRATE\Plot Results\ZG) dowhy doses vs death\ZG) sim no bias doses_vs_deaths_dowhy.html"
CURRENT_YEAR = 2023

dose_date_cols = [f'Datum_{i}' for i in range(1, 8)]
date_cols = ['DatumUmrti'] + dose_date_cols

# Load data
df = pd.read_csv(
    INPUT_CSV,
    usecols=['Rok_narozeni'] + date_cols,
    parse_dates=date_cols,
    dayfirst=False,
    low_memory=False
)

df['death'] = (~df['DatumUmrti'].isna()).astype(int)
df['age'] = CURRENT_YEAR - df['Rok_narozeni']

bins = np.arange(0, 115 + AGE_BIN_WIDTH, AGE_BIN_WIDTH)
labels = [f"{i}-{i}" for i in range(0, 115)]
df['age_bin'] = pd.cut(df['age'], bins=bins, right=False, labels=labels)
df = df[df['age_bin'].notna()]
df['age_bin'] = df['age_bin'].astype('category')

results = []

for age_bin in df['age_bin'].cat.categories:
    sub = df[df['age_bin'] == age_bin].copy()
    if len(sub) < 50:
        continue

    dose_days_all = pd.Series(dtype='int')
    for col in dose_date_cols:
        dose_days = (sub[col] - REFERENCE_DATE).dt.days.dropna()
        dose_days_all = pd.concat([dose_days_all, dose_days], ignore_index=True)

    death_days = (sub['DatumUmrti'] - REFERENCE_DATE).dt.days.dropna()

    if dose_days_all.empty and death_days.empty:
        continue

    min_day = int(min(
        dose_days_all.min() if not dose_days_all.empty else np.inf,
        death_days.min() if not death_days.empty else np.inf))
    max_day = int(max(
        dose_days_all.max() if not dose_days_all.empty else -np.inf,
        death_days.max() if not death_days.empty else -np.inf))

    days_range = np.arange(min_day, max_day + 1)

    dose_curve = dose_days_all.value_counts().reindex(days_range, fill_value=0).sort_index()
    death_curve = death_days.value_counts().reindex(days_range, fill_value=0).sort_index()

    sub['first_dose_day'] = sub[dose_date_cols].apply(
        lambda row: (row.min() - REFERENCE_DATE).days if pd.notna(row.min()) else np.nan, axis=1)

    vx_deaths_per_day = pd.Series(0, index=days_range)
    uvx_deaths_per_day = pd.Series(0, index=days_range)

    death_dates = sub.loc[sub['death'] == 1, ['DatumUmrti', 'first_dose_day']]

    for _, row in death_dates.iterrows():
        death_day = (row['DatumUmrti'] - REFERENCE_DATE).days
        first_dose_day = row['first_dose_day']
        if pd.notna(first_dose_day) and death_day >= first_dose_day:
            vx_deaths_per_day[death_day] += 1
        else:
            uvx_deaths_per_day[death_day] += 1

    def run_dowhy_ate_estimation(treatment, outcome, age_val):
        ts_data = pd.DataFrame({
            'Doses_curve': treatment.values,
            'D_Curve': outcome.values,
            'AgeGroup': [age_val] * len(outcome),
            'Time': np.arange(len(outcome))
        })

        if ts_data['Doses_curve'].std() == 0 or ts_data['D_Curve'].std() == 0:
            return np.nan, False, np.nan, np.nan

        try:
            model = CausalModel(
                data=ts_data,
                treatment="Doses_curve",
                outcome="D_Curve",
                graph="""
                digraph {
                    Doses_curve -> D_Curve;
                    Time -> D_Curve;
                    AgeGroup -> Doses_curve;
                    AgeGroup -> D_Curve;
                }
                """
            )

            identified_estimand = model.identify_effect()
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression",
                test_significance=True
            )

            est_value = getattr(estimate, 'value', np.nan)
            p_val = getattr(estimate, 'p_value', np.nan)
            stderr = getattr(estimate, 'stderr', np.nan)
            ci95 = 1.96 * stderr if not np.isnan(stderr) else np.nan

            try:
                sig = estimate.test_stat_significance()
            except Exception:
                sig = False

            return est_value, sig, p_val, ci95

        except Exception:
            return np.nan, False, np.nan, np.nan

    age_val = int(age_bin.split('-')[0])
    ate_total, sig_total, pval_total, ci_total = run_dowhy_ate_estimation(dose_curve, death_curve, age_val)
    ate_vx, sig_vx, pval_vx, ci_vx = run_dowhy_ate_estimation(dose_curve, vx_deaths_per_day, age_val)
    ate_uvx, sig_uvx, pval_uvx, ci_uvx = run_dowhy_ate_estimation(dose_curve, uvx_deaths_per_day, age_val)

    results.append({
        'age_bin': age_bin,
        'age': age_val,
        'ate_total': ate_total,
        'ate_vx': ate_vx,
        'ate_uvx': ate_uvx,
        'ci_total': ci_total,
        'ci_vx': ci_vx,
        'ci_uvx': ci_uvx,
        'sig_total': sig_total,
        'sig_vx': sig_vx,
        'sig_uvx': sig_uvx,
        'pval_total': pval_total,
        'pval_vx': pval_vx,
        'pval_uvx': pval_uvx,
        'death_rate_total': death_curve.mean(),
        'death_rate_vx': vx_deaths_per_day.mean(),
        'death_rate_uvx': uvx_deaths_per_day.mean(),
        'mean_doses': dose_curve.mean(),
        'N_total': len(sub)
    })
    
    print(
        f"Age bin {age_bin}: DoWhy ATE total={ate_total:.6f} (p={pval_total:.4g}, significant={sig_total}), "
        f"vx={ate_vx:.6f} (p={pval_vx:.4g}, significant={sig_vx}, ci_vx={'ci_vx'}), "
        f"uvx={ate_uvx:.6f} (p={pval_uvx:.4g}, significant={sig_uvx}, ci_uvx={'ci_uvx'})"
        f"Mean Daily Deaths total={death_curve.mean():.6f}, vx={vx_deaths_per_day.mean():.6f}, uvx={uvx_deaths_per_day.mean():.6f}, "
        f"Mean Daily Doses={dose_curve.mean():.2f}, N={len(sub)}"

    )

# Plotting
df_causal = pd.DataFrame(results).dropna(subset=['ate_total'])
if df_causal.empty:
    print("No valid DoWhy results; skipping plot.")


def to_str_safe(x):
    if isinstance(x, dict):
        return str(x)  # or extract a specific key if you want
    elif pd.isna(x):
        return ""
    else:
        return str(x)

df_causal['hover_total'] = (
    "Age: " + df_causal['age'].astype(str) + "<br>" +
    "ATE Total Deaths: " + df_causal['ate_total'].map(lambda x: f"{x:.6f}" if pd.notna(x) and not isinstance(x, dict) else to_str_safe(x)) + "<br>" +
    "CI95: ±" + df_causal['ci_total'].map(lambda x: f"{x:.6f}" if pd.notna(x) and not isinstance(x, dict) else to_str_safe(x)) + "<br>" +
    "Significant: " + df_causal['sig_total'].map(to_str_safe) + "<br>" +
    "P-Value: " + df_causal['pval_total'].map(to_str_safe) + "<br>" +
    "Mean Death Rate Total: " + df_causal['death_rate_total'].map(lambda x: f"{x:.6f}" if pd.notna(x) and not isinstance(x, dict) else to_str_safe(x)) + "<br>" +
    "Mean Daily Doses: " + df_causal['mean_doses'].map(lambda x: f"{x:.2f}" if pd.notna(x) and not isinstance(x, dict) else to_str_safe(x)) + "<br>" +
    "N Total: " + df_causal['N_total'].astype(str)
)

df_causal['hover_vx'] = (
    "Age: " + df_causal['age'].astype(str) + "<br>" +
    "ATE Vaccinated Deaths: " + df_causal['ate_vx'].map(lambda x: f"{x:.6f}" if pd.notna(x) and not isinstance(x, dict) else to_str_safe(x)) + "<br>" +
    "CI95: ±" + df_causal['ci_vx'].map(lambda x: f"{x:.6f}" if pd.notna(x) and not isinstance(x, dict) else to_str_safe(x)) + "<br>" +
    "Significant: " + df_causal['sig_vx'].map(to_str_safe) + "<br>" +
    "P-Value: " + df_causal['pval_vx'].map(to_str_safe) + "<br>" +
    "Mean Death Rate Vx: " + df_causal['death_rate_vx'].map(lambda x: f"{x:.6f}" if pd.notna(x) and not isinstance(x, dict) else to_str_safe(x)) + "<br>" +
    "Mean Daily Doses: " + df_causal['mean_doses'].map(lambda x: f"{x:.2f}" if pd.notna(x) and not isinstance(x, dict) else to_str_safe(x)) + "<br>" +
    "N Total: " + df_causal['N_total'].astype(str)
)

df_causal['hover_uvx'] = (
    "Age: " + df_causal['age'].astype(str) + "<br>" +
    "ATE Unvaccinated Deaths: " + df_causal['ate_uvx'].map(lambda x: f"{x:.6f}" if pd.notna(x) and not isinstance(x, dict) else to_str_safe(x)) + "<br>" +
    "CI95: ±" + df_causal['ci_uvx'].map(lambda x: f"{x:.6f}" if pd.notna(x) and not isinstance(x, dict) else to_str_safe(x)) + "<br>" +
    "Significant: " + df_causal['sig_uvx'].map(to_str_safe) + "<br>" +
    "P-Value: " + df_causal['pval_uvx'].map(to_str_safe) + "<br>" +
    "Mean Death Rate UVx: " + df_causal['death_rate_uvx'].map(lambda x: f"{x:.6f}" if pd.notna(x) and not isinstance(x, dict) else to_str_safe(x)) + "<br>" +
    "Mean Daily Doses: " + df_causal['mean_doses'].map(lambda x: f"{x:.2f}" if pd.notna(x) and not isinstance(x, dict) else to_str_safe(x)) + "<br>" +
    "N Total: " + df_causal['N_total'].astype(str)
)

fig = go.Figure()

# Now use hovertext from df_causal columns for each trace, and set hoverinfo='text'

fig.add_trace(go.Scatter(
    x=df_causal['age'],
    y=df_causal['ate_total'],
    mode='markers+lines',
    name='ATE Total Deaths',
    line=dict(color='red'),
    marker=dict(symbol='circle', size=8, color='red'),    
    hovertext=df_causal['hover_total'],
    hoverinfo='text'
))

fig.add_trace(go.Scatter(
    x=df_causal['age'],
    y=df_causal['ate_vx'],
    mode='markers+lines',
    name='ATE Vaccinated Deaths',
    line=dict(color='blue'),
    marker=dict(symbol='diamond', size=8, color='blue'),    
    hovertext=df_causal['hover_vx'],
    hoverinfo='text'
))

fig.add_trace(go.Scatter(
    x=df_causal['age'],
    y=df_causal['ate_uvx'],
    mode='markers+lines',
    name='ATE Unvaccinated Deaths',
    line=dict(color='green'),
    marker=dict(symbol='square', size=8, color='green'),    
    hovertext=df_causal['hover_uvx'],
    hoverinfo='text'
))

# The rest remains the same, including the mean daily deaths and mean daily doses traces

fig.add_trace(go.Scatter(
    x=df_causal['age'],
    y=df_causal['death_rate_total'],
    mode='lines+markers',
    name='Mean Daily Deaths (Total)',
    yaxis='y2',
    marker=dict(color='black'),
    hovertemplate='Age=%{x}<br>Mean Daily Deaths Total=%{y:.6f}'
))

fig.add_trace(go.Scatter(
    x=df_causal['age'],
    y=df_causal['mean_doses'],
    mode='lines+markers',
    name='Mean Daily Dose Count',
    yaxis='y3',
    marker=dict(color='purple'),
    hovertemplate='Age=%{x}<br>Mean Daily Doses=%{y:.2f}'
))

# Layout unchanged...
fig.update_layout(
    title="DoWhy: Estimated Causal Effect of Dose Count on Death Risk by Age Bin (Total, Vaccinated, Unvaccinated)",
    xaxis_title='Age',
    yaxis=dict(
        title='Estimated ATE',
        side='left',
        showgrid=True,
        zeroline=True,
    ),
    yaxis2=dict(
        title='Mean Daily Deaths (Total)',
        overlaying='y',
        side='right',
        position=0.95
    ),
    yaxis3=dict(
        title='Mean Daily Dose Count',
        anchor='free',
        overlaying='y',
        side='right',
        position=1,
        showgrid=False
    ),
    template='plotly_white',
    height=700
)

fig.write_html(OUTPUT_HTML)
print(f"Plot saved to: {OUTPUT_HTML}")
