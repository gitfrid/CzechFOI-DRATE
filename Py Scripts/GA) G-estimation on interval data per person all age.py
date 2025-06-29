import pandas as pd
import numpy as np
from lifelines import CoxTimeVaryingFitter
from scipy.optimize import minimize_scalar
import plotly.graph_objects as go
import warnings
from lifelines.exceptions import ConvergenceWarning

# === Configuration ===
AGE_FILTER = [70]  # Set to [] or None to process all ages
# INPUT_CSV = r"C:\CzechFOI-DRATE\intervals_per_agebin\AG70_real_intervals_for_cox_model_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\intervals_per_agebin\AG70_sim_minbias_intervals_for_cox_model_Vesely_106_202403141131.csv"
INPUT_CSV = r"C:\CzechFOI-DRATE\intervals_per_agebin\real_intervals_for_cox_model_Vesely_106_202403141131.csv"

OUTPUT_HTML_PSI = r"C:\CzechFOI-DRATE\Plot Results\G) G-estimation\minbias_g_estimation_phi_plots.html"
OUTPUT_HTML_COMPARISON = r"C:\CzechFOI-DRATE\Plot Results\G) G-estimation\minbias_g_estimation_comparison_plot.html"

# === Optionally suppress convergence warnings ===
# warnings.filterwarnings("ignore", category=ConvergenceWarning)

# === Load and validate data ===
try:
    df_all = pd.read_csv(INPUT_CSV)
except Exception as e:
    raise RuntimeError(f"Failed to load CSV file: {e}")

required_cols = {'person_id', 'age', 'start_day', 'end_day', 'dose_number', 'event'}
missing = required_cols - set(df_all.columns)
if missing:
    raise ValueError(f"Missing required columns in CSV: {missing}")

df_all['person_id'] = df_all['person_id'].astype(str)
df_all['start_day'] = pd.to_numeric(df_all['start_day'], errors='coerce')
df_all['end_day'] = pd.to_numeric(df_all['end_day'], errors='coerce')
df_all['dose_number'] = pd.to_numeric(df_all['dose_number'], errors='coerce')
df_all['age'] = pd.to_numeric(df_all['age'], errors='coerce')
df_all.dropna(subset=['person_id', 'start_day', 'end_day', 'dose_number', 'age'], inplace=True)

# Fix multiple death intervals per person_id
multi_death_counts = df_all[df_all['event'] == 1].groupby('person_id').size()
multi_death_ids = multi_death_counts[multi_death_counts > 1].index
if len(multi_death_ids) > 0:
    print(f"\n🚨 Found {len(multi_death_ids)} person_ids with multiple death intervals. Keeping only earliest per person.\n")
    deaths_to_keep = (
        df_all[(df_all['person_id'].isin(multi_death_ids)) & (df_all['event'] == 1)]
        .sort_values(['person_id', 'end_day'])
        .groupby('person_id')
        .first()
        .reset_index()
    )
    # Remove all death intervals for these person_ids
    df_all = df_all[~((df_all['person_id'].isin(multi_death_ids)) & (df_all['event'] == 1))]
    # Add back only the earliest death interval per person_id
    df_all = pd.concat([df_all, deaths_to_keep], ignore_index=True)
    print("✅ Multiple death intervals cleaned.\n")
else:
    print("\nNo multiple death intervals found.\n")

df_all['event'] = df_all['event'].astype(int)
df_all['vaccinated'] = (df_all['dose_number'] > 0).astype(int)
MAX_OBSERVED_DAY = df_all.loc[df_all['event'] == 1, 'end_day'].max()
MAX_OBSERVED_DAY = MAX_OBSERVED_DAY +1 
print(f"MAX_OBSERVED_DAY: {MAX_OBSERVED_DAY}")

# Summary stats
total_pop_start = df_all['person_id'].nunique()
death_ids = df_all[df_all['event'] == 1]['person_id'].unique()
total_deaths = len(death_ids)
total_pop_end = total_pop_start - total_deaths
print("\nSummary:")
print(f"Total POP START: {total_pop_start:,}")
print(f"Total POP END:   {total_pop_end:,}")
print(f"Deaths:          {total_deaths:,}")

def compute_cox_coef(psi, df):
    print(f"\n[compute_cox_coef] Computing for psi={psi:.6f}")
    df_adj = df.copy()
    
    # Adjust end_day with exp(-psi * vaccinated)
    df_adj['end_day_adj'] = df_adj['start_day'] + (df_adj['end_day'] - df_adj['start_day']) * np.exp(-psi * df_adj['vaccinated'])
    
    # Clip adjusted end_day to MAX_OBSERVED_DAY
    df_adj['end_day_adj'] = np.minimum(df_adj['end_day_adj'], MAX_OBSERVED_DAY)

    # Fix zero-length intervals by adding small epsilon if start_day == end_day_adj
    same_time = df_adj['start_day'] == df_adj['end_day_adj']
    if same_time.any():
        print(f"  Warning: zero-length intervals found at indices {df_adj[same_time].index.tolist()}")
        df_adj.loc[same_time, 'end_day_adj'] += 0.5  # add half a day to avoid zero-length intervals

    n_events = df_adj['event'].sum()
    vacc_var = df_adj['vaccinated'].var()
    print(f"  Number of events: {n_events}")
    print(f"  Variance of vaccinated: {vacc_var:.6f}")
    if n_events == 0 or vacc_var == 0:
        print("  No events or no vaccinated variance - returning NaN")
        return np.nan

    for max_steps in [50, 100, 200]:
        try:
            ctv = CoxTimeVaryingFitter(penalizer=0.0)
            ctv.fit(
                df_adj.drop(columns=['age'], errors='ignore'),
                id_col='person_id',
                start_col='start_day',
                stop_col='end_day_adj',
                event_col='event',
                show_progress=False,
                robust=True,
                fit_options={'max_steps': max_steps}
            )
            coef = ctv.params_.get('vaccinated', np.nan)
            print(f"  Cox fit success with max_steps={max_steps}, coef (vaccinated) = {coef}")
            return abs(coef) if pd.notna(coef) else np.nan
        except Exception as e:
            print(f"  Cox fit failed for psi={psi} with max_steps={max_steps}. Error: {e}")
            continue
    print("  Cox fit failed for all max_steps attempts, returning NaN")
    return np.nan

def objective(psi, df):
    coef = compute_cox_coef(psi, df)
    if np.isfinite(coef):
        print(f"  Objective: coef={coef}")
        return coef
    else:
        print(f"  Objective: coef is not finite (NaN or Inf)")
        return np.inf

def g_estimate_cox(df):
    print(f"\n[g_estimate_cox] Starting optimization for age group with {df['person_id'].nunique()} unique persons")
    print("  Sample data:")
    print(df.head(5))

    for test_psi in [-0.05, 0, 0.05]:
        print(f"\n  Testing compute_cox_coef at psi={test_psi}")
        val = compute_cox_coef(test_psi, df)
        print(f"    Result: {val}")

    result = minimize_scalar(
        lambda psi: objective(psi, df),
        bounds=(-0.05, 0.05),
        method='bounded',
        options={'xatol': 1e-6}
    )
    if result.success and np.isfinite(result.fun):
        print(f"\n  Optimization success: Estimated psi = {result.x:.6f}, function value = {result.fun}")
        return result.x
    else:
        print("\n  Optimization failed or returned non-finite result.")
        return np.nan

results = []
print("\nStarting processing of age groups...")

age_groups = df_all['age'].unique() if not AGE_FILTER else AGE_FILTER
age_groups = sorted(age_groups)

for age in age_groups:
    group_df = df_all[df_all['age'] == age]
    if group_df.empty:
        print(f"Age {age}: no data, skipping.")
        continue

    print(f"\nProcessing age group {age}...")
    psi = g_estimate_cox(group_df)

    n_vx = group_df[group_df['vaccinated'] == 1]['person_id'].nunique()
    n_uvx = group_df[group_df['vaccinated'] == 0]['person_id'].nunique()
    n_total = group_df['person_id'].nunique()

    results.append({
        'age': age,
        'psi': psi,
        'n_total': n_total,
        'n_vx': n_vx,
        'n_uvx': n_uvx
    })

df_result = pd.DataFrame(results).sort_values('age')

# Plot: Estimated ψ by Age
fig_psi = go.Figure()
fig_psi.add_trace(go.Scatter(
    x=df_result['age'],
    y=df_result['psi'],
    mode='lines+markers',
    name='Estimated ψ',
    hovertemplate='Age: %{x}<br>ψ: %{y:.6f}<extra></extra>'
))
fig_psi.update_layout(
    title='G-estimation of Vaccine Effect (ψ) by Age',
    xaxis_title='Age',
    yaxis_title='Estimated ψ',
    template='plotly_white'
)

# Save figure to HTML
fig_psi.write_html(OUTPUT_HTML_PSI)

print(f"\nPlot saved to: {OUTPUT_HTML_PSI}")
