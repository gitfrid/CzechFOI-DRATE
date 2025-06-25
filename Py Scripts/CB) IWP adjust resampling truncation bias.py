import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LogisticRegression


# This is a textbook correction for immortal time bias (resampling truncation - constraint death day >= last dose day ) 
# using the "landmark" restriction method.

# === File Paths ===
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_Vesely_106_SIMULATED.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_TWO_DEATHRATES_10X_Vesely_106_SIMULATED.csv"

OUTPUT_HTML = r"C:\github\CzechFOI-DRATE\Plot Results\CB) IWP adjust resampling truncation bias\CB) minbias RR_AGEBIN_IWP_Comparison.html"

# === Parameters ===
START_DATE = pd.Timestamp('2020-01-01')
REFERENCE_YEAR = 2023
MAX_AGE = 113
WINDOW_SIZE = 30
HIDELIM = 10

# === Load Data ===
dose_cols = [f"Datum_{i}" for i in range(1, 8)]
cols = ['Rok_narozeni', 'DatumUmrti'] + dose_cols
df = pd.read_csv(INPUT_CSV, usecols=cols, parse_dates=['DatumUmrti'] + dose_cols, dayfirst=False)
df.columns = df.columns.str.lower().str.strip()
dose_cols = [c.lower() for c in dose_cols]

df['birth_year'] = pd.to_numeric(df['rok_narozeni'], errors='coerce')
df['age'] = REFERENCE_YEAR - df['birth_year']
df = df[df['age'].between(0, MAX_AGE)].copy()
df['is_vaxed'] = df[dose_cols].notna().any(axis=1).astype(int)

def to_day(d):
    return (d - START_DATE).dt.days

df['death_day'] = to_day(df['datumumrti'])
for c in dose_cols:
    df[c + '_day'] = to_day(df[c])
df['last_dose_day'] = df[[c + '_day' for c in dose_cols]].max(axis=1, skipna=True)
df = df.dropna(subset=['death_day'])

# === Compute IPW ===
logreg = LogisticRegression()
X = df[['age']]
y = df['is_vaxed']
logreg.fit(X, y)
p_vax = logreg.predict_proba(X)[:, 1]
df['ipw'] = np.where(df['is_vaxed'] == 1, 1 / p_vax, 1 / (1 - p_vax))
df['ipw'] = df['ipw'].clip(upper=10)  # prevent outliers

# --- PRINT IPW weights summary for whole dataset ---
print("=== IPW Weights Summary for Full Dataset ===")
print(df['ipw'].describe())

# --- PRINT IPW weights summary per age group ---
print("\n=== IPW Weights Summary by Age Group ===")
for age in sorted(df['age'].unique()):
    ipw_age = df.loc[df['age'] == age, 'ipw']
    if len(ipw_age) > 0:
        print(f"Age {age}: count={len(ipw_age)}, mean={ipw_age.mean():.3f}, std={ipw_age.std():.3f}, min={ipw_age.min():.3f}, max={ipw_age.max():.3f}")

# === Split Groups ===
vx = df[df['is_vaxed'] == 1].copy()
uvx = df[df['is_vaxed'] == 0].copy()

# === Aggregated Weighted Hazard Function ===
def aggregated_hazard(df_, start, end, window=7):
    bins = np.arange(start, end + 1, window)
    hz = []
    centers = []
    counts = []
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1] - 1
        at_risk = (df_['entry'] <= bin_end) & (df_['exit'] >= bin_start)
        n_at_risk = df_.loc[at_risk, 'ipw'].sum()
        n_events = df_[(df_['death_day'] >= bin_start) & (df_['death_day'] <= bin_end) & at_risk]['ipw'].sum()
        hz.append(n_events / n_at_risk if n_at_risk > 0 else np.nan)
        centers.append((bin_start + bin_end) // 2)
        counts.append(n_at_risk)
    return np.array(centers), np.array(hz), np.array(counts)

# === Plotting ===
fig = go.Figure()

for age in range(0, MAX_AGE + 1):
    vx_age = vx[vx['age'] == age].copy()
    uvx_age = uvx[uvx['age'] == age].copy()

    if len(vx_age) < 20 or len(uvx_age) < 20:
        continue

    landmark = vx_age['last_dose_day'].median()
    if pd.isna(landmark):
        continue
    landmark = int(landmark)

    vx_a = vx_age[vx_age['death_day'] >= landmark].copy()
    uvx_a = uvx_age[uvx_age['death_day'] >= landmark].copy()

    age_end_day = pd.concat([vx_a['death_day'], uvx_a['death_day']]).max()

    vx_a['entry'] = landmark
    uvx_a['entry'] = landmark
    vx_a['exit'] = vx_a['death_day'].clip(upper=age_end_day)
    uvx_a['exit'] = uvx_a['death_day'].clip(upper=age_end_day)

    vx_a['pdays'] = vx_a['exit'] - vx_a['entry']
    uvx_a['pdays'] = uvx_a['exit'] - uvx_a['entry']
    vx_a = vx_a[vx_a['pdays'] > 0]
    uvx_a = uvx_a[uvx_a['pdays'] > 0]

    if vx_a.empty or uvx_a.empty:
        continue

    days_vx, hz_vx, pop_vx = aggregated_hazard(vx_a, landmark, age_end_day, window=WINDOW_SIZE)
    days_uvx, hz_uvx, pop_uvx = aggregated_hazard(uvx_a, landmark, age_end_day, window=WINDOW_SIZE)

    with np.errstate(divide='ignore', invalid='ignore'):
        rr_ipw = np.where(
            (hz_vx > 0) & ~np.isnan(hz_vx) & ~np.isnan(hz_uvx),
            hz_uvx / hz_vx,
            np.nan
        )

    total_pop = pop_vx + pop_uvx
    rr_ipw[total_pop < HIDELIM] = np.nan
    rr_smooth = pd.Series(rr_ipw).rolling(window=7, min_periods=1, center=True).median()

    fig.add_trace(go.Scatter(
        x=days_vx,
        y=rr_smooth,
        mode='lines',
        name=f'Age {age}',
        line=dict(width=1)
    ))

fig.update_layout(
    title="IPW-Adjusted Relative Risk (UVX / VX) by Age — Landmark = Median Last Dose Day",
    xaxis_title="Day from Age-specific Last Dose Landmark (weekly bins)",
    yaxis_title="Relative Risk (UVX / VX)",
    yaxis_type="linear",
    template="plotly_white",
    height=700,
    width=1000,
    legend=dict(itemsizing='constant', bgcolor='rgba(0,0,0,0)')
)

fig.write_html(OUTPUT_HTML)
print("Plot saved to:", OUTPUT_HTML)
