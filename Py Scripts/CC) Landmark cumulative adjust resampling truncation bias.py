import pandas as pd
import numpy as np
import plotly.graph_objs as go

# ==============================================================================
# Title: Landmark Adjustment for Resampling Truncation Bias (Immortal Time Bias)
# Description:
#   This script applies a textbook correction for immortal time bias using the
#   "landmark method", aligning vaccinated (VX) and unvaccinated (UVX) groups
#   based on the median last dose date per age. It computes weekly relative
#   risks (UVX/VX) of death for each age group and visualizes smoothed RR curves.
# ==============================================================================

# === File Paths ===
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_Vesely_106_SIMULATED.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_TWO_DEATHRATES_10X_Vesely_106_SIMULATED.csv"

OUTPUT_HTML = r"C:\github\CzechFOI-DRATE\Plot Results\CC) cumulative Landmark adjust resampling truncation bias\CC) nobias CUM_RR_AGEBIN_Landmark_Comparison.html"

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

# Standardize column names
df.columns = df.columns.str.lower().str.strip()
dose_cols = [c.lower() for c in dose_cols]

# === Preprocessing ===
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

vx = df[df['is_vaxed'] == 1].copy()
uvx = df[df['is_vaxed'] == 0].copy()

def aggregated_hazard(df_, start, end, window=7):
    bins = np.arange(start, end + 1, window)
    hz = []
    centers = []
    counts = []
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1] - 1
        at_risk = (df_['entry'] <= bin_end) & (df_['exit'] >= bin_start)
        n_at_risk = at_risk.sum()
        n_events = ((df_['death_day'] >= bin_start) & (df_['death_day'] <= bin_end) & at_risk).sum()
        hz.append(n_events / n_at_risk if n_at_risk > 0 else np.nan)
        centers.append((bin_start + bin_end) // 2)
        counts.append(n_at_risk)
    return np.array(centers), np.array(hz), np.array(counts)

fig = go.Figure()

# Collect all death days after landmark across all ages to find global end_measure
all_death_days_after_landmark = []
landmarks = []

for age in range(0, MAX_AGE + 1):
    vx_age = vx[vx['age'] == age].copy()
    uvx_age = uvx[uvx['age'] == age].copy()

    if len(vx_age) < 20 or len(uvx_age) < 20:
        print(f"Age {age}: Skipped due to small group size (VX: {len(vx_age)}, UVX: {len(uvx_age)})")
        continue

    landmark = vx_age['last_dose_day'].median()
    if pd.isna(landmark):
        print(f"Age {age}: Skipped due to NaN landmark")
        continue
    landmark = int(landmark)
    landmarks.append(landmark)

    vx_a = vx_age[vx_age['death_day'] >= landmark].copy()
    uvx_a = uvx_age[uvx_age['death_day'] >= landmark].copy()

    if len(vx_a) == 0 or len(uvx_a) == 0:
        print(f"Age {age}: Skipped due to no data after landmark alignment")
        continue

    all_death_days_after_landmark.append(vx_a['death_day'])
    all_death_days_after_landmark.append(uvx_a['death_day'])

# Global last death day across all ages after landmark alignment
end_measure = pd.concat(all_death_days_after_landmark).max()
print(f"Global end_measure (last death day after landmark): {end_measure}")

# Minimum landmark across all ages — start x-axis here
min_landmark = min(landmarks)
print(f"Minimum landmark (start of x-axis): {min_landmark}")

# Now plot per age but clip all to end_measure
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

    if len(vx_a) == 0 or len(uvx_a) == 0:
        continue

    # Clip end day to global end_measure
    age_end_day = min(pd.concat([vx_a['death_day'], uvx_a['death_day']]).max(), end_measure)

    vx_a['entry'] = landmark
    uvx_a['entry'] = landmark
    vx_a['exit'] = vx_a['death_day'].clip(upper=age_end_day)
    uvx_a['exit'] = uvx_a['death_day'].clip(upper=age_end_day)

    vx_a['pdays'] = vx_a['exit'] - vx_a['entry']
    uvx_a['pdays'] = uvx_a['exit'] - uvx_a['entry']
    vx_a = vx_a[vx_a['pdays'] > 0]
    uvx_a = uvx_a[uvx_a['pdays'] > 0]

    if len(vx_a) == 0 or len(uvx_a) == 0:
        continue

    days_vx, hz_vx, pop_vx = aggregated_hazard(vx_a, landmark, age_end_day, window=WINDOW_SIZE)
    days_uvx, hz_uvx, pop_uvx = aggregated_hazard(uvx_a, landmark, age_end_day, window=WINDOW_SIZE)

    with np.errstate(divide='ignore', invalid='ignore'):
        rr_raw = np.where(
            (hz_vx > 0) & ~np.isnan(hz_vx) & ~np.isnan(hz_uvx),
            hz_uvx / hz_vx,
            np.nan
        )

    total_pop = pop_vx + pop_uvx
    rr_raw[total_pop < HIDELIM] = np.nan
    rr_smooth = pd.Series(rr_raw).rolling(window=7, min_periods=1, center=True).median()

    fig.add_trace(go.Scatter(
        x=days_vx,
        y=rr_smooth,
        mode='lines',
        name=f'Age {age}',
        line=dict(width=1)
    ))

fig.update_layout(
    title="Relative Risk (UVX / VX) by Age — Landmark = Median Last Dose Day (Per Age)",
    xaxis_title="Day from Age-specific Last Dose Landmark (weekly bins)",
    yaxis_title="Relative Risk (UVX / VX)",
    yaxis_type="linear",
    template="plotly_white",
    height=900,
    legend=dict(itemsizing='constant', bgcolor='rgba(0,0,0,0)'),
    xaxis=dict(range=[min_landmark, end_measure])  # Start x-axis at min landmark
)

fig.write_html(OUTPUT_HTML)
print("Plot saved to:", OUTPUT_HTML)
