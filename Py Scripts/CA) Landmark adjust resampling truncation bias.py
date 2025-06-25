import pandas as pd
import numpy as np
import plotly.graph_objs as go

# This is a textbook correction for immortal time bias (resampling truncation - constraint death day >= last dose day ) 
# using the "landmark" restriction method.

# === File Paths ===
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_Vesely_106_SIMULATED.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_TWO_DEATHRATES_10X_Vesely_106_SIMULATED.csv"

OUTPUT_HTML = r"C:\github\CzechFOI-DRATE\Plot Results\CA) Landmark adjust resampling truncation bias\CA) minbias RR_AGEBIN_Landmark_Comparison.html"

# === Parameters ===
START_DATE = pd.Timestamp('2020-01-01')
REFERENCE_YEAR = 2023
MAX_AGE = 113
WINDOW_SIZE = 30
HIDELIM = 10  # Hide RR values where person-days < HIDELIM

# === Load Data ===
dose_cols = [f"Datum_{i}" for i in range(1, 8)]
cols = ['Rok_narozeni', 'DatumUmrti'] + dose_cols
df = pd.read_csv(INPUT_CSV, usecols=cols, parse_dates=['DatumUmrti'] + dose_cols, dayfirst=False)
df.columns = df.columns.str.lower().str.strip()
dose_cols = [c.lower() for c in dose_cols]

# --- Preprocess ---
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

# --- Split groups ---
vx = df[df['is_vaxed'] == 1].copy()
uvx = df[df['is_vaxed'] == 0].copy()

# --- Aggregated Hazard Function ---
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

# --- Plotting ---
fig = go.Figure()

for age in range(0, MAX_AGE + 1):
    vx_age = vx[vx['age'] == age].copy()
    uvx_age = uvx[uvx['age'] == age].copy()

    if len(vx_age) < 20 or len(uvx_age) < 20:
        # Debug print for skipping due to small group size
        print(f"Age {age}: Skipped due to small group size (VX: {len(vx_age)}, UVX: {len(uvx_age)})")
        continue

    # Median last dose day for this age
    landmark = vx_age['last_dose_day'].median()
    if pd.isna(landmark):
        print(f"Age {age}: Skipped due to NaN landmark")
        continue
    landmark = int(landmark)
    print(f"Age {age}: Landmark (median last dose day) = {landmark}")

    # Align both groups at this landmark
    vx_a = vx_age[vx_age['death_day'] >= landmark].copy()
    uvx_a = uvx_age[uvx_age['death_day'] >= landmark].copy()

    # Check counts after alignment
    print(f"Age {age}: VX after alignment: {len(vx_a)}, UVX after alignment: {len(uvx_a)}")
    if len(vx_a) == 0 or len(uvx_a) == 0:
        print(f"Age {age}: Skipped due to no data after landmark alignment")
        continue

    # Use per-age last death day for follow-up cutoff
    age_end_day = pd.concat([vx_a['death_day'], uvx_a['death_day']]).max()
    print(f"Age {age}: Follow-up end day = {age_end_day}")

    vx_a['entry'] = landmark
    uvx_a['entry'] = landmark
    vx_a['exit'] = vx_a['death_day'].clip(upper=age_end_day)
    uvx_a['exit'] = uvx_a['death_day'].clip(upper=age_end_day)

    vx_a['pdays'] = vx_a['exit'] - vx_a['entry']
    uvx_a['pdays'] = uvx_a['exit'] - uvx_a['entry']
    vx_a = vx_a[vx_a['pdays'] > 0]
    uvx_a = uvx_a[uvx_a['pdays'] > 0]

    # Check counts after filtering for positive person-days
    print(f"Age {age}: VX with positive pdays: {len(vx_a)}, UVX with positive pdays: {len(uvx_a)}")
    if len(vx_a) == 0 or len(uvx_a) == 0:
        print(f"Age {age}: Skipped due to no data after filtering positive person-days")
        continue

    # Compute hazard
    days_vx, hz_vx, pop_vx = aggregated_hazard(vx_a, landmark, age_end_day, window=WINDOW_SIZE)
    days_uvx, hz_uvx, pop_uvx = aggregated_hazard(uvx_a, landmark, age_end_day, window=WINDOW_SIZE)

    print(f"Age {age}: Hazard VX (first 5): {hz_vx[:5]}")
    print(f"Age {age}: Hazard UVX (first 5): {hz_uvx[:5]}")

    with np.errstate(divide='ignore', invalid='ignore'):
        rr_raw = np.where(
            (hz_vx > 0) & ~np.isnan(hz_vx) & ~np.isnan(hz_uvx),
            hz_uvx / hz_vx,
            np.nan
        )

    # Filter by person-days
    total_pop = pop_vx + pop_uvx
    rr_raw[total_pop < HIDELIM] = np.nan

    # Debug print some RR values
    print(f"Age {age}: RR raw (first 5): {rr_raw[:5]}")

    # Smooth RR
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
    height=700,
    width=1000,
    legend=dict(itemsizing='constant', bgcolor='rgba(0,0,0,0)')
)

fig.write_html(OUTPUT_HTML)
print("Plot saved to:", OUTPUT_HTML)
