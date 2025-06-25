
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
# 
# Method:
#   - Align both VX and UVX cohorts using age-specific median last dose days.
#   - Include only individuals whose death occurred on or after this landmark.
#   - Use aggregated hazard functions within weekly bins.
#   - Compute and smooth relative risk per age and render with Plotly.
# 
# Inputs:
#   - Simulated or real dataset with birth year, death date, and up to 7 dose dates.
# 
# Output:
#   - Interactive HTML plot showing smoothed RR curves by age.
# ==============================================================================

# === File Paths ===
# Choose one of the input files below
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_Vesely_106_SIMULATED.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_TWO_DEATHRATES_10X_Vesely_106_SIMULATED.csv"

OUTPUT_HTML = r"C:\github\CzechFOI-DRATE\Plot Results\CA) Landmark adjust resampling truncation bias\CA) minbias RR_AGEBIN_Landmark_Comparison.html"

# === Parameters ===
START_DATE = pd.Timestamp('2020-01-01')  # Day zero for conversion to days
REFERENCE_YEAR = 2023  # Used to calculate age from birth year
MAX_AGE = 113  # Maximum age to include
WINDOW_SIZE = 30  # Bin size for aggregated hazard computation
HIDELIM = 10  # Minimum person-days required to include RR value

# === Load Data ===
dose_cols = [f"Datum_{i}" for i in range(1, 8)]  # Dose date columns
cols = ['Rok_narozeni', 'DatumUmrti'] + dose_cols
df = pd.read_csv(INPUT_CSV, usecols=cols, parse_dates=['DatumUmrti'] + dose_cols, dayfirst=False)

# Standardize column names
df.columns = df.columns.str.lower().str.strip()
dose_cols = [c.lower() for c in dose_cols]

# === Preprocessing ===
df['birth_year'] = pd.to_numeric(df['rok_narozeni'], errors='coerce')
df['age'] = REFERENCE_YEAR - df['birth_year']  # Calculate age at reference year
df = df[df['age'].between(0, MAX_AGE)].copy()  # Filter out-of-range ages
df['is_vaxed'] = df[dose_cols].notna().any(axis=1).astype(int)  # Mark vaccinated individuals

# Convert dates to days from START_DATE
def to_day(d):
    return (d - START_DATE).dt.days

df['death_day'] = to_day(df['datumumrti'])
for c in dose_cols:
    df[c + '_day'] = to_day(df[c])

# Compute last dose day for each individual
df['last_dose_day'] = df[[c + '_day' for c in dose_cols]].max(axis=1, skipna=True)
df = df.dropna(subset=['death_day'])  # Remove records without death day

# === Split groups ===
vx = df[df['is_vaxed'] == 1].copy()
uvx = df[df['is_vaxed'] == 0].copy()

# === Aggregated Hazard Function ===
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

# === Plotting Setup ===
fig = go.Figure()

# Loop through each age
for age in range(0, MAX_AGE + 1):
    vx_age = vx[vx['age'] == age].copy()
    uvx_age = uvx[uvx['age'] == age].copy()

    if len(vx_age) < 20 or len(uvx_age) < 20:
        # Skip small groups
        print(f"Age {age}: Skipped due to small group size (VX: {len(vx_age)}, UVX: {len(uvx_age)})")
        continue

    # Compute landmark: median last dose day for VX group
    landmark = vx_age['last_dose_day'].median()
    if pd.isna(landmark):
        print(f"Age {age}: Skipped due to NaN landmark")
        continue
    landmark = int(landmark)
    print(f"Age {age}: Landmark (median last dose day) = {landmark}")

    # Landmark alignment: only keep records with death day >= landmark
    vx_a = vx_age[vx_age['death_day'] >= landmark].copy()
    uvx_a = uvx_age[uvx_age['death_day'] >= landmark].copy()

    # Skip if no one survives past the landmark
    print(f"Age {age}: VX after alignment: {len(vx_a)}, UVX after alignment: {len(uvx_a)}")
    if len(vx_a) == 0 or len(uvx_a) == 0:
        print(f"Age {age}: Skipped due to no data after landmark alignment")
        continue

    # Determine last death day for follow-up period
    age_end_day = pd.concat([vx_a['death_day'], uvx_a['death_day']]).max()
    print(f"Age {age}: Follow-up end day = {age_end_day}")

    # Set entry and exit points for hazard computation
    vx_a['entry'] = landmark
    uvx_a['entry'] = landmark
    vx_a['exit'] = vx_a['death_day'].clip(upper=age_end_day)
    uvx_a['exit'] = uvx_a['death_day'].clip(upper=age_end_day)

    # Calculate person-days
    vx_a['pdays'] = vx_a['exit'] - vx_a['entry']
    uvx_a['pdays'] = uvx_a['exit'] - uvx_a['entry']
    vx_a = vx_a[vx_a['pdays'] > 0]
    uvx_a = uvx_a[uvx_a['pdays'] > 0]

    # Skip if no positive follow-up time
    print(f"Age {age}: VX with positive pdays: {len(vx_a)}, UVX with positive pdays: {len(uvx_a)}")
    if len(vx_a) == 0 or len(uvx_a) == 0:
        print(f"Age {age}: Skipped due to no data after filtering positive person-days")
        continue

    # Compute hazard functions
    days_vx, hz_vx, pop_vx = aggregated_hazard(vx_a, landmark, age_end_day, window=WINDOW_SIZE)
    days_uvx, hz_uvx, pop_uvx = aggregated_hazard(uvx_a, landmark, age_end_day, window=WINDOW_SIZE)

    print(f"Age {age}: Hazard VX (first 5): {hz_vx[:5]}")
    print(f"Age {age}: Hazard UVX (first 5): {hz_uvx[:5]}")

    # Compute raw relative risk (RR = UVX / VX)
    with np.errstate(divide='ignore', invalid='ignore'):
        rr_raw = np.where(
            (hz_vx > 0) & ~np.isnan(hz_vx) & ~np.isnan(hz_uvx),
            hz_uvx / hz_vx,
            np.nan
        )

    # Hide RR values with insufficient person-days
    total_pop = pop_vx + pop_uvx
    rr_raw[total_pop < HIDELIM] = np.nan

    print(f"Age {age}: RR raw (first 5): {rr_raw[:5]}")

    # Smooth RR values using centered rolling median
    rr_smooth = pd.Series(rr_raw).rolling(window=7, min_periods=1, center=True).median()

    # Add trace for this age
    fig.add_trace(go.Scatter(
        x=days_vx,
        y=rr_smooth,
        mode='lines',
        name=f'Age {age}',
        line=dict(width=1)
    ))

# === Final Plot Styling ===
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

# === Save Plot ===
fig.write_html(OUTPUT_HTML)
print("Plot saved to:", OUTPUT_HTML)
