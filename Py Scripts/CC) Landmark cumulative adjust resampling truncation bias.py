import pandas as pd
import numpy as np
import plotly.graph_objs as go

# ==============================================================================
# Title: Landmark Adjustment for Resampling Truncation Bias (Immortal Time Bias)
# Description:
#   This script applies a textbook correction for immortal time bias using the
#   "landmark method". It aligns vaccinated (VX) and unvaccinated (UVX) groups
#   based on the median last vaccine dose date per age group. The script then
#   computes weekly relative risks (RR = UVX/VX) of death for each age group,
#   smoothing and visualizing these RR curves to compare mortality risk.
# 
# Approach:
#   - Calculate 'landmark' as median last dose day per age group (VX only).
#   - Align data to this landmark, excluding events before landmark to remove
#     immortal time bias.
#   - Calculate hazards (death rates) in sliding windows after landmark.
#   - Compute RR of UVX vs VX hazard.
#   - Smooth RR curves and plot all age groups on one figure.
# 
# Output:
#   - Interactive Plotly HTML file with RR curves per age group.
# ==============================================================================

# === File Paths ===
# INPUT_CSV points to the CSV file with individual data including birth year,
# death date, and up to 7 vaccine dose dates.
# OUTPUT_HTML is the path where the interactive plot HTML will be saved.
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
OUTPUT_HTML = r"C:\github\CzechFOI-DRATE\Plot Results\CC) cumulative Landmark adjust resampling truncation bias\CC) nobias CUM_RR_AGEBIN_Landmark_Comparison.html"

# === Parameters ===
START_DATE = pd.Timestamp('2020-01-01')  # Reference start date for day calculations
REFERENCE_YEAR = 2023                    # Year to calculate age from birth year
MAX_AGE = 113                           # Maximum age to consider
WINDOW_SIZE = 30                        # Window size (days) for hazard aggregation
HIDELIM = 10                           # Minimum population at risk threshold for RR

# === Load Data ===
# Read selected columns, parsing date columns as datetime objects
dose_cols = [f"Datum_{i}" for i in range(1, 8)]  # Dose date columns 1-7
cols = ['Rok_narozeni', 'DatumUmrti'] + dose_cols

df = pd.read_csv(INPUT_CSV, usecols=cols, parse_dates=['DatumUmrti'] + dose_cols, dayfirst=False)

# Standardize column names to lowercase and strip whitespace
df.columns = df.columns.str.lower().str.strip()
dose_cols = [c.lower() for c in dose_cols]  # Lowercase dose columns list

# === Preprocessing ===
# Calculate age based on birth year relative to REFERENCE_YEAR
df['birth_year'] = pd.to_numeric(df['rok_narozeni'], errors='coerce')
df['age'] = REFERENCE_YEAR - df['birth_year']

# Keep only valid ages within the range 0 to MAX_AGE
df = df[df['age'].between(0, MAX_AGE)].copy()

# Identify vaccinated individuals (at least one non-NA dose date)
df['is_vaxed'] = df[dose_cols].notna().any(axis=1).astype(int)

# Helper function to convert dates to integer days since START_DATE
def to_day(d):
    return (d - START_DATE).dt.days

# Convert death date and dose dates to days since START_DATE
df['death_day'] = to_day(df['datumumrti'])
for c in dose_cols:
    df[c + '_day'] = to_day(df[c])

# Calculate the last dose day per individual (max of all dose days)
df['last_dose_day'] = df[[c + '_day' for c in dose_cols]].max(axis=1, skipna=True)

# Remove records with missing death day (exclude those without death date)
df = df.dropna(subset=['death_day'])

# Split data into vaccinated (VX) and unvaccinated (UVX) groups
vx = df[df['is_vaxed'] == 1].copy()
uvx = df[df['is_vaxed'] == 0].copy()

# === Function to calculate aggregated hazard rate in windows ===
def aggregated_hazard(df_, start, end, window=7):
    """
    Calculate hazard rate in windows between start and end days.
    
    Parameters:
      df_    - DataFrame with 'entry', 'exit', and 'death_day' columns.
      start  - Start day for hazard calculation.
      end    - End day for hazard calculation.
      window - Window size in days for aggregation.
      
    Returns:
      centers - Center day of each window.
      hz      - Hazard rate for each window (events / at risk).
      counts  - Number at risk in each window.
    """
    bins = np.arange(start, end + 1, window)  # Create window bins
    hz = []      # Store hazard rates
    centers = [] # Store window center days
    counts = []  # Store population at risk counts
    
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1] - 1
        
        # Determine individuals at risk during the window (entry <= window end and exit >= window start)
        at_risk = (df_['entry'] <= bin_end) & (df_['exit'] >= bin_start)
        n_at_risk = at_risk.sum()
        
        # Count death events occurring in the window among those at risk
        n_events = ((df_['death_day'] >= bin_start) & (df_['death_day'] <= bin_end) & at_risk).sum()
        
        # Calculate hazard rate, handle zero risk population
        hz.append(n_events / n_at_risk if n_at_risk > 0 else np.nan)
        centers.append((bin_start + bin_end) // 2)
        counts.append(n_at_risk)
        
    return np.array(centers), np.array(hz), np.array(counts)

# Initialize Plotly figure for RR curves
fig = go.Figure()

# === Prepare for global landmark and end_measure calculation ===
all_death_days_after_landmark = []  # Collect death days across all ages after landmark
landmarks = []                     # Collect landmark days per age

# First pass: Calculate landmarks and gather death days after landmark for all ages
for age in range(0, MAX_AGE + 1):
    vx_age = vx[vx['age'] == age].copy()
    uvx_age = uvx[uvx['age'] == age].copy()

    # Skip ages with too few vaccinated or unvaccinated individuals
    if len(vx_age) < 20 or len(uvx_age) < 20:
        print(f"Age {age}: Skipped due to small group size (VX: {len(vx_age)}, UVX: {len(uvx_age)})")
        continue

    # Calculate median last dose day (landmark) for vaccinated group of this age
    landmark = vx_age['last_dose_day'].median()
    if pd.isna(landmark):
        print(f"Age {age}: Skipped due to NaN landmark")
        continue
    landmark = int(landmark)
    landmarks.append(landmark)

    # Filter death days to those occurring after the landmark for VX and UVX
    vx_a = vx_age[vx_age['death_day'] >= landmark].copy()
    uvx_a = uvx_age[uvx_age['death_day'] >= landmark].copy()

    # Skip if no data after landmark for either group
    if len(vx_a) == 0 or len(uvx_a) == 0:
        print(f"Age {age}: Skipped due to no data after landmark alignment")
        continue

    # Collect death days for global max calculation later
    all_death_days_after_landmark.append(vx_a['death_day'])
    all_death_days_after_landmark.append(uvx_a['death_day'])

# Determine global last death day (end_measure) after all landmarks across ages
end_measure = pd.concat(all_death_days_after_landmark).max()
print(f"Global end_measure (last death day after landmark): {end_measure}")

# Determine minimum landmark (x-axis start point) across all ages
min_landmark = min(landmarks)
print(f"Minimum landmark (start of x-axis): {min_landmark}")

# === Second pass: Calculate hazards and relative risks per age and plot ===
for age in range(0, MAX_AGE + 1):
    vx_age = vx[vx['age'] == age].copy()
    uvx_age = uvx[uvx['age'] == age].copy()

    # Skip small groups again for consistency
    if len(vx_age) < 20 or len(uvx_age) < 20:
        continue

    landmark = vx_age['last_dose_day'].median()
    if pd.isna(landmark):
        continue
    landmark = int(landmark)

    # Filter data to death events after landmark
    vx_a = vx_age[vx_age['death_day'] >= landmark].copy()
    uvx_a = uvx_age[uvx_age['death_day'] >= landmark].copy()

    if len(vx_a) == 0 or len(uvx_a) == 0:
        continue

    # Define maximum day for this age as min of group's max death day and global end_measure
    age_end_day = min(pd.concat([vx_a['death_day'], uvx_a['death_day']]).max(), end_measure)

    # Define entry and exit times for survival interval after landmark for hazard calculation
    vx_a['entry'] = landmark
    uvx_a['entry'] = landmark
    vx_a['exit'] = vx_a['death_day'].clip(upper=age_end_day)
    uvx_a['exit'] = uvx_a['death_day'].clip(upper=age_end_day)

    # Calculate person-days at risk (time under observation)
    vx_a['pdays'] = vx_a['exit'] - vx_a['entry']
    uvx_a['pdays'] = uvx_a['exit'] - uvx_a['entry']

    # Keep only records with positive follow-up time
    vx_a = vx_a[vx_a['pdays'] > 0]
    uvx_a = uvx_a[uvx_a['pdays'] > 0]

    if len(vx_a) == 0 or len(uvx_a) == 0:
        continue

    # Calculate hazards in sliding windows for VX and UVX groups
    days_vx, hz_vx, pop_vx = aggregated_hazard(vx_a, landmark, age_end_day, window=WINDOW_SIZE)
    days_uvx, hz_uvx, pop_uvx = aggregated_hazard(uvx_a, landmark, age_end_day, window=WINDOW_SIZE)

    # Calculate raw relative risk (UVX hazard / VX hazard), handle divide-by-zero and NaNs gracefully
    with np.errstate(divide='ignore', invalid='ignore'):
        rr_raw = np.where(
            (hz_vx > 0) & ~np.isnan(hz_vx) & ~np.isnan(hz_uvx),
            hz_uvx / hz_vx,
            np.nan
        )

    # Remove RR values where population at risk is below HIDELIM threshold
    total_pop = pop_vx + pop_uvx
    rr_raw[total_pop < HIDELIM] = np.nan

    # Smooth RR curve using a rolling median with window=7, centered
    rr_smooth = pd.Series(rr_raw).rolling(window=7, min_periods=1, center=True).median()

    # Add smoothed RR trace for this age group to the plot
    fig.add_trace(go.Scatter(
        x=days_vx,
        y=rr_smooth,
        mode='lines',
        name=f'Age {age}',
        line=dict(width=1)
    ))

# === Final plot adjustments ===
fig.update_layout(
    title="Relative Risk (UVX / VX) by Age — Landmark = Median Last Dose Day (Per Age)",
    xaxis_title="Day from Age-specific Last Dose Landmark (weekly bins)",
    yaxis_title="Relative Risk (UVX / VX)",
    yaxis_type="linear",
    template="plotly_white",
    height=900,
    width=1400,  # Increased width for better visualization
    legend=dict(itemsizing='constant', bgcolor='rgba(0,0,0,0)'),
    xaxis=dict(range=[min_landmark, end_measure])  # Start x-axis at min landmark to end_measure
)

# Save interactive plot to HTML file
fig.write_html(OUTPUT_HTML)
print("Plot saved to:", OUTPUT_HTML)
