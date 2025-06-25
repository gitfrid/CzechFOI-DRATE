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

OUTPUT_HTML = r"C:\github\CzechFOI-DRATE\Plot Results\CC) cumulative Landmark adjust resampling truncation bias\CA) minbias RR_AGEBIN_Landmark_Comparison.html"

# === Parameters ===
START_DATE = pd.Timestamp('2020-01-01')
REFERENCE_YEAR = 2023
MAX_AGE = 113
WINDOW_SIZE = 30
HIDELIM = 10  # hide small risk sets

# === Load & Preprocess ===
dose_cols = [f"Datum_{i}" for i in range(1, 8)]
cols = ['Rok_narozeni', 'DatumUmrti'] + dose_cols
df = pd.read_csv(
    INPUT_CSV,
    usecols=cols,
    parse_dates=['DatumUmrti'] + dose_cols,
    dayfirst=False
)
df.columns = df.columns.str.lower().str.strip()
dose_cols = [c.lower() for c in dose_cols]

df['birth_year'] = pd.to_numeric(df['rok_narozeni'], errors='coerce')
df['age'] = REFERENCE_YEAR - df['birth_year']
df = df[df['age'].between(0, MAX_AGE)].copy()
df['is_vaxed'] = df[dose_cols].notna().any(axis=1).astype(int)

def to_day(d): return (d - START_DATE).dt.days
df['death_day'] = to_day(df['datumumrti'])
for c in dose_cols:
    df[c + '_day'] = to_day(df[c])

df['last_dose_day'] = df[[c + '_day' for c in dose_cols]].max(axis=1, skipna=True)
df = df.dropna(subset=['death_day'])

# --- Split groups ---
vx = df[df['is_vaxed'] == 1].copy()
uvx = df[df['is_vaxed'] == 0].copy()

# --- Aggregated Hazard Function ---
def aggregated_hazard(df_, start, end, window):
    bins = np.arange(start, end + 1, window)
    hz, centers, counts = [], [], []
    for i in range(len(bins) - 1):
        b0, b1 = bins[i], bins[i+1] - 1
        at_risk = (df_['entry'] <= b1) & (df_['exit'] >= b0)
        n0 = at_risk.sum()
        n1 = ((df_['death_day'] >= b0) & (df_['death_day'] <= b1) & at_risk).sum()
        hz.append(n1 / n0 if n0 > 0 else np.nan)
        centers.append((b0 + b1)//2)
        counts.append(n0)
    return np.array(centers), np.array(hz), np.array(counts)

# --- Plotting cumulative incidence ---
fig = go.Figure()

for age in range(MAX_AGE+1):
    vx_age = vx[vx['age']==age]
    uvx_age = uvx[uvx['age']==age]
    if len(vx_age)<20 or len(uvx_age)<20:
        continue

    lm = int(vx_age['last_dose_day'].median())
    vx_a = vx_age[vx_age['death_day']>=lm].copy()
    uvx_a = uvx_age[uvx_age['death_day']>=lm].copy()
    end_day = int(pd.concat([vx_a['death_day'], uvx_a['death_day']]).max())

    for grp, df_grp, name, dash in [
        ('VX', vx_a, f'VX Age {age}', 'solid'),
        ('UVX', uvx_a, f'UVX Age {age}', 'dash'),
    ]:
        df_grp = df_grp.copy()
        df_grp['entry'] = lm
        df_grp['exit']  = df_grp['death_day'].clip(upper=end_day)
        df_grp['pdays'] = df_grp['exit'] - df_grp['entry']
        df_grp = df_grp[df_grp['pdays']>0]
        if df_grp.empty: 
            continue

        days, hz, counts = aggregated_hazard(df_grp, lm, end_day, WINDOW_SIZE)
        # mask small sets
        hz[counts < HIDELIM] = np.nan

        # cumulative hazard and incidence
        cum_hz = np.nancumsum(hz * WINDOW_SIZE)
        ci     = 1 - np.exp(-cum_hz)

        fig.add_trace(go.Scatter(
            x=days, y=ci,
            mode='lines', name=name,
            line=dict(dash=dash, width=1)
        ))

fig.update_layout(
    title="Cumulative Incidence by Age Group (Landmark = Median Last Dose Day)",
    xaxis_title=f"Days since Landmark (bins of {WINDOW_SIZE}d)",
    yaxis_title="Cumulative Incidence",
    template="plotly_white",
    height=700, width=1000,
    legend=dict(font=dict(size=8), traceorder='normal')
)

fig.write_html(OUTPUT_HTML)
print("Plot saved to:", OUTPUT_HTML)
