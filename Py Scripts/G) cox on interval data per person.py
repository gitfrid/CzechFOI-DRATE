import pandas as pd
from lifelines import CoxTimeVaryingFitter

# === 1. Load your interval dataset for age 70 ===
intervals_csv = r"C:\CzechFOI-DRATE\intervals_per_agebin\real_intervals_age_70.csv"
df = pd.read_csv(intervals_csv)

# === 2. Clean and prepare data ===
df.dropna(subset=['person_id', 'start_day', 'end_day', 'dose_number'], inplace=True)

# Compute event indicator (death)
df['death_day'] = df.groupby('person_id')['end_day'].transform('max')
df['event'] = (df['end_day'] == df['death_day']).astype(int)

# Define time-varying vaccination status
df['vaccinated'] = (df['dose_number'] > 0).astype(int)

# Fix zero-length intervals if death occurs at entry
mask = (df['start_day'] == df['end_day']) & (df['event'] == 1)
df.loc[mask, 'end_day'] = df.loc[mask, 'end_day'] + 0.5

# Keep only needed columns for Cox
needed_cols = ['person_id', 'start_day', 'end_day', 'event', 'vaccinated']
df = df[needed_cols]

print(f"Data shape after cleaning: {df.shape}")

# === 3. Fit Cox proportional hazards model (time-varying) ===
ctv = CoxTimeVaryingFitter()
ctv.fit(df, id_col='person_id', start_col='start_day', stop_col='end_day', event_col='event', show_progress=True)

print("\nCox model summary:")
print(ctv.summary)

# Calculate hazard ratio for vaccination
hr = ctv.hazard_ratios_.get('vaccinated', None)
if hr is not None:
    print(f"\nObserved Hazard Ratio (vaccinated vs unvaccinated): {hr:.4f}")
else:
    print("Vaccination covariate not found in model parameters.")
