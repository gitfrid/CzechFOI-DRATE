import pandas as pd
import numpy as np
from lifelines import CoxTimeVaryingFitter
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt


# G-Estimation of Causal Effect for Vaccination on Survival Using Cox Proportional Hazards Model

# This script loads interval-censored survival data for age 70, 
# prepares time-varying covariates, and performs G-estimation by 
# transforming interval lengths to find the causal effect (psi) of vaccination.

# The core idea is to adjust time intervals to remove the vaccination effect,
# then fit a Cox model to identify the psi that minimizes the absolute vaccination coefficient.

# Finally, it plots the absolute vaccination coefficient as a function of psi 
# to visualize the optimization landscape around the estimated causal effect.

# Lecture slides and notes:
#  Many universities provide lecture notes on G-estimation, for example:
#  https://www.biostat.washington.edu/sites/default/files/files/courses/594/G-Estimation.pdf


# Tutorial and overview on G-estimation and structural nested models:
#  https://www.stat.columbia.edu/~gelman/arm/examples/robins/robins.pdf
#  (This is a well-cited tutorial by Robins and colleagues.)


# === 1. Load your interval dataset for age 70 ===
# Path to CSV file containing interval data with columns: person_id, start_day, end_day, dose_number, etc.
intervals_csv = r"C:\CzechFOI-DRATE\intervals_per_agebin\real_intervals_age_70.csv"
df = pd.read_csv(intervals_csv)

# === 2. Clean and prepare data ===
# Remove rows with missing essential data
df.dropna(subset=['person_id', 'start_day', 'end_day', 'dose_number'], inplace=True)

# Compute the day of death per person as the maximum end_day observed
df['death_day'] = df.groupby('person_id')['end_day'].transform('max')

# Create event indicator: 1 if this interval ends at death, else 0
df['event'] = (df['end_day'] == df['death_day']).astype(int)

# Define time-varying vaccination status: vaccinated if dose_number > 0 in the interval
df['vaccinated'] = (df['dose_number'] > 0).astype(int)

# Fix zero-length intervals for death events by extending end_day slightly if start_day == end_day
mask = (df['start_day'] == df['end_day']) & (df['event'] == 1)
df.loc[mask, 'end_day'] = df.loc[mask, 'end_day'] + 0.5

# Retain only columns necessary for Cox model fitting
needed_cols = ['person_id', 'start_day', 'end_day', 'event', 'vaccinated']
df = df[needed_cols]

print(f"Data shape after cleaning: {df.shape}")

# === 3. Define G-estimation function to find causal effect psi ===
def g_estimate_cox(df):
    
    # Finds psi by minimizing the absolute Cox regression coefficient
    # for vaccination after adjusting interval lengths by psi.

    # Args:
    #    df (pd.DataFrame): interval survival data with vaccination indicator.

    # Returns:
    #    float: estimated causal effect psi.
    
    def objective(psi):
    
        # Create a copy of the data for transformation
        df_adj = df.copy()

        # Adjust interval lengths by psi only if vaccinated (exponentially scale the interval length)
        df_adj['end_day_adj'] = df_adj['start_day'] + (df_adj['end_day'] - df_adj['start_day']) * np.exp(-psi * df_adj['vaccinated'])

        # Fix zero-length intervals for death events after transformation
        mask = (df_adj['start_day'] == df_adj['end_day_adj']) & (df_adj['event'] == 1)
        df_adj.loc[mask, 'end_day_adj'] += 0.5

        # Fit Cox model with time-varying intervals
        ctv = CoxTimeVaryingFitter()
        try:
            ctv.fit(df_adj, id_col='person_id', start_col='start_day', stop_col='end_day_adj', event_col='event', show_progress=False)
            coef = ctv.params_.get('vaccinated', 0)  # Extract coefficient for vaccination
        except Exception as e:
            print(f"Fit failed for psi={psi}: {e}")
            return np.inf  # Return large value to avoid selecting failed fits

        # Objective: minimize absolute vaccination effect coefficient
        return abs(coef)

    print("Starting G-estimation optimization...")
    # Optimize psi in the bounded interval [-5, 5] with high precision tolerance
    res = minimize_scalar(objective, bounds=(-5, 5), method='bounded', options={'xatol':1e-4})
    psi_hat = res.x
    print(f"Estimated causal effect psi: {psi_hat:.4f}")
    return psi_hat

# === 4. Run G-estimation ===
psi_hat = g_estimate_cox(df)

# === 5. Visualize how the absolute vaccination coefficient changes around the estimated psi ===
psis = np.linspace(psi_hat - 2, psi_hat + 2, 50)  # range of psi values near estimated psi
objective_vals = []

for psi in psis:
    # Adjust intervals for each psi and fit Cox model
    objective_vals.append(
        abs(CoxTimeVaryingFitter().fit(
            df.assign(end_day_adj=df['start_day'] + (df['end_day'] - df['start_day']) * np.exp(-psi * df['vaccinated'])),
            id_col='person_id', start_col='start_day', stop_col='end_day_adj', event_col='event', show_progress=False
        ).params_.get('vaccinated', 0))
    )

# Plot absolute vaccination coefficient vs psi
plt.figure(figsize=(8,5))
plt.plot(psis, objective_vals, label='|Cox coef for vaccinated|')
plt.axvline(psi_hat, color='red', linestyle='--', label=f'Estimated psi = {psi_hat:.3f}')
plt.xlabel('Psi (causal effect)')
plt.ylabel('Absolute Vaccination Coefficient')
plt.title('G-estimation: Finding psi to Remove Treatment Effect')
plt.legend()
plt.grid(True)
plt.show()
