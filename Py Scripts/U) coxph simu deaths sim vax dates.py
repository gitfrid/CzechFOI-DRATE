import numpy as np
import pandas as pd
from lifelines import CoxTimeVaryingFitter

# Simulated Time-Dependent Cox Analysis of Vaccination Effect
# -----------------------------------------------------------
# This script simulates a time-to-event (survival) dataset for 5000 individuals and performs
# a Cox proportional hazards regression with vaccination as a **time-dependent covariate**.
#
# Key Features:
# - Simulates random death times with a constant daily hazard.
# - Randomly assigns vaccination dates (for 50% of individuals), independent of death.
# - Uses `lifelines`' time-varying Cox model to estimate the effect of vaccination on survival.
# - Outputs hazard ratio and confidence intervals for the vaccinated vs. unvaccinated period.
#
# This example demonstrates proper data structuring (long format) for time-varying covariate analysis.

np.random.seed(42)
N = 5000
MAX_FOLLOWUP = 1080
TRUE_DEATH_HAZARD = 0.001  # tägliche Sterbewahrscheinlichkeit

# --- Simuliere Überlebenszeit (Exponentielle Verteilung) ---
death_times = np.random.exponential(scale=1/TRUE_DEATH_HAZARD, size=N)
death_times = np.minimum(death_times, MAX_FOLLOWUP)
observed = death_times < MAX_FOLLOWUP

# --- Simuliere Impfzeit unabhängig vom Tod (z.B. ab Tag 50 bis 400, oder keine Impfung) ---
vac_times = np.full(N, np.inf)
for i in range(N):
    if np.random.rand() < 0.5:  # 50% Impfquote
        # Impfung nur möglich bis max Tod oder 400 Tage
        vac_times[i] = np.random.uniform(50, min(death_times[i], 400))

vaccinated = vac_times < np.inf

# --- Erzeuge long-Format für Cox mit zeitabhängiger Impfung ---
rows = []
for i in range(N):
    death_time = death_times[i]
    event = int(observed[i])
    vac_time = vac_times[i]

    if not vaccinated[i]:
        rows.append({"id": i, "start": 0, "stop": death_time, "event": event, "vaccinated": 0})
    else:
        # Vor Impfung (ohne Ereignis)
        if vac_time < death_time:
            rows.append({"id": i, "start": 0, "stop": vac_time, "event": 0, "vaccinated": 0})
            rows.append({"id": i, "start": vac_time, "stop": death_time, "event": event, "vaccinated": 1})
        else:
            # Impfung nach Tod (eigentlich nicht möglich, aber zur Sicherheit)
            rows.append({"id": i, "start": 0, "stop": death_time, "event": event, "vaccinated": 0})

tv_df = pd.DataFrame(rows)
tv_df = tv_df.sort_values(by=["id", "start"]).reset_index(drop=True)

# --- Fit Cox-Modell ---
ctv = CoxTimeVaryingFitter()
ctv.fit(tv_df, id_col="id", start_col="start", stop_col="stop", event_col="event")

print(ctv.summary)
