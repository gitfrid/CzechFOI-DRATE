import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

# Simulated Czech Vaccination and Death Dataset Generator (Minimal Bias Version)
# ===============================================================================

# This script creates a fully synthetic dataset of vaccination and death records 
# based on real-world dose patterns and observed age-specific death rates, with 
# the following enhancements:

# Key Features:
# 1. **Real-world Dose Patterns**: Dose date patterns are taken from real individuals 
#    and assigned randomly but fairly to simulated individuals.
# 2. **Death Simulation**:
#    - Deaths are simulated per age group using observed death probabilities.
#    - Exactly 10% of each age group is randomly marked as high-risk.
#    - High-risk individuals have a 10× higher chance of dying.
#    - The combined death rate per age matches the real observed death rate.
# 3. **Minimal Dose Assignment Bias**: Individuals are assigned dose schedules only 
#    if they are alive at the time of the last dose in the set - which inevitable introduces immortality bias

# Input:
# - Vesely_106_202403141131.csv: Original data file with birth year, death dates, and dose date columns.

# === File paths ===
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"  # Input real data
OUTPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_TWO_DEATHRATES_10X_Vesely_106_SIMULATED.csv"  # Output simulated dataset

# === Parameters ===
START_DATE = pd.Timestamp('2020-01-01')  # Reference start date for time calculations
REFERENCE_YEAR = 2023  # Reference year for age calculation
MAX_AGE = 113  # Maximum age considered
DOSE_DATE_COLS = [f'Datum_{i}' for i in range(1, 8)]  # Columns for vaccine dose dates
NEEDED_COLS = ['Rok_narozeni', 'DatumUmrti'] + DOSE_DATE_COLS  # Columns to read from input

# Convert date to number of days since START_DATE
def to_day_number(date_series):
    return (date_series - START_DATE).dt.days

# Parse all relevant columns as datetime
def parse_dates(df):
    for col in DOSE_DATE_COLS + ["DatumUmrti"]:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

# Compute age from birth year and REFERENCE_YEAR
def calculate_age(df):
    df["Age"] = REFERENCE_YEAR - df["Rok_narozeni"].astype(int)
    return df

# Estimate death rate per age group
def estimate_age_death_rates(df):
    grouped = df.groupby("Age")["DatumUmrti"]
    age_rates = {
        age: np.clip(deaths.notna().sum() / len(deaths), 1e-4, 0.999)  # Clip to avoid 0 or 1
        for age, deaths in grouped
    }
    return age_rates

# === ✅ Simulate deaths with age-based high-risk logic ===
def simulate_deaths_unconditionally(df, end_measure, age_death_rates):
    df = df.copy()
    df['DatumUmrti'] = pd.NaT  # Reset death date
    df['death_day'] = np.nan  # Death day in integer form
    np.random.seed(42)  # For reproducibility

    # Process each age group separately
    for age, group in tqdm(df.groupby("Age"), desc="Simulating deaths", file=sys.stdout):
        idx = group.index
        n = len(group)
        base_rate = age_death_rates.get(age, 0.01)
        if n == 0:
            continue

        # Assign 10% of group as high-risk
        high_risk_size = max(1, int(0.10 * n))
        high_risk_indices = np.random.choice(idx, size=high_risk_size, replace=False)
        low_risk_indices = list(set(idx) - set(high_risk_indices))

        # Compute adjusted risk so weighted mean = base_rate
        r = base_rate / 1.9  # Derived from 0.9*r + 0.1*10*r = base_rate
        r_low = r
        r_high = 10 * r

        # Simulate low-risk group
        n_low = len(low_risk_indices)
        will_die_low = np.random.rand(n_low) < r_low  # Random draw for death
        death_days_low = np.full(n_low, np.nan)
        death_days_low[will_die_low] = np.random.randint(0, end_measure + 1, size=will_die_low.sum())
        df.loc[low_risk_indices, "death_day"] = death_days_low
        df.loc[np.array(low_risk_indices)[will_die_low], "DatumUmrti"] = START_DATE + pd.to_timedelta(death_days_low[will_die_low], unit='D')

        # Simulate high-risk group
        n_high = len(high_risk_indices)
        will_die_high = np.random.rand(n_high) < r_high
        death_days_high = np.full(n_high, np.nan)
        death_days_high[will_die_high] = np.random.randint(0, end_measure + 1, size=will_die_high.sum())
        df.loc[high_risk_indices, "death_day"] = death_days_high
        df.loc[np.array(high_risk_indices)[will_die_high], "DatumUmrti"] = START_DATE + pd.to_timedelta(death_days_high[will_die_high], unit='D')

    return df

# Assign doses to individuals in an unbiased way (only if alive at last dose)
def assign_doses_per_age(age, dose_sets, death_day_arr, rng_seed):
    rng = np.random.default_rng(rng_seed)
    updates = []
    if len(dose_sets) == 0 or len(death_day_arr) == 0:
        return age, updates

    vax_stat_arr = np.zeros(len(death_day_arr), dtype=np.int8)  # Track vaccination status

    # Go through real dose sets for this age
    for dose_dates in dose_sets:
        valid_dates = [d for d in dose_dates if pd.notna(d)]
        if not valid_dates:
            continue
        valid_days = np.array(to_day_number(pd.Series(valid_dates)))
        last_dose_day = valid_days.max()

        # Find alive and unvaccinated individuals
        alive_mask = np.isnan(death_day_arr) | (death_day_arr >= last_dose_day)
        eligible_mask = (vax_stat_arr == 0) & alive_mask
        eligible_indices = np.where(eligible_mask)[0]

        if eligible_indices.size == 0:
            continue

        # Randomly select one eligible individual and assign this dose set
        selected_pos = rng.choice(eligible_indices)
        updates.append((selected_pos, dose_dates))
        vax_stat_arr[selected_pos] = 1  # Mark as vaccinated

    return age, updates

# Loop over age groups and assign real dose patterns fairly
def assign_doses_minbias(df, dose_source_df):
    print("\n🔄 Assigning doses with minimal bias...")

    # Group real dose patterns by age
    dose_list_by_age = {
        age: [list(row) for row in dose_source_df.loc[dose_source_df["Age"] == age, DOSE_DATE_COLS].dropna(how='all').values]
        for age in range(MAX_AGE + 1)
    }

    total_doses = sum(len(sets) for sets in dose_list_by_age.values())
    print(f"📦 Total real dose sets available: {total_doses}")

    # Initialize vaccine columns
    for col in DOSE_DATE_COLS:
        df[col] = pd.NaT
    df["vax_stat"] = 0

    # Sort and map individuals by age
    df.sort_values("Age", inplace=True)
    age_indices_map = {age: df[df["Age"] == age].index.to_numpy() for age in range(MAX_AGE + 1)}
    death_days_map = {age: df.loc[idx, "death_day"].to_numpy() for age, idx in age_indices_map.items()}
    updates_all = {}

    # Assign doses per age
    for age in tqdm(range(MAX_AGE + 1), desc="Assigning doses per age", file=sys.stdout):
        if len(death_days_map[age]) == 0:
            updates_all[age] = []
            continue
        age_val, updates = assign_doses_per_age(age, dose_list_by_age[age], death_days_map[age], 42 + age)
        updates_all[age_val] = updates

    print(f"\n✅ Dose assignment completed.")
    assigned_count = sum(len(upd) for upd in updates_all.values())
    print(f"  → Assigned: {assigned_count}")
    print(f"  → Skipped: {total_doses - assigned_count} (no eligible candidate)")

    # Apply dose updates to the dataframe
    for age, updates in updates_all.items():
        idx_arr = age_indices_map[age]
        for pos, dose_dates in updates:
            row_idx = idx_arr[pos]
            for i, date in enumerate(dose_dates):
                if pd.notna(date):
                    df.at[row_idx, DOSE_DATE_COLS[i]] = date
            df.at[row_idx, "vax_stat"] = 1

    return df

# === Main program entry point ===
def main():
    print("📥 Loading data...")
    df = pd.read_csv(INPUT_CSV, usecols=NEEDED_COLS, dtype=str)  # Load input
    df = parse_dates(df)  # Convert date columns
    df = calculate_age(df)  # Compute age
    df = df[df['Age'].between(0, MAX_AGE)].copy()  # Filter valid age range

    print("📏 Calculating max measure day...")
    df["death_day"] = to_day_number(df["DatumUmrti"])  # Convert death date to int
    end_measure = int(df["death_day"].dropna().max())  # Last observed death day
    print(f"📈 Max day (end_measure): {end_measure}")

    print("🧮 Estimating death probabilities by age...")
    age_death_rates = estimate_age_death_rates(df)  # Death probabilities

    print("☠️ Simulating deaths...")
    df_sim = simulate_deaths_unconditionally(df, end_measure, age_death_rates)

    print("💉 Assigning doses from reference data...")
    df_final = assign_doses_minbias(df_sim, df)  # Assign vaccine dates

    print("🧼 Formatting dates...")
    for col in DOSE_DATE_COLS + ['DatumUmrti']:
        df_final[col] = pd.to_datetime(df_final[col], errors='coerce').dt.strftime('%Y-%m-%d').fillna('')

    print(f"💾 Saving to: {OUTPUT_CSV}")
    df_final.to_csv(OUTPUT_CSV, index=False)  # Save to output CSV
    print("\n🎉 All done!")

# Execute main when script runs
if __name__ == "__main__":
    main()
