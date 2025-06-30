import pandas as pd
import numpy as np
from tqdm import tqdm
import sys

# ===============================================================================
# Simulated Deaths and Dose Assignment Script (Minimal Bias)

# The goal of the script is to simulate vaccination and death data incorporating immortal time and selection bias, 
# to test whether methods like Cox regression can correctly adjust for these biases 
# and enable a fair comparison between vaccinated and unvaccinated groups.


# This script performs the following steps:
# 1. Loads Czech vaccination and death records from a CSV file.
# 2. Parses and standardizes date columns, and calculates age from year of birth.
# 3. Estimates age-specific death probabilities based on real observed death rates.
# 4. Simulates random death dates per individual using age-specific death probabilities,
#    so individuals has the same constant death probability,
#    generating random death days uniformly between day 0 and the last observed death day.
# 5. Assigns real-world dose date patterns (dose sets) to individuals in the simulated dataset
#    by randomly selecting recipients from those alive on or after the last
#    dose day in each dose set. Each dose set is assigned to a randomly chosen eligible individual
#    within the same age group who has not already received a dose and for whom the last dose of the set
#    occurs before their simulated death.

# 6. Outputs a fully simulated dataset with internally consistent vaccination and death records.


# Dose assignment details:

#    After random death simulation, everyone is unvaccinated (vax_stat == 0).
#    For each dose sequence per age group:
#        Only unvaccinated and alive (death day ≥ last dose day or no death day) individuals are considered eligible.
#        One eligible individual is randomly assigned that dose sequence.
#        If no eligible candidate is found, the dose sequence remains unassigned (counted as skipped).

# ===============================================================================

# === File paths ===
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"  # Input dataset
OUTPUT_CSV = r"C:\CzechFOI-BUCKET\TERRA\sim_MINBIAS_Vesely_106_202403141131.csv"  # Output path

# === Parameters ===
START_DATE = pd.Timestamp('2020-01-01')  # Day 0 for time reference
REFERENCE_YEAR = 2023                   # Year to calculate age from birth year
MAX_AGE = 113                           # Upper age cap
DOSE_DATE_COLS = [f'Datum_{i}' for i in range(1, 8)]  # Dose date column names
NEEDED_COLS = ['Rok_narozeni', 'DatumUmrti'] + DOSE_DATE_COLS  # Required columns from CSV

# Convert date series to number of days since START_DATE
def to_day_number(date_series):
    return (date_series - START_DATE).dt.days

# Parse date columns safely, setting invalid ones to NaT
def parse_dates(df):
    for col in DOSE_DATE_COLS + ["DatumUmrti"]:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

# Calculate age from birth year
def calculate_age(df):
    df["Age"] = REFERENCE_YEAR - df["Rok_narozeni"].astype(int)
    return df

# Estimate per-age death probability from real data
def estimate_age_death_rates(df):
    grouped = df.groupby("Age")["DatumUmrti"]
    age_rates = {
        age: np.clip(deaths.notna().sum() / len(deaths), 1e-4, 0.999)  # Clamp values for numerical stability
        for age, deaths in grouped
    }
    return age_rates

# Simulate unconditional death dates using estimated death rates
def simulate_deaths_unconditionally(df, end_measure, age_death_rates):
    df = df.copy()
    df['DatumUmrti'] = pd.NaT      # Reset death date
    df['death_day'] = np.nan       # Death as day number
    np.random.seed(42)             # Fixed seed for reproducibility

    for age, group in tqdm(df.groupby("Age"), desc="Simulating deaths", file=sys.stdout):
        idx = group.index
        n = len(group)
        death_rate = age_death_rates.get(age, 0.01)  # Default to 1% if unknown
        will_die = np.random.rand(n) < death_rate
        death_days = np.full(n, np.nan)
        death_days[will_die] = np.random.randint(0, end_measure + 1, size=will_die.sum())

        # Assign death days and convert to actual dates
        df.loc[idx, "death_day"] = death_days
        df.loc[idx[will_die], "DatumUmrti"] = pd.to_datetime(START_DATE) + pd.to_timedelta(death_days[will_die], unit='D')

    return df

# Assign dose dates to simulated people with minimal bias by age group
def assign_doses_per_age(age, dose_sets, death_day_arr, rng_seed):
    rng = np.random.default_rng(rng_seed)
    updates = []

    # Skip if no data available
    if len(dose_sets) == 0 or len(death_day_arr) == 0:
        return age, updates

    vax_stat_arr = np.zeros(len(death_day_arr), dtype=np.int8)  # Track vaccination status

    for dose_dates in dose_sets:
        valid_dates = [d for d in dose_dates if pd.notna(d)]
        if not valid_dates:
            continue
        valid_days = np.array(to_day_number(pd.Series(valid_dates)))
        last_dose_day = valid_days.max()

        # Select only alive and unvaccinated individuals
        alive_mask = np.isnan(death_day_arr) | (death_day_arr >= last_dose_day)
        eligible_mask = (vax_stat_arr == 0) & alive_mask

        eligible_indices = np.where(eligible_mask)[0]
        if eligible_indices.size == 0:
            continue

        selected_pos = rng.choice(eligible_indices)  # Randomly assign one person
        updates.append((selected_pos, dose_dates))
        vax_stat_arr[selected_pos] = 1  # Mark as vaccinated

    return age, updates

# Full wrapper to assign all doses per age group to the simulated population
def assign_doses_minbias(df, dose_source_df):
    print("\n🔄 Assigning doses with minimal bias...")

    # Collect available dose sets per age from real data
    dose_list_by_age = {
        age: [list(row) for row in dose_source_df.loc[dose_source_df["Age"] == age, DOSE_DATE_COLS].dropna(how='all').values]
        for age in range(MAX_AGE + 1)
    }

    total_doses = sum(len(sets) for sets in dose_list_by_age.values())
    print(f"📦 Total real dose sets available: {total_doses}")

    # Reset dose columns and vaccination status
    for col in DOSE_DATE_COLS:
        df[col] = pd.NaT
    df["vax_stat"] = 0

    df.sort_values("Age", inplace=True)  # Optimize group access

    # Create helper maps for fast access
    age_indices_map = {age: df[df["Age"] == age].index.to_numpy() for age in range(MAX_AGE + 1)}
    death_days_map = {age: df.loc[idx, "death_day"].to_numpy() for age, idx in age_indices_map.items()}

    updates_all = {}
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

    # Apply dose assignments to the main DataFrame
    for age, updates in updates_all.items():
        idx_arr = age_indices_map[age]
        for pos, dose_dates in updates:
            row_idx = idx_arr[pos]
            for i, date in enumerate(dose_dates):
                if pd.notna(date):
                    df.at[row_idx, DOSE_DATE_COLS[i]] = date
            df.at[row_idx, "vax_stat"] = 1

    return df

# === Main Execution ===
def main():
    print("📥 Loading data...")
    df = pd.read_csv(INPUT_CSV, usecols=NEEDED_COLS, dtype=str)
    df = parse_dates(df)
    df = calculate_age(df)
    df = df[df['Age'].between(0, MAX_AGE)].copy()  # Keep only valid ages

    print("📏 Calculating max measure day...")
    df["death_day"] = to_day_number(df["DatumUmrti"])
    end_measure = int(df["death_day"].dropna().max())
    print(f"📈 Max day (end_measure): {end_measure}")

    print("🧮 Estimating death probabilities by age...")
    age_death_rates = estimate_age_death_rates(df)

    print("☠️ Simulating deaths...")
    df_sim = simulate_deaths_unconditionally(df, end_measure, age_death_rates)

    print("💉 Assigning doses from reference data...")
    df_final = assign_doses_minbias(df_sim, df)

    print("🧼 Formatting dates...")
    for col in DOSE_DATE_COLS + ['DatumUmrti']:
        df_final[col] = pd.to_datetime(df_final[col], errors='coerce').dt.strftime('%Y-%m-%d').fillna('')

    print(f"💾 Saving to: {OUTPUT_CSV}")
    df_final.to_csv(OUTPUT_CSV, index=False)
    print("\n🎉 All done!")

# Run script
if __name__ == "__main__":
    main()