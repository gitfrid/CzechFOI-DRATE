import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px

# SCRIPT A - Aggregate raw daily data by age group and vaccination status
#    - Loads and cleans individual-level death and vaccine dose dates from raw data
#    - Calculates each person’s age at the reference date (e.g., start of observation)
#    - Determines vaccination status at time of death (vaccinated/unvaccinated)
#    - Constructs the following daily tables (rows = days, columns = age 0–113):
#        1) Total deaths
#        2) Deaths among vaccinated individuals
#        3) Deaths among unvaccinated individuals
#        4) Total population by age (constant over time)
#        5) Daily new vaccinated individuals (first dose)
#        6) Daily decrease in unvaccinated individuals (equal to daily new vaccinated, negated)
#        7) All administered doses and first doses (daily counts)
#    - Saves all tables in CSV format for downstream plotting or analysis

# === CONFIG ===
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_NOBIAS_DEATHRISK_2X_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINIMALBIAS_Vesely_106_202403141131.csv"
# INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINIMALBIAS_DEATHRISK_10X_Vesely_106_202403141131.csv"
INPUT_CSV = r"C:\CzechFOI-DRATE\TERRA\sim_MINBIAS_Vesely_106_SIMULATED - Kopie.csv"

OUTPUT_DIR = r"C:\CzechFOI-DRATE\TERRA"
OUTPUT_HTML = r"C:\CzechFOI-DRATE\Plot Results\X) event_stacking\X) sim MINBIAS DoseAligned_Stacked_Normalized_Deaths"
OUTPUT_HTML1 = r"C:\CzechFOI-DRATE\Plot Results\X) event_stacking\X) sim MINBIAS Population_and_Deaths_Trends_with_All_Doses"
MAX_AGE = 113
MAX_DAYS = 1533
START_DATE = pd.to_datetime("2020-01-01")
AGE_REFERENCE_DATE = pd.to_datetime("2023-01-01")
DOSE_COLUMNS = [f"Datum_{i}" for i in range(1, 8)]
IMMUNITY_LAG_DAYS = 0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(INPUT_CSV, parse_dates=["DatumUmrti"] + DOSE_COLUMNS, low_memory=False)

# Exclude individuals who died before their first dose (if they had one)
dose_given = df[DOSE_COLUMNS[0]].notna()
death_before_first_dose = df["DatumUmrti"] < df[DOSE_COLUMNS[0]]
exclude_mask = dose_given & death_before_first_dose

# Filter them out — they did not live long enough to be vaccinated
df = df[~exclude_mask].reset_index(drop=True)
print(f"Excluded {exclude_mask.sum()} individuals who died before receiving their first dose.")

# === COMPUTE AGE AT START ===
df['AgeAtStart'] = (AGE_REFERENCE_DATE.year - df['Rok_narozeni']).clip(0, MAX_AGE).astype(int)

# === DEATH DAY RELATIVE TO START_DATE ===
df['DeathDay'] = (df['DatumUmrti'] - START_DATE).dt.days.clip(lower=0, upper=MAX_DAYS)

# === FIRST DOSE DAY WITH IMMUNITY LAG ===
first_dose_day = (df[DOSE_COLUMNS[0]] - START_DATE).dt.days.fillna(MAX_DAYS + 1).astype(int)
first_dose_day_lagged = first_dose_day + IMMUNITY_LAG_DAYS

# === PREPARE ARRAYS ===
N = len(df)
death_day = df['DeathDay'].values
age = df['AgeAtStart'].values
first_dose = first_dose_day_lagged.values

# === TOTAL (CONSTANT) POPULATION BY AGE ===
pop_total_by_age = np.bincount(age, minlength=MAX_AGE + 1)
population_total = np.tile(pop_total_by_age, (MAX_DAYS + 1, 1))

# === CUMULATIVE FIRST-DOSE VACCINATED BY AGE ===
vx_day = first_dose_day.values.clip(0, MAX_DAYS)
vx_hist = np.zeros((MAX_DAYS + 1, MAX_AGE + 1), dtype=int)
np.add.at(vx_hist, (vx_day, age), 1)
population_vx = np.cumsum(vx_hist, axis=0)

# === UNVACCINATED = TOTAL - VACCINATED ===
population_uvx = population_total - population_vx

# === DAILY CHANGES IN VACCINATED AND UNVACCINATED ===
population_vx_daily = np.diff(population_vx, axis=0, prepend=np.zeros((1, MAX_AGE + 1), dtype=int))
population_uvx_daily = -population_vx_daily

# === DEATH HISTOGRAMS ===
bins_days = np.arange(MAX_DAYS + 2)
bins_age = np.arange(MAX_AGE + 2)

death_hist, _, _ = np.histogram2d(death_day, age, bins=[bins_days, bins_age])

vx_mask = first_dose <= death_day
vx_death_hist, _, _ = np.histogram2d(death_day[vx_mask], age[vx_mask], bins=[bins_days, bins_age])

uvx_death_hist = death_hist - vx_death_hist

# === DOSE HISTOGRAMS ===
all_dose_days = []
first_dose_days = []

for col in DOSE_COLUMNS:
    dose_day = (df[col] - START_DATE).dt.days
    mask = dose_day.between(0, MAX_DAYS)
    all_dose_days.append(np.stack([dose_day[mask], df.loc[mask, 'AgeAtStart']], axis=1))
    if col == DOSE_COLUMNS[0]:
        first_dose_days.append(np.stack([dose_day[mask], df.loc[mask, 'AgeAtStart']], axis=1))

all_dose_arr = np.vstack(all_dose_days)
first_dose_arr = np.vstack(first_dose_days)

all_doses, _, _ = np.histogram2d(all_dose_arr[:, 0], all_dose_arr[:, 1], bins=[bins_days, bins_age])
first_doses, _, _ = np.histogram2d(first_dose_arr[:, 0], first_dose_arr[:, 1], bins=[bins_days, bins_age])

# === SAVE FUNCTION ===
def save_csv(data, filename):
    df_out = pd.DataFrame(data.astype(int), columns=[str(a) for a in range(MAX_AGE + 1)])
    df_out.insert(0, 'DAY', range(data.shape[0]))
    df_out.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)


# Verify that cumulative sum of daily vaccinated matches cumulative vaccinated population
reconstructed_population_vx = np.cumsum(population_vx_daily, axis=0)

if not np.array_equal(reconstructed_population_vx, population_vx):
    print("Warning: population_vx_daily cumulative sum does NOT match population_vx!")
else:
    print("Check passed: population_vx_daily cumulative sum matches population_vx.")

    
# === SAVE OUTPUT FILES ===
print("Saving CSV files...")

save_csv(population_total[:1533], "population_total.csv")
save_csv(population_vx[:1533], "population_vx.csv")
save_csv(population_uvx[:1533], "population_uvx.csv")

# Save daily changes (only days 0 to 1532)
save_csv(population_vx_daily[:1533], "population_vx_daily.csv")
save_csv(population_uvx_daily[:1533], "population_uvx_daily.csv")

save_csv(death_hist[:1533], "deaths_total.csv")
save_csv(vx_death_hist[:1533], "deaths_vx.csv")
save_csv(uvx_death_hist[:1533], "deaths_uvx.csv")

save_csv(all_doses[:1533], "all_doses.csv")
save_csv(first_doses[:1533], "first_doses.csv")

print(f"Script completed successfully with immunity lag = {IMMUNITY_LAG_DAYS}")

# Simulate death data with identical constant death rates across total population group.
# - Randomly assigns deaths over days 0–END_MEASURE per age group using the real total death counts of the age groups.
# - Applies real vaccination schedule from dose_first_df.
# - Classifies randomly each death into vx or uvx based on vaccination timing.
# - Outputs raw daily deaths for validation as csv files: total, vaccinated, and unvaccinated.
# - Plots raw / normalized death traces for total, vaccinated, and unvaccinated group
# - Additional plots raw / normalized stacked event curves for each age group

# Load CSV files
dose_first_df = pd.read_csv(r"C:\CzechFOI-DRATE\TERRA\first_doses.csv").set_index("DAY")
dose_all_df = pd.read_csv(r"C:\CzechFOI-DRATE\TERRA\all_doses.csv").set_index("DAY")
deaths_total_df = pd.read_csv(r"C:\CzechFOI-DRATE\TERRA\deaths_total.csv").set_index("DAY")
deaths_uvx_df = pd.read_csv(r"C:\CzechFOI-DRATE\TERRA\deaths_uvx.csv").set_index("DAY")
deaths_vx_df = pd.read_csv(r"C:\CzechFOI-DRATE\TERRA\deaths_vx.csv").set_index("DAY")
pop_df = pd.read_csv(r"C:\CzechFOI-DRATE\TERRA\population_total.csv").set_index("DAY")


END_MEASURE = 1110
POST_VX_DELAY = 0
sim_extension = ""

# Set a fixed seed for reproducibility
np.random.seed(42)

# --- Dynamically determine dimensions ---
days = pop_df.index.to_numpy()
ages = pop_df.columns.astype(int).to_numpy()

num_days = len(days)
num_ages = len(ages)

# --- Initialize simulation arrays ---
deaths_sim_total = np.zeros((num_days, num_ages), dtype=int)
deaths_sim_vx = np.zeros_like(deaths_sim_total)
deaths_sim_uvx = np.zeros_like(deaths_sim_total)

# --- Simulate per age group ---
for age_idx, age in enumerate(ages):
    pop = int(pop_df.iloc[0, age_idx])
    total_deaths = int(deaths_total_df.iloc[:, age_idx].sum())
    first_dose = dose_first_df.iloc[:, age_idx].to_numpy()

    # Assign random death days
    death_days = np.random.choice(np.arange(END_MEASURE), size=total_deaths, replace=True)
    death_day_counts = np.bincount(death_days, minlength=num_days)
    deaths_sim_total[:, age_idx] = death_day_counts

    # Assign simulated people
    person_ids = np.random.choice(np.arange(pop), size=total_deaths, replace=True)

    # Assign doses based on real schedule
    dose_schedule = np.zeros(num_days, dtype=int)
    dose_schedule[:len(first_dose)] = first_dose
    total_dosed = dose_schedule.sum()

    dose_day_assignments = np.full(pop, -1, dtype=int)  # -1 means not vaccinated
    if total_dosed > 0:
        dose_person_ids = np.random.choice(np.arange(pop), size=total_dosed, replace=False)
        pointer = 0
        for day in range(num_days):
            count = dose_schedule[day]
            if count > 0 and pointer + count <= total_dosed:
                dose_day_assignments[dose_person_ids[pointer:pointer+count]] = day
                pointer += count

    # Classify each death as vx or uvx
    for dday_idx, pid in zip(death_days, person_ids):
        vday_idx = dose_day_assignments[pid]
        if vday_idx != -1 and dday_idx >= vday_idx + POST_VX_DELAY:
            deaths_sim_vx[dday_idx, age_idx] += 1
        else:
            deaths_sim_uvx[dday_idx, age_idx] += 1

# --- Save results ---
index = pop_df.index
columns = pop_df.columns

pd.DataFrame(deaths_sim_total, index=index, columns=columns).to_csv(r"C:\CzechFOI-StackSim\TERRA\PVT_NUM_D_SIM.csv")
pd.DataFrame(deaths_sim_uvx, index=index, columns=columns).to_csv(r"C:\CzechFOI-StackSim\TERRA\PVT_NUM_DUVX_SIM.csv")
pd.DataFrame(deaths_sim_vx, index=index, columns=columns).to_csv(r"C:\CzechFOI-StackSim\TERRA\PVT_NUM_DVX_SIM.csv")

# for test with simulated data uncoment the four lines below
#sim_extension += "_sim"
#deaths_total_df = pd.read_csv(r"C:\CzechFOI-StackSim\TERRA\PVT_NUM_D_SIM.csv").set_index("DAY")
#deaths_uvx_df = pd.read_csv(r"C:\CzechFOI-StackSim\TERRA\PVT_NUM_DUVX_SIM.csv").set_index("DAY")
#deaths_vx_df = pd.read_csv(r"C:\CzechFOI-StackSim\TERRA\PVT_NUM_DVX_SIM.csv").set_index("DAY")


# --- Plotting and Calculations (Rest of the code stays unchanged) ---
# (Use the same plot creation code you already had for the rest of the analysis)

# Use the first row as initial population (constant)
initial_population = pop_df.iloc[0]

# Compute cumulative deaths
cumulative_deaths_total = deaths_total_df.cumsum()
cumulative_deaths_vx = deaths_vx_df.cumsum()
cumulative_deaths_uvx = deaths_uvx_df.cumsum()
cumulative_first_dose = dose_first_df.cumsum()

# Create figure for trends
fig = go.Figure()

# Window size for rolling average
window = 21

# --- Original Plotting Loop ---
for age in range(114):
    age_str = str(age)

    # Extract data for current age
    pop = initial_population[age_str]
    deaths_total = deaths_total_df[age_str]
    deaths_vx = deaths_vx_df[age_str]
    deaths_uvx = deaths_uvx_df[age_str]
    first_dose = dose_first_df[age_str]
    all_dose = dose_all_df[age_str]

    cum_deaths_total = cumulative_deaths_total[age_str]
    cum_deaths_vx = cumulative_deaths_vx[age_str]
    cum_deaths_uvx = cumulative_deaths_uvx[age_str]
    cum_first_dose = cumulative_first_dose[age_str]

    # Remaining pop (pop - deaths) 
    remaining_pop = pop - cum_deaths_total
    vx_pop = cum_first_dose - cum_deaths_vx
    uvx_pop = pop - cum_first_dose - cum_deaths_uvx

    # --- Consistency Checks  -> delete it if you like ---
    valid_mask = (remaining_pop > 0) & (vx_pop > 0) & (uvx_pop > 0)

    # Skip if any population is zero to avoid division errors
    if not valid_mask.any():
        print(f"Skipping age {age} due to zero population")
        continue

    # Normalized deaths per 100k
    norm_deaths_total = (deaths_total / remaining_pop) * 100000
    norm_deaths_vx = (deaths_vx / vx_pop) * 100000
    norm_deaths_uvx = (deaths_uvx / uvx_pop) * 100000

    # --- Normalized Population Check -> delete it if you want ---
    # Skip this check entirely if no valid data remains
    if not valid_mask.any():
        print(f"Skipping normalized check at age {age} due to zero population")
    else:
        sum_norm = norm_deaths_vx * (vx_pop / remaining_pop) + norm_deaths_uvx * (uvx_pop / remaining_pop)
        if not np.allclose(norm_deaths_total[valid_mask], sum_norm[valid_mask], rtol=1e-3, atol=0.1):
            raise ValueError(
                f"Normalized mismatch at age {age}\n"
                f"Remaining pop:\n{remaining_pop}\n"
                f"VX pop:\n{vx_pop}\n"
                f"UVX pop:\n{uvx_pop}\n"
                f"Norm deaths total:\n{norm_deaths_total}\n"
                f"Norm deaths vx:\n{norm_deaths_vx}\n"
                f"Norm deaths uvx:\n{norm_deaths_uvx}\n"
                f"Sum norm:\n{sum_norm}"
            )

    # --- further Checks  -> delete it if you like ---
    assert np.allclose(deaths_total, deaths_vx + deaths_uvx, rtol=1e-6), f"Mismatch in raw deaths for age {age}"
    day = 500
    expected_norm = (deaths_total.iloc[day] / (initial_population[age_str] - cumulative_deaths_total[age_str].iloc[day])) * 100_000
    assert np.isclose(norm_deaths_total.iloc[day], expected_norm, rtol=1e-6), f"Normalization error at age {age}, day {day}"

    # Rolling means
    deaths_total_roll = deaths_total.rolling(window, center=True).mean()
    deaths_vx_roll = deaths_vx.rolling(window, center=True).mean()
    deaths_uvx_roll = deaths_uvx.rolling(window, center=True).mean()
    norm_total_roll = norm_deaths_total.rolling(window, center=True).mean()
    norm_vx_roll = norm_deaths_vx.rolling(window, center=True).mean()
    norm_uvx_roll = norm_deaths_uvx.rolling(window, center=True).mean()
    all_dose_roll = all_dose.rolling(window, center=True).mean()
    
    # Death Traces
    fig.add_trace(go.Scatter(x=deaths_vx.index, y=deaths_vx, mode='lines', name=f"Age {age} VX Deaths", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=deaths_vx.index, y=deaths_vx_roll, mode='lines', name=f"Age {age} VX Deaths ({window}d avg)", line=dict(width=0.8, dash='solid')))
    fig.add_trace(go.Scatter(x=deaths_uvx.index, y=deaths_uvx, mode='lines', name=f"Age {age} UVX Deaths", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=deaths_uvx.index, y=deaths_uvx_roll, mode='lines', name=f"Age {age} UVX Deaths ({window}d avg)", line=dict(width=0.8, dash='solid')))
    fig.add_trace(go.Scatter(x=deaths_total.index, y=deaths_total, mode='lines', name=f"Age {age} Total Deaths", line=dict(width=0.8)))
    fig.add_trace(go.Scatter(x=deaths_total.index, y=deaths_total_roll, mode='lines', name=f"Age {age} Total Deaths ({window}d avg)", line=dict(width=0.8, dash='solid')))
    
    fig.add_trace(go.Scatter(x=deaths_vx.index, y=norm_deaths_vx, mode='lines', name=f"Age {age} Norm VX Deaths", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=deaths_vx.index, y=norm_vx_roll, mode='lines', name=f"Age {age} Norm VX Deaths ({window}d avg)", line=dict(width=0.8, dash='solid')))
    fig.add_trace(go.Scatter(x=deaths_uvx.index, y=norm_deaths_uvx, mode='lines', name=f"Age {age} Norm UVX Deaths", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=deaths_uvx.index, y=norm_uvx_roll, mode='lines', name=f"Age {age} Norm UVX Deaths ({window}d avg)", line=dict(width=0.8, dash='solid')))
    fig.add_trace(go.Scatter(x=deaths_total.index, y=norm_deaths_total, mode='lines', name=f"Age {age} Norm Total Deaths", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=deaths_total.index, y=norm_total_roll, mode='lines', name=f"Age {age} Norm Total Deaths ({window}d avg)", line=dict(width=0.8, dash='solid')))
    
    # Dose and population traces
    fig.add_trace(go.Scatter(x=deaths_total.index, y=remaining_pop, mode='lines', name=f"Age {age} Remaining Total Pop", line=dict(width=0.8), yaxis="y2" ))
    fig.add_trace(go.Scatter(x=all_dose.index, y=vx_pop, mode='lines', name=f"Age {age} VX Pop", line=dict(width=1), yaxis="y2"))
    fig.add_trace(go.Scatter(x=all_dose.index, y=uvx_pop, mode='lines', name=f"Age {age} UVX Pop", line=dict(width=1), yaxis="y2"))
    fig.add_trace(go.Scatter(x=all_dose.index, y=all_dose, mode='lines', name=f"Age {age} All Doses", line=dict(width=1), yaxis="y3"))
    fig.add_trace(go.Scatter(x=all_dose.index, y=all_dose_roll, mode='lines', name=f"Age {age} All Doses ({window}d avg)", line=dict(width=0.8, dash='solid'), yaxis="y3"))

# Update layout for the main figure
fig.update_layout(
    title=f"Deaths and Population Trends per Age Group {sim_extension}",
    colorway=px.colors.qualitative.Plotly[1:] + [px.colors.qualitative.Plotly[0]] ,  # Using a predefined qualitative color palette
    xaxis_title="Day",
    yaxis=dict(title="Raw Deaths / Normalized Deaths per 100k"),
    yaxis2=dict(
        title="Population", 
        overlaying='y', 
        side='right', 
        showgrid=False
    ),
    yaxis3=dict(
        title="Doses",   # Title for the new y-axis
        overlaying='y',  # Make it overlay the existing y-axis
        side='right',    # Place it on the right side
        position=0.95,   # Adjust the position for the new axis (further right)
        showgrid=False,  # Hide gridlines for this axis
    ),
    legend=dict(
        x=1.05,  # Move it farther to the right (default is 1.0 = just outside plot)
        y=1.0,
        xanchor='left',
        yanchor='top'
    ),    
    template="plotly_white",
    height=1000,
    width=1800
)


# Save original plot

output_path = fr"{OUTPUT_HTML1}{sim_extension}.html"
fig.write_html(output_path)
print(f"Plot saved to {output_path}")


# --- Dose-Aligned Event Stacking Plot ---
days_before = 125
days_after = 125
window_size = days_before + days_after + 1

age_curves = {}
x = np.arange(-days_before, days_after + 1)

for age in range(114):
    age_str = str(age)
    doses = dose_all_df[age_str]
    deaths_total = deaths_total_df[age_str]
    deaths_vx = deaths_vx_df[age_str]
    deaths_uvx = deaths_uvx_df[age_str]

    pop = initial_population[age_str]
    cum_deaths_total = cumulative_deaths_total[age_str]
    cum_first_dose = cumulative_first_dose[age_str]

    if pop == 0 or np.isnan(pop):
        print(f"Skipping age {age_str}: zero or NaN initial population")
        continue

    dose_threshold = 0.02 * doses.max()
    dose_days = doses[doses > dose_threshold].index

    for label, deaths, calc_pop in zip(
        ["Total", "VX", "UVX"], 
        [deaths_total, deaths_vx, deaths_uvx], 
        [lambda: pop - cum_deaths_total, 
         lambda: cum_first_dose - cum_deaths_vx, 
         lambda: pop - cum_first_dose - cum_deaths_uvx]
    ):
        stacked_deaths = np.zeros(window_size)
        stacked_pops = np.zeros(window_size)
        valid_stacks = 0

        for dose_day in dose_days:
            start_day = dose_day - days_before
            end_day = dose_day + days_after

            try:
                deaths_window = deaths.loc[start_day:end_day]
                pop_window = calc_pop().loc[start_day:end_day]

                if len(deaths_window) != window_size or len(pop_window) != window_size:
                    print(f"Skipping dose day {dose_day} for age {age_str}, label {label}, invalid window size")
                    continue

                stacked_deaths += deaths_window.values
                stacked_pops += pop_window.values
                valid_stacks += 1

            except KeyError:
                print(f"Skipping dose day {dose_day} for age {age_str}, label {label}, KeyError")
                continue

        if valid_stacks == 0:
            print(f"No valid stacks for age {age_str}, label {label}")
            continue

        mean_deaths = stacked_deaths / valid_stacks
        mean_pop = stacked_pops / valid_stacks

        # Day-by-day normalization: return 0 if pop is 0 or NaN
        normalized_curve = np.zeros(window_size)
        for i in range(window_size):
            p = mean_pop[i]
            d = mean_deaths[i]
            normalized_curve[i] = (d / p * 100_000) if p > 0 and not np.isnan(p) else 0.0

        age_curves[(age, label)] = normalized_curve

# --- Plot ---
stack_fig = go.Figure()

for (age, label), curve in age_curves.items():
    early_pre = (x >= -125) & (x < -60)
    late_pre = (x >= -60) & (x < 0)
    post = (x >= 0) & (x <= 125)

    mean_early_pre = np.mean(curve[early_pre])
    mean_late_pre = np.mean(curve[late_pre])
    mean_post = np.mean(curve[post])

    summary_text = (
        f"Age {age} {label}<br>"
        f"-125:-60: {mean_early_pre:.6f}<br>"
        f"-60:-1: {mean_late_pre:.6f}<br>"
        f"0:125: {mean_post:.6f}"
    )

    stack_fig.add_trace(go.Scatter(
        x=x,
        y=curve,
        mode="lines",
        name=f"Age {age} {label}",
        line=dict(width=1),
        hovertemplate=summary_text + "<br>Day %{x}, Value %{y:.6f}<extra></extra>"
    ))

stack_fig.update_layout(
    title=f"Normalized Stacked Mean Deaths per Age (Aligned to Doses) {sim_extension}",
    xaxis_title="Days Relative to Dose",
    yaxis_title="Normalized Stacked Mean Death",
    legend_title="Age Group",
    template="plotly_white",
    height=900,
    width=1600
)

# --- Add Raw (Non-normalized) Traces to Existing stack_fig ---
for (age, label), norm_curve in age_curves.items():
    age_str = str(age)
    doses = dose_all_df[age_str]
    deaths = {
        "Total": deaths_total_df[age_str],
        "VX": deaths_vx_df[age_str],
        "UVX": deaths_uvx_df[age_str]
    }[label]

    dose_threshold = 0.02 * doses.max()
    dose_days = doses[doses > dose_threshold].index

    stacked_raw = np.zeros(window_size)
    valid_raw_stacks = 0

    for dose_day in dose_days:
        start_day = dose_day - days_before
        end_day = dose_day + days_after

        try:
            deaths_window = deaths.loc[start_day:end_day]
            if len(deaths_window) != window_size:
                continue
            stacked_raw += deaths_window.values
            valid_raw_stacks += 1
        except KeyError:
            continue

    if valid_raw_stacks == 0:
        continue

    raw_mean_curve = stacked_raw / valid_raw_stacks

    # Add raw trace to existing figure using dashed line and secondary y-axis
    stack_fig.add_trace(go.Scatter(
        x=x,
        y=raw_mean_curve,
        mode="lines",
        name=f"Age {age} {label} (Raw)",
        line=dict(width=0.8),
        yaxis="y2",
        hovertemplate=f"Age {age} {label} (Raw)<br>Day %{{x}}, Deaths %{{y:.6f}}<extra></extra>"
    ))


# Update layout with secondary y-axis for raw deaths
stack_fig.update_layout(
    yaxis2=dict(
        title="Raw Stacked Mean Deaths",
        overlaying='y',
        side='right',
        showgrid=False
    )
)

# Save
stack_output_path = fr"{OUTPUT_HTML}{sim_extension}.html"
stack_fig.write_html(stack_output_path)
print(f"Stacked plot saved to {stack_output_path}")