import pandas as pd
import numpy as np

# === CONFIG ===
INPUT_CSV = r"C:\github\CzechFOI-DRATE\TERRA\Vesely_106_202403141131.csv"
OUTPUT_CSV = r"C:\github\CzechFOI-DRATE\TERRA\matched_Vesely_106_202403141131.csv"
DOSE_COLS = [f"Datum_{i}" for i in range(1, 8)]
CHUNK_SIZE = 500_000  # Manage memory usage for large file
START_DATE = pd.to_datetime("2020-01-01")

# === LOAD DATA ===
df = pd.read_csv(INPUT_CSV, parse_dates=["DatumUmrti"] + DOSE_COLS, low_memory=False)

# === FILTER DECEASED INDIVIDUALS ONLY ===
df = df[df["DatumUmrti"].notna()].reset_index(drop=True)

# === CLASSIFY VACCINATED (VX) VS UNVACCINATED (UVX) ===
is_vx = df["Datum_1"].notna()
is_uvx = df[DOSE_COLS].isna().all(axis=1)

vx_df = df[is_vx].copy()
uvx_df = df[is_uvx].copy()

# Drop individuals who died before first dose
vx_df = vx_df[vx_df["DatumUmrti"] > vx_df["Datum_1"]].reset_index(drop=True)
uvx_df = uvx_df.reset_index(drop=True)

print(f"Eligible vaccinated (vx): {len(vx_df)}")
print(f"Eligible unvaccinated (uvx): {len(uvx_df)}")

# === SHUFFLE FOR RANDOM MATCHING ===
vx_df = vx_df.sample(frac=1, random_state=42).reset_index(drop=True)
uvx_df = uvx_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Convert datetime for comparison
vx_dates = vx_df["Datum_1"].values
uvx_dates = uvx_df["DatumUmrti"].values

# Prepare indices for fast matching
vx_used = np.zeros(len(vx_df), dtype=bool)
uvx_used = np.zeros(len(uvx_df), dtype=bool)

vx_idx_list = []
uvx_idx_list = []

# === MATCHING LOOP ===
vx_ptr = 0
uvx_ptr = 0

while vx_ptr < len(vx_df) and uvx_ptr < len(uvx_df):
    while uvx_ptr < len(uvx_df) and (uvx_used[uvx_ptr] or uvx_dates[uvx_ptr] <= vx_dates[vx_ptr]):
        uvx_ptr += 1
    if uvx_ptr >= len(uvx_df):
        break
    vx_idx_list.append(vx_ptr)
    uvx_idx_list.append(uvx_ptr)
    vx_used[vx_ptr] = True
    uvx_used[uvx_ptr] = True
    vx_ptr += 1
    uvx_ptr += 1

print(f"Total matches found: {len(vx_idx_list)}")

# === COMBINE AND SAVE MATCHED PAIRS ===
matched_vx = vx_df.iloc[vx_idx_list].copy().reset_index(drop=True)
matched_uvx = uvx_df.iloc[uvx_idx_list].copy().reset_index(drop=True)

# Append matched rows vertically, adding a column to indicate group
matched_vx["Group"] = "VX"
matched_uvx["Group"] = "UVX"

matched = pd.concat([matched_vx, matched_uvx], ignore_index=True)
matched.to_csv(OUTPUT_CSV, index=False)
print(f"Saved matched pairs to: {OUTPUT_CSV}")
