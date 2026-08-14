# =========================================================
# CABRERA SEA LEVEL RISE PROJECTIONS
# IPCC AR6 - Medium Confidence
# =========================================================

from pathlib import Path

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# 1. PATHS
# Set DATA_DIR to the folder holding the downloaded datasets.
# See the README for the products and where to obtain them.
# =========================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("figures")

# =========================================================
# AR6 MEDIUM CONFIDENCE
# Directory structure as published in the AR6 sea level archive
# =========================================================

DATA_PATH = (

    DATA_DIR /

    "ar6" /

    "regional" /

    "confidence_output_files" /

    "medium_confidence"

)

# =========================================================
# OUTPUT DIRECTORY
# =========================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("\nOUTPUT DIRECTORY:")
print(OUTPUT_DIR.resolve())

# =========================================================
# SCENARIOS
# =========================================================

SCENARIOS = [

    "ssp126",

    "ssp245",

    "ssp585"

]

# =========================================================
# CABRERA COORDINATES
# =========================================================

CABRERA_LAT = 39.15
CABRERA_LON = 2.95

# =========================================================
# BASELINE / TIME RANGE
# =========================================================
# The AR6 regional projection files are provided at decadal
# timesteps from 2020 to 2150. The projections are expressed as
# change RELATIVE TO the 1995-2014 mean, which therefore defines
# the zero line rather than appearing as a plotted period.

BASELINE_LABEL = "1995-2014"

# Plot range. The dataset extends to 2150, but this assessment
# considers horizons up to 2100 only.
XMIN = 2020
XMAX = 2100
XTICK_STEP = 10

# =========================================================
# TEXT SIZES
# =========================================================

FS_TICKS = 13
FS_LABEL = 15
FS_TITLE = 17
FS_LEGEND = 13

# =========================================================
# COLORS
# =========================================================

COLORS = {

    "ssp126": "#2b83ba",

    "ssp245": "#fdae61",

    "ssp585": "#d7191c"

}

LABELS = {

    "ssp126": "SSP1-2.6 (Low emissions)",

    "ssp245": "SSP2-4.5 (Intermediate)",

    "ssp585": "SSP5-8.5 (High emissions)"

}

# =========================================================
# FIGURE
# =========================================================

fig, ax = plt.subplots(
    figsize=(11, 7)
)

# =========================================================
# FINAL TABLE
# =========================================================

final_table = []

# =========================================================
# MAIN LOOP
# =========================================================

for scenario in SCENARIOS:

    file_name = (
        f"total_{scenario}_medium_confidence_values.nc"
    )

    full_path = (

        DATA_PATH /

        scenario /

        file_name

    )

    print(f"\n>>> Opening:\n{full_path}")

    if not full_path.exists():

        print(
            f"\n>>> File not found:\n{full_path}"
        )

        continue

    # =====================================================
    # OPEN DATASET
    # =====================================================

    ds = xr.open_dataset(
        full_path
    )

    print("YEARS:", ds.years.values)

    # =====================================================
    # FIND CLOSEST GRID POINT
    # =====================================================

    dist = np.sqrt(

        (ds.lat - CABRERA_LAT) ** 2 +

        (ds.lon - CABRERA_LON) ** 2

    )

    loc_idx = dist.argmin().item()

    # =====================================================
    # SEA LEVEL DATA
    # =====================================================

    data = ds.sea_level_change.isel(
        locations=loc_idx
    )

    # =====================================================
    # QUANTILES
    # =====================================================

    p17 = data.sel(
        quantiles=0.17
    )

    p50 = data.sel(
        quantiles=0.5
    )

    p83 = data.sel(
        quantiles=0.83
    )

    # =====================================================
    # mm -> m
    # =====================================================

    df_p17 = pd.Series(

        p17.values / 1000.0,

        index=ds.years.values

    )

    df_p50 = pd.Series(

        p50.values / 1000.0,

        index=ds.years.values

    )

    df_p83 = pd.Series(

        p83.values / 1000.0,

        index=ds.years.values

    )

    # =====================================================
    # RESTRICT TO PLOT RANGE (2020-2100)
    # =====================================================

    _keep = (df_p50.index >= XMIN) & (df_p50.index <= XMAX)

    df_p17_plot = df_p17[_keep]
    df_p50_plot = df_p50[_keep]
    df_p83_plot = df_p83[_keep]

    # =====================================================
    # DEBUG
    # =====================================================

    print(f"\n{scenario.upper()}")

    print(
        f"2060 p50 = {df_p50.loc[2060]:.3f} m"
    )

    print(
        f"2100 p50 = {df_p50.loc[2100]:.3f} m"
    )

    print(
        f"2100 p83 = {df_p83.loc[2100]:.3f} m"
    )

    # =====================================================
    # PLOT CENTRAL LINE
    # =====================================================

    ax.plot(

        df_p50_plot.index,

        df_p50_plot.values,

        color=COLORS[scenario],

        linewidth=2.5,

        marker="o",

        markersize=5,

        label=LABELS[scenario],

        zorder=3

    )

    # =====================================================
    # UNCERTAINTY BAND
    # =====================================================

    ax.fill_between(

        df_p50_plot.index,

        df_p17_plot.values,

        df_p83_plot.values,

        color=COLORS[scenario],

        alpha=0.18,

        zorder=1

    )

    # =====================================================
    # EXACT VALUES
    # =====================================================

    val_2060 = float(
        df_p50.loc[2060]
    )

    val_2100 = float(
        df_p50.loc[2100]
    )

    final_table.append({

        "Scenario": scenario,

        "2060 (m)": round(
            val_2060,
            3
        ),

        "2100 (m)": round(
            val_2100,
            3
        )

    })

# =========================================================
# FIGURE STYLE
# =========================================================

# Zero line. This is the REFERENCE LEVEL defined by the
# 1995-2014 mean, not a period plotted on the axis.
ax.axhline(

    0,

    color="black",

    linestyle="--",

    linewidth=1.2,

    zorder=2,

    label=f"{BASELINE_LABEL} reference level"

)

ax.set_title(

    "Relative Sea Level Rise Projections\n"
    "Cabrera Archipelago (IPCC AR6, medium confidence)",

    fontsize=FS_TITLE,

    fontweight="bold"

)

ax.set_xlabel(
    "Year",
    fontsize=FS_LABEL
)

ax.set_ylabel(
    f"Sea level change (m) relative to {BASELINE_LABEL}",
    fontsize=FS_LABEL
)

ax.set_xlim(XMIN, XMAX)

ax.set_xticks(
    np.arange(XMIN, XMAX + 1, XTICK_STEP)
)

ax.tick_params(axis="x", labelsize=FS_TICKS)
ax.tick_params(axis="y", labelsize=FS_TICKS)

ax.grid(

    True,

    linestyle=":",

    alpha=0.6

)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    loc="upper left",
    fontsize=FS_LEGEND,
    frameon=True,
    facecolor="white",
    edgecolor="none",
    framealpha=0.9,
    borderpad=0.8,
    labelspacing=0.6
)

plt.tight_layout()

# =========================================================
# SAVE FIGURE
# =========================================================

figure_path = (
    OUTPUT_DIR /
    "Cabrera_SLR_Final.png"
)

plt.savefig(
    figure_path,
    dpi=300,
    bbox_inches="tight"
)

print(
    f"\n>>> Figure saved:\n{figure_path}"
)

# =========================================================
# FINAL TABLE
# =========================================================

df_final = (
    pd.DataFrame(final_table)
    .sort_values("Scenario")
)

print("\n>>> FINAL TABLE:\n")
print(df_final)

# =========================================================
# SAVE CSV
# =========================================================

table_path = (
    OUTPUT_DIR /
    "Cabrera_SLR_Table.csv"
)

print("\n>>> Saving CSV to:")
print(table_path.resolve())

df_final.to_csv(
    table_path,
    index=False
)

print(
    f"\n>>> Table saved:\n{table_path}"
)

# =========================================================
# PRINT SCENARIOS FOR FLOOD MODEL
# =========================================================

print("\n================================")
print("SLR SCENARIOS FOR FLOOD MODEL")
print("================================\n")

for _, row in df_final.iterrows():

    scenario = row["Scenario"]

    print(
        f'"{scenario}_2060": {row["2060 (m)"]},'
    )

    print(
        f'"{scenario}_2100": {row["2100 (m)"]},'
    )

plt.close()
