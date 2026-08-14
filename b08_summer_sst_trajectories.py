# =========================================================
# CABRERA SST CLIMATE TRAJECTORIES
# Summer (JJA) Regression & Climate Evolution
# =========================================================

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[3])
)

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================================================
# 1. PATHS
# =========================================================

from config.paths import (

    CMEMS_SST_DIR,

    CMIP6_HISTORICAL_FILE,

    CMIP6_SST_GAP_DIR,

    SST_CMEMS_FIGURES_DIR

)

# =========================================================
# CMEMS OBSERVATIONS
# =========================================================

FILE_CMEMS = (

    CMEMS_SST_DIR /

    "cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021_1777296864009.nc"

)

# =========================================================
# HISTORICAL MODEL
# =========================================================

FILE_HIST = CMIP6_HISTORICAL_FILE

# =========================================================
# FUTURE SCENARIOS
# =========================================================

BASE_FOLDER = CMIP6_SST_GAP_DIR

FILES_FUT = {

    "SSP1-2.6":

    BASE_FOLDER /

    "tos_Omon_CNRM-CM6-1-HR_ssp126_r1i1p1f2_gn_20150116-21001216.nc",

    "SSP2-4.5":

    BASE_FOLDER /

    "tos_Omon_CNRM-CM6-1-HR_ssp245_r1i1p1f2_gn_20150116-21001216.nc",

    "SSP5-8.5":

    BASE_FOLDER /

    "tos_Omon_CNRM-CM6-1-HR_ssp585_r1i1p1f2_gn_20150116-21001216.nc",

}

# =========================================================
# OUTPUT
# =========================================================

OUTPUT_DIR = SST_CMEMS_FIGURES_DIR

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (

    OUTPUT_DIR /

    "Cabrera_Summer_Regression_SST.png"

)

# =========================================================
# 2. PARAMETERS
# =========================================================

LAT_MIN, LAT_MAX = 38.8, 39.6
LON_MIN, LON_MAX = 2.4, 3.6

BASELINE_START_YEAR = 1995
BASELINE_END_YEAR = 2014

SUMMER_MONTHS = [6, 7, 8]  # June, July, August

OBS_START_YEAR = 1982
OBS_END_YEAR = 2025

# X-axis tick marks. Start at the beginning of the observed record so the
# early years are labelled, rather than leaving the left of the axis blank.
XTICK_START = 1980
XTICK_STEP = 10

COLORS = {
    "SSP1-2.6": "#2b83ba",
    "SSP2-4.5": "#fdae61",
    "SSP5-8.5": "#d7191c"
}

# ---------------------------------------------------------
# TEXT SIZES  (adjust here if anything is still too small)
# ---------------------------------------------------------

FS_TICKS = 14        # axis tick numbers
FS_YLABEL = 15       # y-axis label
FS_XLABEL = 16       # x-axis label
FS_PANEL_TITLE = 18  # per-panel scenario title
FS_SUPTITLE = 22     # overall figure title
FS_LEGEND = 13       # legend text
FS_SPLIT_LABEL = 14  # year label on the transition line

# Vertical position of the transition year label, as a fraction of
# the axes height (0 = bottom, 1 = top).
SPLIT_LABEL_HEIGHT = 0.94

# ---------------------------------------------------------
# BACKGROUND (RAW ANNUAL SUMMER) POINT STYLE
# raised from alpha 0.25/0.30 so the variability is visible
# ---------------------------------------------------------

BG_ALPHA_OBS = 0.45
BG_ALPHA_FUT = 0.45
BG_LINEWIDTH = 1.0
BG_MARKERSIZE = 3

# ---------------------------------------------------------
# Y-AXIS RANGE
# Set YLIM = None to fit the axis tightly to the data (recommended:
# the lines then fill the full panel height and are easier to read).
# Set YLIM = (20, 32) or similar to force a fixed range instead.
# ---------------------------------------------------------

YLIM = None

# Padding added above and below the data when YLIM is None, in degC.
YPAD = 0.4

# ---------------------------------------------------------
# LEGEND PLACEMENT
# "outside" puts it to the right of the panels, clear of the data.
# "inside"  puts it in the upper-left corner of each panel.
# ---------------------------------------------------------

LEGEND_POSITION = "outside"

# =========================================================
# 3. FUNCTIONS
# =========================================================

def get_summer_series(file_path):

    print(f"\n>>> Loading:")
    print(file_path.name)

    ds = xr.open_dataset(
        file_path,
        decode_times=True,
        chunks={"time": 12}
    )

    if "tos" in ds.data_vars:
        var_name = "tos"
    elif "analysed_sst" in ds.data_vars:
        var_name = "analysed_sst"
    else:
        var_name = "sst"

    sst = ds[var_name]

    lat_n = "lat" if "lat" in ds.variables else "latitude"
    lon_n = "lon" if "lon" in ds.variables else "longitude"

    mask = (
        (ds[lat_n] >= LAT_MIN) &
        (ds[lat_n] <= LAT_MAX) &
        (ds[lon_n] >= LON_MIN) &
        (ds[lon_n] <= LON_MAX)
    )

    sst_region = sst.where(mask)

    horiz_dims = [
        d for d in sst.dims
        if d != "time"
    ]

    series = sst_region.mean(
        dim=horiz_dims,
        skipna=True
    ).compute()

    if float(series.mean()) > 100:
        series = series - 273.15

    # Isolate Summer Months (JJA)
    summer_series = series.where(
        series["time.month"].isin(SUMMER_MONTHS),
        drop=True
    )

    # Calculate Mean Summer SST per Year
    annual_summer = summer_series.groupby("time.year").mean()

    return annual_summer


def calculate_trend(years, values):
    """Calculates linear regression and decadal trend for annual data."""

    idx = np.isfinite(values)

    if not np.any(idx):
        return None, None, 0.0

    x_yr = years[idx]
    y_vals = values[idx]

    # Linear fit (Degree 1)
    coeffs = np.polyfit(x_yr, y_vals, 1)
    poly = np.poly1d(coeffs)
    y_trend = poly(x_yr)

    # Slope is degC per year. Multiply by 10 for degC/decade.
    trend_decadal = coeffs[0] * 10

    return x_yr, y_trend, trend_decadal

# =========================================================
# 4. LOAD & PREPARE CMEMS (SUMMER)
# =========================================================

print("\n>>> Processing CMEMS observations")

obs_summer = get_summer_series(FILE_CMEMS)
obs_summer = obs_summer.sel(year=slice(OBS_START_YEAR, OBS_END_YEAR))

# =========================================================
# 5. BASELINES
# =========================================================

obs_baseline = obs_summer.sel(
    year=slice(BASELINE_START_YEAR, BASELINE_END_YEAR)
).mean().item()

hist_summer = get_summer_series(FILE_HIST)

hist_baseline = hist_summer.sel(
    year=slice(BASELINE_START_YEAR, BASELINE_END_YEAR)
).mean().item()

print(f"\n>>> Observed summer baseline (1995-2014): {obs_baseline:.2f} degC")
print(f">>> Model summer baseline    (1995-2014): {hist_baseline:.2f} degC")

# =========================================================
# 6. CMEMS SMOOTHING & TREND
# =========================================================

LAST_CMEMS_YEAR = int(obs_summer.year.max())

print(f"\n>>> Last CMEMS year: {LAST_CMEMS_YEAR}")

# 5-year rolling mean to smooth interannual summer variability
obs_smooth = obs_summer.rolling(
    year=5,
    center=True
).mean()

# Calculate strict linear regression for CMEMS Summer
obs_t, obs_trend, obs_dec_trend = calculate_trend(
    obs_summer.year.values,
    obs_summer.values
)

# =========================================================
# 7. FIGURE SETUP
# =========================================================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(17, 13),
    sharex=True
)

# leave room on the right when the legend sits outside the panels
_right = 0.80 if LEGEND_POSITION == "outside" else 0.97

plt.subplots_adjust(
    hspace=0.20,
    left=0.08,
    right=_right,
    top=0.93,
    bottom=0.08
)

# collects the data range across all panels so the y-axis can be
# fitted tightly to the values actually plotted
data_min = np.inf
data_max = -np.inf

# =========================================================
# 8. LOOP FUTURE SCENARIOS
# =========================================================

for ax, (scenario, file_fut) in zip(
    axes,
    FILES_FUT.items()
):

    print(f"\n>>> Processing {scenario}")

    # =====================================================
    # FUTURE MODEL
    # =====================================================

    fut_summer = get_summer_series(file_fut)

    # =====================================================
    # DELTA CORRECTION (SUMMER)
    # =====================================================

    fut_anomaly = fut_summer - hist_baseline

    fut_corrected = obs_baseline + fut_anomaly

    # =====================================================
    # FILTER AFTER CMEMS
    # =====================================================

    fut_corrected = fut_corrected.where(
        fut_corrected.year >= LAST_CMEMS_YEAR,
        drop=True
    )

    # =====================================================
    # SMOOTH & TREND
    # =====================================================

    fut_smooth = fut_corrected.rolling(
        year=5,
        center=True
    ).mean()

    fut_t, fut_trend, fut_dec_trend = calculate_trend(
        fut_corrected.year.values,
        fut_corrected.values
    )

    # track the range of everything actually drawn on this panel
    for _arr in (obs_summer.values, fut_corrected.values):
        if np.any(np.isfinite(_arr)):
            data_min = min(data_min, float(np.nanmin(_arr)))
            data_max = max(data_max, float(np.nanmax(_arr)))

    # =====================================================
    # PLOT RAW SUMMER POINTS (BACKGROUND)
    # =====================================================

    ax.plot(
        obs_summer.year,
        obs_summer.values,
        color="black",
        alpha=BG_ALPHA_OBS,
        linewidth=BG_LINEWIDTH,
        marker="o",
        markersize=BG_MARKERSIZE,
        zorder=1
    )

    ax.plot(
        fut_corrected.year,
        fut_corrected.values,
        color=COLORS[scenario],
        alpha=BG_ALPHA_FUT,
        linewidth=BG_LINEWIDTH,
        marker="o",
        markersize=BG_MARKERSIZE,
        zorder=1
    )

    # =====================================================
    # PLOT SMOOTHED (SOLID)
    # =====================================================

    ax.plot(
        obs_smooth.year,
        obs_smooth.values,
        color="black",
        linewidth=2.5,
        label="Observed CMEMS (5y mean)",
        zorder=3
    )

    ax.plot(
        fut_smooth.year,
        fut_smooth.values,
        color=COLORS[scenario],
        linewidth=2.5,
        label=f"{scenario} (5y mean)",
        zorder=3
    )

    # =====================================================
    # PLOT REGRESSION TRENDS (DASHED)
    # =====================================================

    ax.plot(
        obs_t,
        obs_trend,
        color="black",
        linestyle="--",
        linewidth=2.5,
        label=f"Obs Trend (+{obs_dec_trend:.2f}\u00b0C / dec)",
        zorder=4
    )

    ax.plot(
        fut_t,
        fut_trend,
        color=COLORS[scenario],
        linestyle="--",
        linewidth=2.5,
        label=f"Fut Trend (+{fut_dec_trend:.2f}\u00b0C / dec)",
        zorder=4
    )

    # =====================================================
    # TRANSITION LINE
    # =====================================================

    ax.axvline(
        LAST_CMEMS_YEAR,
        color="gray",
        linestyle=":",
        linewidth=2,
        zorder=2
    )

    # =====================================================
    # YEAR LABEL ON THE TRANSITION LINE
    # =====================================================

    ax.annotate(
        str(LAST_CMEMS_YEAR),
        xy=(LAST_CMEMS_YEAR, SPLIT_LABEL_HEIGHT),
        xycoords=("data", "axes fraction"),
        xytext=(0, 0),
        textcoords="offset points",
        ha="center",
        va="center",
        fontsize=FS_SPLIT_LABEL,
        color="dimgray",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="none",
            alpha=0.85
        ),
        zorder=5
    )

    # =====================================================
    # STYLE
    # =====================================================

    ax.set_title(
        scenario,
        fontsize=FS_PANEL_TITLE,
        fontweight="bold",
        loc="left"
    )

    ax.set_ylabel(
        "Mean Summer SST (\u00b0C)",
        fontsize=FS_YLABEL
    )

    ax.grid(
        alpha=0.2,
        linestyle="--"
    )

    ax.set_xticks(
        np.arange(XTICK_START, 2101, XTICK_STEP)
    )

    ax.tick_params(
        axis="x",
        labelsize=FS_TICKS
    )

    ax.tick_params(
        axis="y",
        labelsize=FS_TICKS
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if LEGEND_POSITION == "outside":
        ax.legend(
            frameon=False,
            fontsize=FS_LEGEND,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            ncol=1,
            labelspacing=0.8,
            handlelength=2.4
        )
    else:
        ax.legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.9,
            fontsize=FS_LEGEND,
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            ncol=2,
            borderpad=0.8,
            labelspacing=0.6,
            columnspacing=1.5
        )

# =========================================================
# 8b. Y-AXIS LIMITS
# =========================================================
# Fitted to the plotted data so the lines fill the panel height.

if YLIM is None:
    _lo = np.floor((data_min - YPAD) * 2) / 2
    _hi = np.ceil((data_max + YPAD) * 2) / 2
else:
    _lo, _hi = YLIM

print(f"\n>>> Y-axis range: {_lo:.1f} to {_hi:.1f} degC")

for ax in axes:
    ax.set_ylim(_lo, _hi)

# =========================================================
# 9. X LABEL
# =========================================================

axes[-1].set_xlabel(
    "Year",
    fontsize=FS_XLABEL,
    fontweight="bold"
)

# =========================================================
# 10. TITLE
# =========================================================

fig.suptitle(
    "Cabrera Summer (JJA) Sea Surface Temperature\n",
    fontsize=FS_SUPTITLE,
    fontweight="bold"
)

# =========================================================
# 11. SAVE
# =========================================================

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

print(f"\n>>> Figure saved:")
print(output_file)

plt.close()

print("\n>>> Summer SST regression trajectories generated successfully.")