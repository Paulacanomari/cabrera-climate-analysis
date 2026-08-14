# =========================================================
# CABRERA SST CLIMATE TRAJECTORIES
# AUGUST Regression & Climate Evolution
# With mortality threshold (28 C)
# =========================================================

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

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

FILE_CMEMS = (
    CMEMS_SST_DIR /
    "cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021_1777296864009.nc"
)
FILE_HIST = CMIP6_HISTORICAL_FILE
BASE_FOLDER = CMIP6_SST_GAP_DIR

FILES_FUT = {
    "SSP1-2.6": BASE_FOLDER / "tos_Omon_CNRM-CM6-1-HR_ssp126_r1i1p1f2_gn_20150116-21001216.nc",
    "SSP2-4.5": BASE_FOLDER / "tos_Omon_CNRM-CM6-1-HR_ssp245_r1i1p1f2_gn_20150116-21001216.nc",
    "SSP5-8.5": BASE_FOLDER / "tos_Omon_CNRM-CM6-1-HR_ssp585_r1i1p1f2_gn_20150116-21001216.nc",
}

OUTPUT_DIR = SST_CMEMS_FIGURES_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_file = (
    OUTPUT_DIR /
    "Cabrera_August_Regression_SST_Mortality_Threshold.png"
)

# =========================================================
# 2. PARAMETERS
# =========================================================

LAT_MIN, LAT_MAX = 38.8, 39.6
LON_MIN, LON_MAX = 2.4, 3.6

BASELINE_START_YEAR = 1995
BASELINE_END_YEAR = 2014

# Observed record clipped to complete years (matches other SST figures)
OBS_START_YEAR = 1982
OBS_END_YEAR = 2025

AUGUST_MONTH = [8]            # AUGUST only (warmest month; 28 C threshold)

MORTALITY_THRESHOLD = 28.0

COLORS = {
    "SSP1-2.6": "#2b83ba",
    "SSP2-4.5": "#fdae61",
    "SSP5-8.5": "#d7191c"
}

# ---------------------------------------------------------
# TEXT SIZES  (adjust here if anything is too small)
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
# X-AXIS TICKS
# Start at the beginning of the observed record so the early years
# are labelled, rather than leaving the left of the axis blank.
# ---------------------------------------------------------

XTICK_START = 1980
XTICK_END = 2100
XTICK_STEP = 10

# ---------------------------------------------------------
# BACKGROUND (RAW ANNUAL AUGUST) POINT STYLE
# raised from alpha 0.25/0.30 so the variability is visible
# ---------------------------------------------------------

BG_ALPHA_OBS = 0.45
BG_ALPHA_FUT = 0.45
BG_LINEWIDTH = 1.0
BG_MARKERSIZE = 3

# ---------------------------------------------------------
# Y-AXIS RANGE
# Set YLIM = None to fit the axis tightly to the data, so the lines
# fill the panel height. Set YLIM = (22, 34) to force a fixed range.
# ---------------------------------------------------------

YLIM = None
YPAD = 0.4           # degC of padding above and below the data

# ---------------------------------------------------------
# LEGEND PLACEMENT
# "outside" puts it to the right of the panels, clear of the data.
# "inside"  puts it in the upper-left corner of each panel.
# ---------------------------------------------------------

LEGEND_POSITION = "outside"

# =========================================================
# 3. FUNCTIONS
# =========================================================

def get_august_series(file_path):

    print(f"\n>>> Loading:")
    print(file_path.name)

    ds = xr.open_dataset(file_path, decode_times=True, chunks={"time": 12})

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
        (ds[lat_n] >= LAT_MIN) & (ds[lat_n] <= LAT_MAX) &
        (ds[lon_n] >= LON_MIN) & (ds[lon_n] <= LON_MAX)
    )

    sst_region = sst.where(mask)
    horiz_dims = [d for d in sst.dims if d != "time"]
    series = sst_region.mean(dim=horiz_dims, skipna=True).compute()

    if float(series.mean()) > 100:
        series = series - 273.15

    august_series = series.where(
        series["time.month"].isin(AUGUST_MONTH), drop=True
    )
    annual_august = august_series.groupby("time.year").mean()

    return annual_august


def calculate_trend(years, values):
    idx = np.isfinite(values)
    if not np.any(idx):
        return None, None, 0.0
    x_yr = years[idx]
    y_vals = values[idx]
    coeffs = np.polyfit(x_yr, y_vals, 1)
    poly = np.poly1d(coeffs)
    y_trend = poly(x_yr)
    trend_decadal = coeffs[0] * 10
    return x_yr, y_trend, trend_decadal

# =========================================================
# 4. LOAD & PREPARE CMEMS
# =========================================================

print("\n>>> Processing CMEMS observations")

obs_august = get_august_series(FILE_CMEMS)
obs_august = obs_august.sel(year=slice(OBS_START_YEAR, OBS_END_YEAR))

# =========================================================
# 5. BASELINES
# =========================================================

obs_baseline = obs_august.sel(
    year=slice(BASELINE_START_YEAR, BASELINE_END_YEAR)
).mean().item()

hist_august = get_august_series(FILE_HIST)
hist_baseline = hist_august.sel(
    year=slice(BASELINE_START_YEAR, BASELINE_END_YEAR)
).mean().item()

print(f"\n>>> Observed August baseline (1995-2014): {obs_baseline:.2f} degC")
print(f">>> Model August baseline    (1995-2014): {hist_baseline:.2f} degC")

# =========================================================
# 6. CMEMS SMOOTHING & TREND
# =========================================================

LAST_CMEMS_YEAR = int(obs_august.year.max())
print(f">>> Last CMEMS year: {LAST_CMEMS_YEAR}")

obs_smooth = obs_august.rolling(year=5, center=True).mean()

obs_t, obs_trend, obs_dec_trend = calculate_trend(
    obs_august.year.values, obs_august.values
)

# =========================================================
# 7. FIGURE
# =========================================================

fig, axes = plt.subplots(3, 1, figsize=(17, 13), sharex=True)

# leave room on the right when the legend sits outside the panels
_right = 0.78 if LEGEND_POSITION == "outside" else 0.97

plt.subplots_adjust(
    hspace=0.20, left=0.08, right=_right, top=0.93, bottom=0.08
)

# collects the data range across all panels so the y-axis can be
# fitted tightly to the values actually plotted
data_min = np.inf
data_max = -np.inf

for ax, (scenario, file_fut) in zip(axes, FILES_FUT.items()):

    print(f"\n>>> Processing {scenario}")

    fut_august = get_august_series(file_fut)
    fut_anomaly = fut_august - hist_baseline
    fut_corrected = obs_baseline + fut_anomaly

    # clean handoff: strictly after last CMEMS year
    fut_corrected = fut_corrected.where(
        fut_corrected.year > LAST_CMEMS_YEAR, drop=True
    )

    fut_smooth = fut_corrected.rolling(year=5, center=True).mean()
    fut_t, fut_trend, fut_dec_trend = calculate_trend(
        fut_corrected.year.values, fut_corrected.values
    )

    # track the range of everything actually drawn on this panel
    for _arr in (obs_august.values, fut_corrected.values):
        if np.any(np.isfinite(_arr)):
            data_min = min(data_min, float(np.nanmin(_arr)))
            data_max = max(data_max, float(np.nanmax(_arr)))

    # ----- raw annual values (background) --------------------
    ax.plot(obs_august.year, obs_august.values, color="black",
            alpha=BG_ALPHA_OBS, linewidth=BG_LINEWIDTH,
            marker="o", markersize=BG_MARKERSIZE, zorder=1)
    ax.plot(fut_corrected.year, fut_corrected.values,
            color=COLORS[scenario],
            alpha=BG_ALPHA_FUT, linewidth=BG_LINEWIDTH,
            marker="o", markersize=BG_MARKERSIZE, zorder=1)

    # ----- smoothed ------------------------------------------
    ax.plot(obs_smooth.year, obs_smooth.values, color="black",
            linewidth=2.5, label="Observed CMEMS (5-year mean)", zorder=3)
    ax.plot(fut_smooth.year, fut_smooth.values, color=COLORS[scenario],
            linewidth=2.5, label=f"{scenario} (5-year mean)", zorder=3)

    # ----- trends --------------------------------------------
    ax.plot(obs_t, obs_trend, color="black", linestyle="--", linewidth=2.5,
            label=f"Obs trend (+{obs_dec_trend:.2f} \u00b0C/dec)", zorder=4)
    ax.plot(fut_t, fut_trend, color=COLORS[scenario], linestyle="--",
            linewidth=2.5,
            label=f"Future trend (+{fut_dec_trend:.2f} \u00b0C/dec)", zorder=4)

    # ----- transition line and threshold ---------------------
    ax.axvline(LAST_CMEMS_YEAR, color="gray", linestyle=":",
               linewidth=2, zorder=2)

    ax.axhline(MORTALITY_THRESHOLD, color="red", linestyle="-.",
               linewidth=2.5, zorder=5,
               label="Mortality threshold (28 \u00b0C)")

    # ----- year label on the transition line -----------------
    ax.annotate(
        str(LAST_CMEMS_YEAR),
        xy=(LAST_CMEMS_YEAR, SPLIT_LABEL_HEIGHT),
        xycoords=("data", "axes fraction"),
        xytext=(0, 0),
        textcoords="offset points",
        ha="center", va="center",
        fontsize=FS_SPLIT_LABEL, fontweight="bold",
        color="dimgray",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="none", alpha=0.85),
        zorder=6
    )

    # ----- style ---------------------------------------------
    ax.set_title(scenario, fontsize=FS_PANEL_TITLE,
                 fontweight="bold", loc="left")
    ax.set_ylabel("Mean August SST (\u00b0C)", fontsize=FS_YLABEL)
    ax.grid(alpha=0.2, linestyle="--")

    ax.set_xticks(np.arange(XTICK_START, XTICK_END + 1, XTICK_STEP))
    ax.tick_params(axis="x", labelsize=FS_TICKS)
    ax.tick_params(axis="y", labelsize=FS_TICKS)
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
            frameon=True, facecolor="white", edgecolor="none",
            framealpha=0.9,
            fontsize=FS_LEGEND, loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            ncol=2, borderpad=0.8, labelspacing=0.6,
            columnspacing=1.5
        )

# =========================================================
# 7b. Y-AXIS LIMITS
# =========================================================
# Fitted to the plotted data so the lines fill the panel height.
# The threshold line is included so it is always visible.

if YLIM is None:
    _lo = np.floor((min(data_min, MORTALITY_THRESHOLD) - YPAD) * 2) / 2
    _hi = np.ceil((max(data_max, MORTALITY_THRESHOLD) + YPAD) * 2) / 2
else:
    _lo, _hi = YLIM

print(f"\n>>> Y-axis range: {_lo:.1f} to {_hi:.1f} degC")

for ax in axes:
    ax.set_ylim(_lo, _hi)

# =========================================================
# 8. LABELS AND TITLE
# =========================================================

axes[-1].set_xlabel("Year", fontsize=FS_XLABEL, fontweight="bold")

fig.suptitle(
    "Cabrera August Sea Surface Temperature and Mortality Threshold",
    fontsize=FS_SUPTITLE, fontweight="bold"
)

plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"\n>>> Figure saved:\n{output_file}")
plt.close()
print("\n>>> August SST regression trajectories generated successfully.")