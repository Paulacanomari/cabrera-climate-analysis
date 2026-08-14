# =========================================================
# CABRERA SST CLIMATE TRAJECTORIES
# Monthly Regression & Continuous Climate Evolution
# =========================================================
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    "SSP1-2.6": BASE_FOLDER / "tos_Omon_CNRM-CM6-1-HR_ssp126_r1i1p1f2_gn_20150116-21001216.nc",
    "SSP2-4.5": BASE_FOLDER / "tos_Omon_CNRM-CM6-1-HR_ssp245_r1i1p1f2_gn_20150116-21001216.nc",
    "SSP5-8.5": BASE_FOLDER / "tos_Omon_CNRM-CM6-1-HR_ssp585_r1i1p1f2_gn_20150116-21001216.nc",
}

# =========================================================
# OUTPUT
# =========================================================
OUTPUT_DIR = SST_CMEMS_FIGURES_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_file = (
    OUTPUT_DIR /
    "Cabrera_Monthly_Regression_SST_continuous.png"
)

# =========================================================
# 2. PARAMETERS
# =========================================================
LAT_MIN, LAT_MAX = 38.8, 39.6
LON_MIN, LON_MAX = 2.4, 3.6
BASELINE_START = "1995-01-01"
BASELINE_END = "2014-12-31"
# Observed record clipped to complete years (matches standalone observed figure)
OBS_START = "1982-01-01"
OBS_END = "2025-12-31"
COLORS = {
    "SSP1-2.6": "#2b83ba",
    "SSP2-4.5": "#fdae61",
    "SSP5-8.5": "#d7191c"
}

# ---------------------------------------------------------
# TEXT SIZES 
# ---------------------------------------------------------
FS_TICKS = 14  # axis tick numbers
FS_YLABEL = 15  # y-axis label
FS_XLABEL = 16  # x-axis label
FS_PANEL_TITLE = 18  # per-panel scenario title
FS_SUPTITLE = 22  # overall figure title
FS_LEGEND = 13  # legend text
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
# BACKGROUND (RAW MONTHLY) LINE STYLE
# raised from alpha 0.15/0.20 so the variability is visible
# ---------------------------------------------------------
BG_ALPHA_OBS = 0.45
BG_ALPHA_FUT = 0.45
BG_LINEWIDTH = 1.0

# ---------------------------------------------------------
# Y-AXIS RANGE
# Set YLIM = None to fit the axis tightly to the data (recommended:
# the lines then fill the full panel height and are easier to read).
# Set YLIM = (13, 33) or similar to force a fixed range instead.
#
# NOTE: this figure plots the raw monthly series, which follows the
# full seasonal cycle, so the fitted range will still be wide
# (roughly 13-33 degC). That is the seasonal envelope, not an error.
# ---------------------------------------------------------
YLIM = None
# Padding added above and below the data when YLIM is None, in degC.
YPAD = 0.5

# ---------------------------------------------------------
# LEGEND PLACEMENT
# "outside" puts it to the right of the panels, clear of the data.
# "inside" puts it in the upper-left corner of each panel.
# ---------------------------------------------------------
LEGEND_POSITION = "outside"

# =========================================================
# 3. FUNCTIONS
# =========================================================

def get_sst_series(file_path):
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
    horiz_dims = [d for d in sst.dims if d != "time"]
    series = sst_region.mean(dim=horiz_dims, skipna=True).compute()
    if float(series.mean()) > 100:
        series = series - 273.15
    try:
        series["time"] = pd.to_datetime(
            series.indexes["time"].to_datetimeindex()
        )
    except Exception:
        pass
    return series

def calculate_trend(times, values):
    """Calculates linear regression and decadal trend."""
    idx = np.isfinite(values)
    if not np.any(idx):
        return None, None, 0.0
    x_dates = times[idx]
    y_vals = values[idx]
    x_num = mdates.date2num(x_dates)
    coeffs = np.polyfit(x_num, y_vals, 1)
    poly = np.poly1d(coeffs)
    y_trend = poly(x_num)
    # Slope is degC per day. Multiply by days in a decade.
    trend_decadal = coeffs[0] * 365.25 * 10
    return x_dates, y_trend, trend_decadal

# =========================================================
# 4. LOAD & PREPARE CMEMS (MONTHLY)
# =========================================================
print("\n>>> Processing CMEMS observations")
sst_obs = get_sst_series(FILE_CMEMS)
# Resample to monthly means and clip to complete observed years (1982-2025)
obs_monthly = (
    sst_obs.resample(time="1MS").mean()
    .sel(time=slice(OBS_START, OBS_END))
)

# =========================================================
# 5. BASELINES (MONTHLY CLIMATOLOGIES)
# =========================================================
obs_baseline = obs_monthly.sel(time=slice(BASELINE_START, BASELINE_END))
obs_clim = obs_baseline.groupby("time.month").mean()
hist_model = get_sst_series(FILE_HIST)
hist_monthly = hist_model.resample(time="1MS").mean()
hist_baseline = hist_monthly.sel(time=slice(BASELINE_START, BASELINE_END))
hist_clim = hist_baseline.groupby("time.month").mean()
print(
    f"\n>>> Observed annual baseline (1995-2014): "
    f"{float(obs_clim.mean()):.2f} degC"
)
print(
    f">>> Model annual baseline (1995-2014): "
    f"{float(hist_clim.mean()):.2f} degC"
)

# =========================================================
# 6. CMEMS SMOOTHING & TREND
# =========================================================
LAST_CMEMS_DATE = obs_monthly.time.max().values
# Year used to label the observed / projected transition line
SPLIT_YEAR = pd.to_datetime(LAST_CMEMS_DATE).year
print(f"\n>>> Last CMEMS data point: {LAST_CMEMS_DATE}")
print(f">>> Transition year label: {SPLIT_YEAR}")
obs_smooth = obs_monthly.rolling(time=12, center=True).mean()
obs_t, obs_trend, obs_dec_trend = calculate_trend(
    obs_monthly.time.values,
    obs_monthly.values
)

# =========================================================
# 7. FIGURE SETUP
# =========================================================
fig, axes = plt.subplots(3, 1, figsize=(17, 13), sharex=True)
# leave room on the right when the legend sits outside the panels
_right = 0.80 if LEGEND_POSITION == "outside" else 0.97
plt.subplots_adjust(
    hspace=0.20, left=0.08, right=_right, top=0.93, bottom=0.08
)
# collects the data range across all panels so the y-axis can be
# fitted tightly to the values actually plotted
data_min = np.inf
data_max = -np.inf

# =========================================================
# 8. LOOP FUTURE SCENARIOS
# =========================================================
for ax, (scenario, file_fut) in zip(axes, FILES_FUT.items()):
    print(f"\n>>> Processing {scenario}")
    fut_model = get_sst_series(file_fut)
    fut_monthly = fut_model.resample(time="1MS").mean()
    # DELTA CORRECTION (MONTH-BY-MONTH)
    fut_anomaly = fut_monthly.groupby("time.month") - hist_clim
    fut_corrected = fut_anomaly.groupby("time.month") + obs_clim
    # FILTER AFTER CMEMS
    fut_corrected = fut_corrected.sel(time=slice(LAST_CMEMS_DATE, None))
    # SMOOTH & TREND
    fut_smooth = fut_corrected.rolling(time=12, center=True).mean()
    fut_t, fut_trend, fut_dec_trend = calculate_trend(
        fut_corrected.time.values,
        fut_corrected.values
    )
    # track the range of everything actually drawn on this panel
    for _arr in (obs_monthly.values, fut_corrected.values):
        if np.any(np.isfinite(_arr)):
            data_min = min(data_min, float(np.nanmin(_arr)))
            data_max = max(data_max, float(np.nanmax(_arr)))
    # PLOT MONTHLY RAW (BACKGROUND)
    ax.plot(obs_monthly.time, obs_monthly.values,
            color="black", alpha=BG_ALPHA_OBS, linewidth=BG_LINEWIDTH,
            zorder=1)
    ax.plot(fut_corrected.time, fut_corrected.values,
            color=COLORS[scenario], alpha=BG_ALPHA_FUT,
            linewidth=BG_LINEWIDTH, zorder=1)
    # PLOT SMOOTHED (SOLID)
    ax.plot(obs_smooth.time, obs_smooth.values,
            color="black", linewidth=2.5,
            label="Observed CMEMS (12m mean)", zorder=3)
    ax.plot(fut_smooth.time, fut_smooth.values,
            color=COLORS[scenario], linewidth=2.5,
            label=f"{scenario} (12m mean)", zorder=3)
    # PLOT REGRESSION TRENDS (DASHED)
    ax.plot(obs_t, obs_trend, color="black", linestyle="--", linewidth=2.5,
            label=f"Obs Trend (+{obs_dec_trend:.2f}\u00b0C / dec)", zorder=4)
    ax.plot(fut_t, fut_trend, color=COLORS[scenario], linestyle="--",
            linewidth=2.5,
            label=f"Fut Trend (+{fut_dec_trend:.2f}\u00b0C / dec)", zorder=4)
    # TRANSITION LINE
    ax.axvline(pd.to_datetime(LAST_CMEMS_DATE),
               color="gray", linestyle=":", linewidth=2, zorder=2)
    # YEAR LABEL ON THE TRANSITION LINE
    ax.annotate(
        str(SPLIT_YEAR),
        xy=(pd.to_datetime(LAST_CMEMS_DATE), SPLIT_LABEL_HEIGHT),
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
    # STYLE
    ax.set_title(scenario, fontsize=FS_PANEL_TITLE,
                 fontweight="bold", loc="left")
    ax.set_ylabel("Monthly SST (\u00b0C)", fontsize=FS_YLABEL)
    ax.grid(alpha=0.2, linestyle="--")
    # explicit decade ticks so the observed period is labelled
    ax.set_xticks([
        pd.Timestamp(f"{y}-01-01")
        for y in range(XTICK_START, XTICK_END + 1, XTICK_STEP)
    ])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", labelsize=FS_TICKS, rotation=0)
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
# 8b. Y-AXIS LIMITS
# =========================================================
# Fitted to the plotted data so the lines fill the panel height.
if YLIM is None:
    _lo = np.floor(data_min - YPAD)
    _hi = np.ceil(data_max + YPAD)
else:
    _lo, _hi = YLIM
print(f"\n>>> Y-axis range: {_lo:.1f} to {_hi:.1f} degC")
for ax in axes:
    ax.set_ylim(_lo, _hi)

# =========================================================
# 9. X LABEL
# =========================================================
axes[-1].set_xlabel("Year", fontsize=FS_XLABEL, fontweight="bold")

# =========================================================
# 10. TITLE
# =========================================================
fig.suptitle("Cabrera Monthly Sea Surface Temperature\n",
             fontsize=FS_SUPTITLE, fontweight="bold")

# =========================================================
# 11. SAVE
# =========================================================
plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"\n>>> Figure saved:")
print(output_file)
plt.close()
print("\n>>> Monthly SST regression trajectories generated successfully.")
