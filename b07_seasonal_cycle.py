# =========================================================
# CABRERA ANNUAL SST PROJECTIONS ANALYSIS
# Seasonal cycle delta-corrected onto CMEMS baseline (per month)
# Saved to figures/SST/projections
# =========================================================

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[3])
)

import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# =========================================================
# 1. PATHS
# =========================================================

from config.paths import (
    CMIP6_HISTORICAL_FILE,
    CMIP6_SST_DIR,
    CMEMS_SST_DIR,
    SST_PROJECTIONS_FIGURES_DIR
)

# =========================================================
# HISTORICAL
# =========================================================

FILE_HIST = CMIP6_HISTORICAL_FILE

# =========================================================
# CMEMS OBSERVATIONS
# =========================================================

FILE_CMEMS = (
    CMEMS_SST_DIR /
    "cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021_1777296864009.nc"
)

# =========================================================
# SST PROJECTIONS
# =========================================================

BASE_FOLDER = CMIP6_SST_DIR

FILES_FUT = {
    ("ssp126", "mid"): BASE_FOLDER / "tos_Omon_CNRM-CM6-1-HR_ssp126_r1i1p1f2_gn_20400116-20601216.nc",
    ("ssp126", "end"): BASE_FOLDER / "tos_Omon_CNRM-CM6-1-HR_ssp126_r1i1p1f2_gn_20800116-21001216.nc",
    ("ssp245", "mid"): BASE_FOLDER / "tos_Omon_CNRM-CM6-1-HR_ssp245_r1i1p1f2_gn_20400116-20601216.nc",
    ("ssp245", "end"): BASE_FOLDER / "tos_Omon_CNRM-CM6-1-HR_ssp245_r1i1p1f2_gn_20800116-21001216.nc",
    ("ssp585", "mid"): BASE_FOLDER / "tos_Omon_CNRM-CM6-1-HR_ssp585_r1i1p1f2_gn_20400116-20601216.nc",
    ("ssp585", "end"): BASE_FOLDER / "tos_Omon_CNRM-CM6-1-HR_ssp585_r1i1p1f2_gn_20800116-21001216.nc",
}

# =========================================================
# FIGURES
# =========================================================

FIG_DIR = SST_PROJECTIONS_FIGURES_DIR

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# =========================================================
# 2. CABRERA REGION
# =========================================================

LAT_MIN, LAT_MAX = 38.8, 39.6
LON_MIN, LON_MAX = 2.4, 3.6

# =========================================================
# MODEL
# =========================================================

MODEL = "CNRM-CM6-1-HR"

BASELINE_START = "1995-01-01"
BASELINE_END = "2014-12-31"

# 20-year windows, consistent across mid and end
PERIOD_WIN = {
    "mid": ("2041-01-01", "2060-12-31"),
    "end": ("2081-01-01", "2100-12-31"),
}

PERIOD_LABEL = {
    "mid": "2041-2060",
    "end": "2081-2100",
}

SCENARIO_NAME = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp585": "SSP5-8.5"
}

COLORS = {
    "SSP1-2.6": "#1d3354",
    "SSP2-4.5": "#cc9900",
    "SSP5-8.5": "#840b22"
}

# =========================================================
# 3. FUNCTIONS
# =========================================================

def get_sst_series(file_path):

    print(f">>> Reading: {file_path.name}")

    with xr.open_dataset(file_path, decode_cf=True) as ds:

        # VARIABLE
        if "tos" in ds.data_vars:
            var_name = "tos"
        elif "analysed_sst" in ds.data_vars:
            var_name = "analysed_sst"
        else:
            var_name = "sst"

        sst = ds[var_name]

        # COORDINATES
        lat_n = "lat" if "lat" in ds.variables else "latitude"
        lon_n = "lon" if "lon" in ds.variables else "longitude"

        # SPATIAL MASK
        mask = (
            (ds[lat_n] >= LAT_MIN) &
            (ds[lat_n] <= LAT_MAX) &
            (ds[lon_n] >= LON_MIN) &
            (ds[lon_n] <= LON_MAX)
        )

        sst_region = sst.where(mask)

        # HORIZONTAL DIMS
        horiz_dims = [d for d in sst.dims if d != "time"]

        # SPATIAL MEAN
        series = sst_region.mean(dim=horiz_dims, skipna=True).compute()

        # K -> degC
        if float(series.mean()) > 100:
            series = series - 273.15

    return series


def monthly_climatology(series, t0=None, t1=None):
    """Monthly mean and standard deviation climatology (12 values,
    indexed by month). If t0/t1 are given, restrict to that window first.
    The standard deviation is the interannual spread within the window,
    computed separately for each calendar month."""
    if t0 is not None and t1 is not None:
        series = series.sel(time=slice(t0, t1))
    clim_mean = series.groupby("time.month").mean()
    clim_std = series.groupby("time.month").std()
    return clim_mean, clim_std

# =========================================================
# 4. PLOT FUNCTIONS
# =========================================================

def plot_enhanced_deltas(df):

    plt.figure(figsize=(10, 6))

    sns.set_theme(style="whitegrid")

    df = df.copy()

    df["scenario"] = pd.Categorical(
        df["scenario"],
        ["SSP1-2.6", "SSP2-4.5", "SSP5-8.5"]
    )

    df = df.sort_values(["scenario", "period"])

    ax = sns.barplot(
        data=df,
        x="scenario",
        y="delta_annual_C",
        hue="period",
        palette="viridis"
    )

    for i, patch in enumerate(ax.patches):

        height = patch.get_height()

        if abs(height) > 0.01:
            ax.annotate(
                f"+{height:.2f}\u00b0C",
                (patch.get_x() + patch.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=9,
                weight="bold",
                xytext=(0, 10),
                textcoords="offset points"
            )

            std_val = df.iloc[i]["std_annual_C"]

            plt.errorbar(
                patch.get_x() + patch.get_width() / 2,
                height,
                yerr=std_val,
                fmt="none",
                c="black",
                capsize=5,
                lw=1
            )

    plt.title(f"Projected Annual SST Warming in Cabrera ({MODEL})", fontsize=13)
    plt.ylabel("\u0394 Annual SST (\u00b0C) \u00b1 1 interannual SD")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "Cabrera_Annual_SST_Deltas_Stats.png", dpi=300)
    plt.close()

# =========================================================
# DELTA-CORRECTED SEASONAL CYCLE
# =========================================================

def plot_seasonal_cmems_corrected(
    hist_clim_mean,
    obs_clim_mean,
    obs_clim_std,
    fut_dict,
    period_key="end"
):
    """
    Seasonal cycle in absolute degC, delta-corrected onto the CMEMS baseline.

    For each month m:
        corrected[m] = obs_clim_mean[m] + (fut_month[m] - hist_clim_mean[m])

    - obs_clim_mean : CMEMS observed monthly climatology (1995-2014)   -> baseline curve
    - hist_clim_mean: model historical monthly climatology (1995-2014) -> removes model bias
    - fut_month     : model future monthly climatology, restricted to PERIOD_WIN
    """

    plt.figure(figsize=(11, 6))
    sns.set_theme(style="ticks")

    t0, t1 = PERIOD_WIN[period_key]

    # --- Observed baseline curve (CMEMS) ---
    plt.plot(
        obs_clim_mean.month, obs_clim_mean.values,
        color="black", linestyle="--", marker="o", linewidth=2,
        label="Observed CMEMS baseline (1995-2014)"
    )
    plt.fill_between(
        obs_clim_mean.month,
        obs_clim_mean - obs_clim_std,
        obs_clim_mean + obs_clim_std,
        color="black", alpha=0.10
    )

    title_period = "Mid-Century" if period_key == "mid" else "End-Century"

    for sc_key in ["ssp126", "ssp245", "ssp585"]:
        fpath = fut_dict.get((sc_key, period_key))

        if fpath:
            fut_series = get_sst_series(fpath)

            # restricted to the same window used for the delta table
            fut_clim_mean, fut_clim_std = monthly_climatology(
                fut_series, t0, t1
            )

            # --- per-month delta correction onto CMEMS ---
            corrected = obs_clim_mean + (fut_clim_mean - hist_clim_mean)

            color = COLORS[SCENARIO_NAME[sc_key]]

            plt.plot(
                corrected.month, corrected.values,
                color=color, marker="s", linewidth=1.5,
                label=f"{SCENARIO_NAME[sc_key]}"
            )
            # interannual variability band (model within-period SD)
            plt.fill_between(
                corrected.month,
                corrected - fut_clim_std,
                corrected + fut_clim_std,
                color=color, alpha=0.15
            )

    plt.xticks(
        range(1, 13),
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        fontsize=12
    )
    plt.yticks(fontsize=12)
    plt.title(
        f"SST Seasonal Cycle - {title_period} ({PERIOD_LABEL[period_key]})",
        fontsize=15
    )
    plt.ylabel("Absolute SST (\u00b0C)", fontsize=13)
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=13,
        frameon=True
    )
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / f"Cabrera_SST_Seasonal_CMEMS_{period_key}.png",
        dpi=300
    )
    plt.close()

# =========================================================

def plot_anomaly_monthly(sst_hist, fut_dict, period_key="end"):
    # Anomalies are bias-free by construction, so this stays model-frame.

    plt.figure(figsize=(11, 6))
    sns.set_theme(style="whitegrid")

    t0, t1 = PERIOD_WIN[period_key]

    h_month_mean = sst_hist.sel(
        time=slice(BASELINE_START, BASELINE_END)
    ).groupby("time.month").mean()

    title_period = "Mid-Century" if period_key == "mid" else "End-Century"

    for sc_key in ["ssp126", "ssp245", "ssp585"]:
        fpath = fut_dict.get((sc_key, period_key))

        if fpath:
            f_month_mean = (
                get_sst_series(fpath)
                .sel(time=slice(t0, t1))
                .groupby("time.month")
                .mean()
            )
            anomaly = f_month_mean - h_month_mean

            plt.plot(
                anomaly.month, anomaly.values, marker="o", linewidth=2.5,
                label=SCENARIO_NAME[sc_key], color=COLORS[SCENARIO_NAME[sc_key]]
            )

    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.xticks(
        range(1, 13),
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    plt.ylabel("\u0394 SST (\u00b0C) vs 1995-2014")
    plt.title(
        f"SST Monthly Anomalies - {title_period} ({PERIOD_LABEL[period_key]})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIG_DIR / f"Cabrera_SST_Anomalies_Monthly_{period_key}.png",
        dpi=300
    )
    plt.close()

# =========================================================

def plot_anomaly_timeseries(sst_hist, fut_dict):

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="ticks")

    baseline_val = sst_hist.sel(
        time=slice(BASELINE_START, BASELINE_END)
    ).mean().item()

    # HISTORICAL
    h_anom = (sst_hist - baseline_val).rolling(time=12, center=True).mean()

    plt.plot(
        h_anom.time, h_anom.values, color="black",
        linewidth=2, label="Historical"
    )

    # FUTURE
    for sc_key in ["ssp126", "ssp245", "ssp585"]:
        mid_path = fut_dict.get((sc_key, "mid"))
        end_path = fut_dict.get((sc_key, "end"))

        if mid_path and end_path:
            s_mid = get_sst_series(mid_path)
            s_end = get_sst_series(end_path)
            s_combined = xr.concat([s_mid, s_end], dim="time")

            f_anom = (s_combined - baseline_val).rolling(
                time=12, center=True
            ).mean()

            plt.plot(
                f_anom.time, f_anom.values,
                color=COLORS[SCENARIO_NAME[sc_key]],
                linewidth=1.5, label=SCENARIO_NAME[sc_key]
            )

    plt.axhline(0, color="red", linestyle="--", alpha=0.3)
    plt.ylabel("Annual SST Anomaly (\u00b0C)")
    plt.title(
        "Cabrera Annual SST Anomaly Trend (1995-2100) [12-month rolling mean]"
    )
    plt.legend(loc="upper left", ncol=2, fontsize="small")
    sns.despine()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Cabrera_Annual_SST_Anomaly_Trend.png", dpi=300)
    plt.close()

# =========================================================
# 5. MAIN
# =========================================================

if __name__ == "__main__":

    print("\n>>> Initializing ANNUAL SST analysis...")

    # HISTORICAL (model) BASELINE
    sst_hist = get_sst_series(FILE_HIST)

    baseline_mean = sst_hist.sel(
        time=slice(BASELINE_START, BASELINE_END)
    ).mean().item()

    print(f"    Model baseline annual mean (1995-2014): {baseline_mean:.2f}\u00b0C")

    # MODEL historical monthly climatology (for bias removal)
    hist_clim_mean, _ = monthly_climatology(
        sst_hist, BASELINE_START, BASELINE_END
    )

    # CMEMS OBSERVED baseline monthly climatology
    print("\n>>> Reading CMEMS observations for baseline")
    sst_cmems = get_sst_series(FILE_CMEMS)
    obs_clim_mean, obs_clim_std = monthly_climatology(
        sst_cmems, BASELINE_START, BASELINE_END
    )

    print(
        f"    Observed CMEMS annual mean (1995-2014): "
        f"{float(obs_clim_mean.mean()):.2f}\u00b0C"
    )

    # SUMMARY TABLE (ANNUAL deltas, model-frame)
    rows = []

    for (sc_key, per_key), fpath in FILES_FUT.items():

        print(f"--- Processing {sc_key} ({per_key}: {PERIOD_LABEL[per_key]})")

        sst_fut = get_sst_series(fpath)

        t0, t1 = PERIOD_WIN[per_key]

        f_window = sst_fut.sel(time=slice(t0, t1))

        # sanity check: confirm the window length in years
        n_months = f_window.sizes["time"]

        print(f"    months in window: {n_months} ({n_months / 12:.1f} years)")

        rows.append({
            "scenario": SCENARIO_NAME[sc_key],
            "period": per_key,
            "years": PERIOD_LABEL[per_key],
            "delta_annual_C": (f_window.mean().item() - baseline_mean),
            "std_annual_C": f_window.std().item()
        })

    df = pd.DataFrame(rows)

    print("\n>>> Generating seasonal cycle figures...")

    # Delta-corrected seasonal cycles (mid- and end-century)
    plot_seasonal_cmems_corrected(
        hist_clim_mean, obs_clim_mean, obs_clim_std, FILES_FUT, "mid"
    )
    plot_seasonal_cmems_corrected(
        hist_clim_mean, obs_clim_mean, obs_clim_std, FILES_FUT, "end"
    )

    print(f"\n>>> All figures saved in:\n{FIG_DIR}")

    print("\n>>> EXTRACTED ANNUAL DELTAS (model-frame):")
    print(df.to_string(index=False))