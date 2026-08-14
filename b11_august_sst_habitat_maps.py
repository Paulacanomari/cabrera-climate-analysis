# =========================================================
# POSIDONIA + FUTURE SST HABITAT MAPS  --  AUGUST
# Clean version: coloured SST cells + habitats only.
# NO contour lines, NO interpolation. Diverging colour centered on 28 C.
# =========================================================

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import geopandas as gpd
from matplotlib.colors import TwoSlopeNorm
from shapely.geometry import box, Point
from shapely.ops import polygonize
from matplotlib.patches import Patch

from config.paths import (
    CMEMS_SST_DIR, COSTA_DIR, SST_HABITATS_FIGURES_DIR
)

OUTPUT_DIR = SST_HABITATS_FIGURES_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILE_CMEMS = (
    CMEMS_SST_DIR /
    "cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021_1777296864009.nc"
)
PATH_COAST = COSTA_DIR / "COSTA.shp"
PATH_HABITATS = COSTA_DIR / "cabrera_habitat" / "marine_habitats.shp"

# =========================================================
# PARAMETERS
# =========================================================

BASELINE_START = "1995-01-01"
BASELINE_END = "2014-12-31"
TARGET_MONTHS = [8]                       # AUGUST

# Corrected August deltas (baseline 26.54 C)
MID_CENTURY_DELTAS = {"SSP1-2.6": 1.53, "SSP2-4.5": 1.55, "SSP5-8.5": 2.00}
END_CENTURY_DELTAS = {"SSP1-2.6": 1.91, "SSP2-4.5": 2.43, "SSP5-8.5": 4.82}

THRESHOLD = 28.0
NORM = None                     # plain linear scale
CMAP = "turbo"                 # strong contrast; every panel a distinct colour

# ---------------------------------------------------------
# COLOUR SCALE
# ---------------------------------------------------------
# With the observed baseline removed, the scale can be fitted to the
# three scenario fields alone. That is a much narrower range than
# 26-31 C, so the within-cell variation occupies far more of the
# colour range and the contrast between cells is visible.
#
# AUTO_SCALE = True   -> fitted to each figure's own scenarios
# AUTO_SCALE = False  -> uses the fixed VMIN / VMAX below

AUTO_SCALE = True

VMIN, VMAX = 26.0, 31.0        # only used when AUTO_SCALE = False

# ---------------------------------------------------------
# TEXT SIZES  (raise or lower these as needed)
# ---------------------------------------------------------

FS_CELL = 20         # SST value printed in each cell
FS_PANEL_TITLE = 26  # per-panel title
FS_SUPTITLE = 32     # overall figure title
FS_CBAR_LABEL = 24   # colour bar label
FS_CBAR_TICKS = 20   # colour bar tick numbers
FS_LEGEND = 24       # habitat legend
FS_AXIS = 19         # latitude / longitude tick labels

# Label every STEP-th cell in each direction.
# STEP = 1 labels every cell, 2 labels every other cell, etc.
STEP = 2

# ---------------------------------------------------------
# PANEL SPACING
# ---------------------------------------------------------
# w_pad / h_pad are the padding around each panel, in inches.
# wspace / hspace are the extra gap between panels, as a fraction
# of the figure width / height. Raise these for more separation.

W_PAD = 0.5
H_PAD = 0.5
WSPACE = 0.06
HSPACE = 0.06

# Overall canvas size in inches, for the three-panel row.
# Larger values = larger maps. The fonts are set in points, so
# enlarging the canvas makes the MAPS bigger relative to the text.
FIGSIZE = (32, 12)

# ---------------------------------------------------------
# TITLE AND LEGEND POSITION
# ---------------------------------------------------------
# SUPTITLE_Y: vertical position of the main title, as a fraction of
#   the figure height. Default is about 0.98. LOWER it to bring the
#   title DOWN, closer to the maps.
#
# LEGEND_Y: vertical position of the habitat legend. It is negative
#   because the legend sits below the panels. Move it TOWARDS ZERO
#   to bring the legend UP, closer to the maps.

SUPTITLE_Y = 0.99
LEGEND_Y = 0.10

# =========================================================
# SHAPEFILES
# =========================================================

coast = gpd.read_file(PATH_COAST)
habitats = gpd.read_file(PATH_HABITATS)
for gdf in [coast, habitats]:
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    else:
        gdf.to_crs("EPSG:4326", inplace=True)

balearic_zoom = [2.70, 3.15, 39.05, 39.42]
bbox_geom = box(balearic_zoom[0], balearic_zoom[2], balearic_zoom[1], balearic_zoom[3])
coast_clip = gpd.clip(coast, bbox_geom)
habitats_clip = gpd.clip(habitats, bbox_geom).copy()

merged_geo = coast_clip.geometry.union_all().union(bbox_geom.boundary)
polys = list(polygonize(merged_geo))
sea_point = Point(2.9, 39.25)
land = gpd.GeoDataFrame(
    geometry=[p for p in polys if not p.contains(sea_point)], crs="EPSG:4326"
)

filter_column = "EUNIS_N2"
habitat_filters = {
    "Posidonia": ("#00cc44", "Posidonia beds"),
    "Sandbanks": ("#b08d57", "Sandbanks"),
    "coralligenous": ("#7b0d1e", "Reefs (coralligenous)"),
    "detritic": ("#8c510a", "Reefs (detritic)"),
    "Reefs": ("#ff9999", "Reefs"),
}
habitat_legend = [
    Patch(facecolor=d[0], label=d[1], alpha=0.75, edgecolor="black", linewidth=0.5)
    for d in habitat_filters.values()
]

# =========================================================
# CMEMS AUGUST CLIMATOLOGY
# =========================================================

def get_cmems_august_climatology():
    ds = xr.open_dataset(FILE_CMEMS)
    d = ds["analysed_sst"]
    if float(d.max()) > 100:
        d = d - 273.15
    return (d.sel(time=slice(BASELINE_START, BASELINE_END))
              .where(d["time.month"].isin(TARGET_MONTHS), drop=True)
              .mean("time"))

sst_obs = get_cmems_august_climatology()

# =========================================================
# PANEL  (flat cells, no contours, no smoothing)
# =========================================================

def plot_map(ax, sst_data, title, vmin, vmax):
    ax.set_facecolor("black")

    im = ax.pcolormesh(
        sst_data.longitude, sst_data.latitude, sst_data.values,
        vmin=vmin, vmax=vmax, cmap=CMAP,
        shading="nearest",     # each cell = solid block, no interpolation
        zorder=1
    )

    # --- annotate cells with SST value (thinned + larger for readability) ---
    lons = sst_data.longitude.values
    lats = sst_data.latitude.values
    vals = sst_data.values
    x0, x1, y0, y1 = balearic_zoom
    for iy in range(vals.shape[0]):
        lat = lats[iy]
        if not (y0 <= lat <= y1):
            continue
        for ix in range(vals.shape[1]):
            lon = lons[ix]
            if not (x0 <= lon <= x1):
                continue
            if (iy % STEP != 0) or (ix % STEP != 0):
                continue
            v = vals[iy, ix]
            if np.isfinite(v):
                ax.text(lon, lat, f"{v:.1f}",
                        ha="center", va="center",
                        fontsize=FS_CELL, fontweight="bold",
                        color="white", zorder=4, clip_on=True,
                        path_effects=[pe.withStroke(linewidth=3.0, foreground="black")])

    if not habitats_clip.empty:
        habitats_clip[filter_column] = habitats_clip[filter_column].astype(str)
        for keyword, (color, _) in habitat_filters.items():
            if keyword == "Reefs":
                mask = (habitats_clip[filter_column].str.contains("Reefs", case=False, na=False)
                        & ~habitats_clip[filter_column].str.contains("coralligenous", case=False, na=False)
                        & ~habitats_clip[filter_column].str.contains("detritic", case=False, na=False))
            else:
                mask = habitats_clip[filter_column].str.contains(keyword, case=False, na=False)
            subset = habitats_clip[mask]
            if not subset.empty:
                subset.plot(ax=ax, facecolor=color, alpha=0.6,
                            edgecolor="black", linewidth=0.3, zorder=2)

    if not land.empty:
        land.plot(ax=ax, facecolor="#1a1a1a", edgecolor="none", zorder=10)
    coast_clip.plot(ax=ax, color="#1a1a1a", linewidth=0.6, zorder=11)

    ax.set_xlim(balearic_zoom[0], balearic_zoom[1])
    ax.set_ylim(balearic_zoom[2], balearic_zoom[3])
    ax.tick_params(axis="both", labelsize=FS_AXIS)
    ax.set_title(title, fontsize=FS_PANEL_TITLE, fontweight="bold")
    return im

# =========================================================
# FIGURE
# =========================================================

def create_figure(deltas, period_label, output_name):

    # the three scenario fields for this period
    fields = {scen: sst_obs + d for scen, d in deltas.items()}

    if AUTO_SCALE:
        vmin = min(float(np.nanmin(f.values)) for f in fields.values())
        vmax = max(float(np.nanmax(f.values)) for f in fields.values())
    else:
        vmin, vmax = VMIN, VMAX

    print(f"\n{period_label} colour scale: {vmin:.2f} to {vmax:.2f} degC "
          f"(span {vmax - vmin:.2f} degC)")

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, constrained_layout=True)

    # space between panels (in inches). Raise these if the panels
    # still feel tight; lower them to pack the figure more densely.
    fig.get_layout_engine().set(
        w_pad=W_PAD, h_pad=H_PAD, wspace=WSPACE, hspace=HSPACE
    )

    for ax, (scen, field) in zip(axes, fields.items()):
        im = plot_map(
            ax, field,
            f"{scen} {period_label}\n(+{deltas[scen]:.2f} \u00b0C)",
            vmin, vmax
        )

    cbar = fig.colorbar(im, ax=axes, orientation="vertical",
                        shrink=0.8, pad=0.02, extend="both")
    cbar.set_label("August SST (\u00b0C)", fontsize=FS_CBAR_LABEL,
                   fontweight="bold")
    cbar.ax.tick_params(labelsize=FS_CBAR_TICKS)
    if vmin < THRESHOLD < vmax:
        cbar.ax.axhline(THRESHOLD, color="black", linewidth=2.5)   # mark 28 C

    fig.legend(handles=habitat_legend, title="Marine habitats",
               loc="upper center", ncol=5, fontsize=FS_LEGEND,
               title_fontsize=FS_LEGEND + 4,
               frameon=True, facecolor="white", edgecolor="black",
               bbox_to_anchor=(0.5, LEGEND_Y),
               handlelength=2.0, handleheight=1.1,
               borderpad=0.6, labelspacing=0.5,
               columnspacing=1.4, handletextpad=0.7)
    fig.suptitle(
        f"Projected August SST and marine habitats \u2013 {period_label}",
        fontsize=FS_SUPTITLE, fontweight="bold", y=SUPTITLE_Y
    )

    out = OUTPUT_DIR / output_name
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Figure saved:\n{out}")
    plt.close()


create_figure(MID_CENTURY_DELTAS, "Mid-Century",
              "Future_August_SST_Habitats_Cabrera_Mid.png")
create_figure(END_CENTURY_DELTAS, "End-Century",
              "Future_August_SST_Habitats_Cabrera_End.png")
print("\nMaps generated.")