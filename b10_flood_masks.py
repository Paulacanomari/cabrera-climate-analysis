# =========================================================
# CABRERA MULTI-SCENARIO SEA LEVEL RISE FLOOD MASKS
# IPCC AR6 Regional Projections
# Export GeoTIFFs aligned to EPSG:25831
# =========================================================

from pathlib import Path

import rasterio
import numpy as np

# =========================================================
# 1. PATHS
# Set DATA_DIR to the folder holding the downloaded datasets.
# See the README for the products and where to obtain them.
# =========================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

# MDT02 LiDAR digital terrain model, 2 m, ETRS89 / UTM zone 31N
DEM_PATH = DATA_DIR / "cabrera_dem_2m.tif"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. SLR SCENARIOS (meters)
# Regional AR6 projections for Cabrera
# =========================================================

SCENARIOS = {

    "ssp126_2060": 0.27,
    "ssp126_2100": 0.49,

    "ssp245_2060": 0.30,
    "ssp245_2100": 0.59,

    "ssp585_2060": 0.34,
    "ssp585_2100": 0.80

}

# =========================================================
# 3. LOAD DEM
# =========================================================

with rasterio.open(DEM_PATH) as src:

    dem = src.read(1)

    profile = src.profile

    transform = src.transform

    crs = src.crs

    nodata = src.nodata

# =========================================================
# 4. PRINT DEM INFO
# =========================================================

print("\n========================================")
print("DEM INFORMATION")
print("========================================")

print("\nCRS:")
print(crs)

print("\nShape:")
print(dem.shape)

print("\nNoData:")
print(nodata)

# =========================================================
# 5. CLEAN NODATA
# =========================================================

dem = np.where(

    dem == nodata,

    np.nan,

    dem

)

# =========================================================
# 6. UPDATE OUTPUT PROFILE
# =========================================================

profile.update(

    dtype="float32",

    nodata=np.nan,

    compress="lzw",

    crs="EPSG:25831"

)

# =========================================================
# 7. LOOP THROUGH SCENARIOS
# =========================================================

for scenario, slr in SCENARIOS.items():

    print("\n========================================")
    print(f"PROCESSING: {scenario}")
    print(f"SLR = {slr:.2f} m")
    print("========================================")

    # =====================================================
    # CREATE FLOOD MASK
    # =====================================================
    # Flooded if:
    # 0 m < elevation <= SLR
    #
    # Static bathtub model: hydrological connectivity is not
    # considered, so isolated inland depressions below the
    # projected sea level are also flagged.
    # =====================================================

    flooded = (

        (dem > 0.0) &

        (dem <= slr)

    )

    # =====================================================
    # CONVERT TO FLOAT
    # =====================================================
    # Flooded = 1
    # Non-flooded = NaN
    # =====================================================

    flooded = np.where(

        flooded,

        1,

        np.nan

    ).astype("float32")

    # =====================================================
    # OUTPUT FILE
    # =====================================================

    output_path = OUTPUT_DIR / f"flood_{scenario}.tif"

    # =====================================================
    # EXPORT FLOOD MASK
    # =====================================================

    with rasterio.open(

        output_path,

        "w",

        **profile

    ) as dst:

        dst.write(

            flooded,

            1

        )

    # =====================================================
    # PRINT OUTPUT INFO
    # =====================================================

    flooded_pixels = np.sum(

        ~np.isnan(flooded)

    )

    print("\nFlood mask exported:")
    print(output_path)

    print("\nFlooded pixels:")
    print(flooded_pixels)

# =========================================================
# 8. DONE
# =========================================================

print("\n========================================")
print("ALL SCENARIOS COMPLETED")
print("========================================")
