# Cabrera climate risk analysis

Analysis code for a master's thesis applying the LIFE INTEMARES climate-risk
assessment methodology to Natura 2000 site ES0000083 (Arxipèlag de Cabrera),
Balearic Islands, Spain.

**Author:** Paula Cano Marí, 2026

Sea surface temperature and sea-level rise are assessed under SSP1-2.6,
SSP2-4.5 and SSP5-8.5 for mid-century (2041–2060) and end-century (2081–2100),
relative to a 1995–2014 baseline. The scripts produce the figures and tables of
the thesis. Vulnerability scoring was done from published literature and is
documented in Appendix B of the thesis, not here.

## Data

No data is included. All datasets are openly available:

- **Observed SST** — Mediterranean Sea High Resolution L4 SST Reprocessed,
  Copernicus Marine Service.
- **Projected SST** — CMIP6 `tos`, CNRM-CM6-1-HR r1i1p1f2, via the Copernicus
  Climate Data Store.
- **Sea level** — IPCC AR6 regional projections, medium confidence
  (Garner et al., 2021).
- **Elevation** — MDT02 LiDAR DEM, 2 m, Instituto Geográfico Nacional.
- **Marine habitats** — EUSeaMap 2025, EMODnet Seabed Habitats (CC BY 4.0).
- **Coastal habitats** — Habitats of Community Interest 2022,
  Govern de les Illes Balears.
- **Coastline** — Línea de Costa, © Instituto Hidrográfico de la Marina.

Each script sets DATA_DIR at the top. Point it at the folder holding the downloaded files.

## Scripts

| Script | Produces |
|---|---|
| `b06` | Figure 7, Table 10 |
| `b07` | Figures 8, 9 |
| `b08` | Figure 10, Table 11 |
| `b09` | Figure 11, Table 12 |
| `b10` | Inundation masks used in Figures 15–17 |
| `b11` | Figures 12, 13 |
| `b12` | Figure 14 |

The per-habitat flooded percentages in Table B1 were computed in QGIS by
intersecting the habitat polygons with the masks from `b10`. That step is not
scripted.

## Acknowledgements

This study has been conducted using E.U. Copernicus Marine Service Information.
I acknowledge the World Climate Research Programme, which through its Working
Group on Coupled Modelling coordinated CMIP6, and thank the climate modelling
groups, the Earth System Grid Federation, and the agencies supporting CMIP6.
