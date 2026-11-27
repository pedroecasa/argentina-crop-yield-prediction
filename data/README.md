# Data

| File | Description |
|---|---|
| `estimaciones-agricolas-2026-03.csv` | Raw input: MAGyP agricultural estimates (`;`-separated), as downloaded in March 2026 |
| `dataset_agro_clima_con_ratio_perdida.csv` | Processed dataset used by the notebook and the app (109,402 rows × 18 columns, seasons 1981–2024) |

## Sources

- **MAGyP – Estimaciones Agrícolas:** https://datos.magyp.gob.ar/dataset/estimaciones-agricolas
- **NASA POWER API** (daily climate, community AG): https://power.larc.nasa.gov/
- **Georef API** (department centroids): https://apis.datos.gob.ar/georef

## Rebuilding the processed dataset

```bash
cd data
python ../src/build_dataset.py
```

The script reads the raw MAGyP CSV, downloads the daily climate series for each department (cached as Parquet in `cache_clima/`) and writes `dataset_agro_clima_final.csv`. The processed file in this folder also adds the column `pct_perdida_area` (area-loss ratio × 100).

The first run takes a while because it queries NASA POWER for every department; later runs use the local cache.
