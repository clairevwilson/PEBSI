# Climate Data

PEBSI accepts two types of climate forcing: reanalysis data (MERRA-2) and on-glacier weather station data (AWS). You need at least one.

## MERRA-2

Code in `PEBSI/shop/data_retrieval/` walks you through downloading and processing MERRA-2 data into the format expected by PEBSI. The workflow is optimized to fetch as much data from NASA EarthData servers as possible in a single pass.

!!! tip "Download times"
    As a rough benchmark: the full Alaska region from 1980–2026 took approximately **14 hours** to download and **2 hours** to aggregate.

## AWS Preprocessing

Preprocessing notebooks are in `PEBSI/shop/preprocessing/`. PEBSI expects a comma-separated file with:

- Columns named after the PEBSI climate variable names
- Values in the correct units
- An hourly datetime index in UTC

!!! note
    Real AWS data is rarely clean. Expect to adapt the notebooks to your specific data format.

**`preprocess_AWS`** — for data split across multiple files with significant gaps. Walks through:

1. Concatenating files
2. Renaming columns to be self-consistent
3. Identifying periods viable for simulation (i.e., gaps small enough for interpolation)

**`process_AWS`** — for the final preparation steps:

1. Interpolating NaNs
2. Checking units
3. Formatting the `.csv` as expected by PEBSI

## Quantile Mapping (Bias Correction)

Code in `PEBSI/shop/preprocessing/` can generate quantile CDFs from your AWS dataset for bias-correcting MERRA-2 data. This is recommended when both data sources are available for the same site.
