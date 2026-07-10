# Climate Data

PEBSI accepts two types of climate forcing: reanalysis data (MERRA-2) and on-glacier weather station data (AWS). You need at least one, and you likely MERRA-2 even if you have AWS data to fill in variables that were not measured.

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

- Concatenating files
- Renaming columns to be self-consistent
- Identifying periods viable for simulation (i.e., gaps small enough for interpolation)

**`process_AWS`** — for the final preparation steps:

- Interpolating NaNs
- Checking units
- Formatting the `.csv` as expected by PEBSI

## Quantile Mapping (Bias Correction)

Code in `PEBSI/shop/preprocessing/` can generate quantile CDFs from your AWS dataset for bias-correcting MERRA-2 data. This is the recommended way to use AWS data when limited data is available, as bias-corrected MERRA-2 forcings performs similarly to AWS forcings but enable longer periods to be simulated.
