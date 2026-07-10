# File Structure

The recommended directory layout is:

```
home/
├── PEBSI/
│   ├── data/
│   ├── pebsi/
│   ├── shading/
│   └── ...
├── Output/
├── climate_data/
│   ├── AWS/
│   └── MERRA2/
└── RGI/
```

For testing, only the `PEBSI/` folder is needed. The model creates the `Output/` folder automatically when storing results.

For real simulations you will also need:

**Randolph Glacier Inventory (RGI)**
: Used to retrieve the latitude, longitude, and mean elevation for a given glacier. If you only plan to run certain regions, you only need the relevant subregion shapefiles plus the `00_rgi60_attribs` folder.

**Climate data**
: Divided into two categories — weather station data (`AWS/`) and reanalysis data (`MERRA2/`). See the [Climate Data](tutorial/climate-data.md) tutorial for preparation details.
