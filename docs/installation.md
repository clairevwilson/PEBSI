# Installation

## 1. Clone the repository

Navigate to the folder where you want to install the model and clone PEBSI:

```bash
git clone https://github.com/clairevwilson/PEBSI.git
```

## 2. Set up the environment

Navigate inside the PEBSI directory and create a new conda environment from the provided file:

```bash
cd PEBSI
conda env create -f environment.yaml
conda activate pebsi-env
```

!!! note
    PEBSI should work on Python >= 3.10. Please [open an issue](https://github.com/clairevwilson/PEBSI/issues) if you find this is not the case.

## 3. Test the installation

### A. Unit tests

These verify that your environment can import all dependencies and that the model's core physics are operating correctly on your machine:

```bash
pytest tests/
```

### B. Test simulation

The model comes with one year of mock forcing data. Running with `--testing` uses preset parameters and that sample data with no additional setup required:

```bash
python simulation.py --testing
```
