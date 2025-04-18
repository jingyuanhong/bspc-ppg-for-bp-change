# Blood Pressure Change Estimation from PPG Signals

This repository provides an end-to-end framework for preparing datasets and training machine learning models to estimate blood pressure (BP) changes from photoplethysmogram (PPG) signals. The data is derived from [VitalDB](https://vitaldb.net/), a public vital sign database.

##  Project Structure
```
.
├── data/                  # Raw and intermediate HDF5 files
├── code/
│   ├── matlab/            # MATLAB scripts for data preprocessing
│   └── python/            # Python model training and evaluation
├── output/                # Auto-generated results
├── README.md              # Project overview and usage
├── LICENSE                # Open-source license file
└── .gitignore             # File exclusion rules
```
##  Overview

This project includes:

- MATLAB code to process PPG and BP waveforms into labeled training/testing samples.
- Balanced sampling based on systolic/diastolic/mean BP change magnitudes.
- Python modules for model training and evaluation (e.g., classification, regression).

##  Requirements

### MATLAB
- R2021a or later (with HDF5 support)

### Python
- Python 3.8+
- `requirements.txt` will be provided for dependencies (e.g., numpy, torch, scikit-learn)

##  Usage

### 1. Prepare Data (MATLAB)
Place the following input files in the `data/` folder:
- `vitaldb.h5`
- `trn_ppgfea.h5`

Then run the preprocessing scripts:

```matlab
cd code/matlab
run('preprocess_bp_data.m')  % Generates training/test HDF5 files