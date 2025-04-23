# Blood Pressure Change Classification from PPG Signals

![Example: Blood pressure measurement and change label detection. black-: real blood pressure measurement; red --: intitial blood pressure value; black -.-: setting extreme change threshold of blood pressure; green area: predicted change within threshold; red area: predicted change over threshold; yellow area: predicted change below threshold.](figure/animation.gif)
<p align="center">
<b>Figure 1:</b> Blood pressure measurement and change detection. <br>
Legend: 
⚫ <b>black solid line</b> - actual blood pressure measurements; 
🔴 <b>red dashed line</b> - initial blood pressure value; 
⚫ <b>black dash-dot line</b> - extreme change thresholds; 
🟢 <b>green area</b> - predicted changes within threshold; 
🔴 <b>red area</b> - predicted changes above threshold; 
🟡 <b>yellow area</b> - predicted changes below threshold.
</p>

This repository provides an end-to-end framework for preparing datasets and training machine learning models to classify blood pressure (BP) change labels from photoplethysmogram (PPG) signals. The data is derived from [VitalDB](https://vitaldb.net/) and [PulsDB](https://doi.org/10.3389/fdgth.2022.1090854), public vital sign databases.

The current release includes a small test dataset containing 20 subjects for demonstration purposes. Full datasets and complete codebase will be released progressively.

##  Project Structure
```
.
├── data/                  # Raw and intermediate HDF5 files (20 subjects)
├── code/
│   ├── matlab/            # MATLAB scripts for data preprocessing
│   └── python/            # Python model training and evaluation
│       ├── train/         # Training codes
│       └── test/          # Testing codes
├── output/                # Auto-generated results
├── models/                # Saved best performance models
├── README.md              # Project overview and usage
├── LICENSE                # Open-source license file
└── .gitignore             # File exclusion rules
```
##  Overview

This project includes:

- MATLAB code to process PPG and BP waveforms into labeled training/testing samples.
- Balanced sampling based on systolic/diastolic/mean BP change magnitudes.
- Python modules for model training and evaluation.

ΔBP(i,j) = BP(i+j) - BP(i), i = [1, 2, 3,…, N-1], j = [1, 2, 3,…, N-i],
where,
- BP(i) is the initial blood pressure reading at time point i,
- BP(i+j) is a subsequent blood pressure reading j time points after i,
- ΔBP(i,j) represents the difference between these two readings.

The index i ranges from 1 to N-1, where N is the total number of BP readings for a patient.
This allows us to use any reading except the last one as our initial point. For each initial point i, j ranges from 1 to N-i enabling us to calculate BP changes for all possible subsequent readings. The upper limit N-i prevents accessing readings beyond the total number N.

Here is a concrete example: If a patient has 5 BP readings (N=5):
- when i=1: j can be 1,2,3,4 (comparing BP reading 1 with BP readings 2,3,4,5)
- when i=2: j can be 1,2,3 (comparing BP reading 2 with BP readings 3,4,5)
- when i=3: j can be 1,2 (comparing BP reading 3 with BP readings 4,5)
- when i=4: j can be 1 (comparing BP reading 4 with BP reading 5)

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
<!-- - `trn_ppgfea.h5` -->

Then run the preprocessing scripts:

```matlab
cd code/matlab
run('preprocess_bp_data.m')  % Generates training/test HDF5 files