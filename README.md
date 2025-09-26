# egoPPG

Structure based on rPPG-toolbox.

## Install
- Install environment via: conda env create -f environment.yml
- mamba-ssm: how to

## Preprocessing

To get training data from egoPPG for EgoExo4D, preprocess the data with:
downsampling: 3 and upsampling:3
Will be saved into a folder named Down1_*_Up3, 1 as downsampling//upsampling so that in the folder name the "Down" shows the effective downsampling factor.

## ML
validation set size is set to 10% of training set (see ml_helper.py)

Run main_ml.py from within "ml" folder due to relative imports.

ml config file: specify
CACHED_PATH for TRAIN and TEST
DATA_PATH: Explain
FILE_PATH: explain
