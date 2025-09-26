# egoPPG

Structure based on rPPG-toolbox.

## Install
- Install environment via: conda env create -f environment.yml

# Run
1) conda activate egoPPG
2) Preprocessing: python -m preprocessing.preprocessing_egoppg --cfg_path configs/preprocessing/config_preprocessing_egoppg.yaml
3) ML: python -m ml.main_ml --cfg_path configs/ml/CONFIG_FILE_NAME.yaml

Important: 
1) Run from the root folder (egoPPG) due to relative imports
2) Change the config file paths accordingly
   1) preprocessing config files: "original_data_path" and "preprocessed_data_path"
   2) ml config files: "CACHED_PATH" for TRAIN and TEST, "DATA_PATH", "FILE_PATH"


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
