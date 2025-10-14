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

# Proficiency estimation

## Download
Data has to be downloaded from official EgoExo4D repository: https://ego-exo4d-data.org/#intro 
You need the annotations, VRS files (for IMU data), the ET videos and the POV videos (for the downstream proficiency estimation task).
Alternatively, you can also run PulseFormer without the motion-informed temporal attention (MITA) module. Then, you do not need the VRS files (IMU data).
To get the needed data, run the following commands with the Ego4D downloader:
1) egoexo -o PATH_SAVE_FOLDER --parts take_vrs_noimagestream metadata annotations downscaled_takes/448


3) egoexo -o PATH_SAVE_FOLDER --views ego --parts takes
3) egoexo -o PATH_SAVE_FOLDER --views ego --parts downscaled_takes/448

