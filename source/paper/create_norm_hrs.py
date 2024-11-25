import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from source.preprocessing.preprocessing_helper import get_egoexo4d_takes

# Get takes
with open('./configs/preprocessing/config_preprocessing_egoexo4d.yml', 'r') as yamlfile:
    configs = yaml.load(yamlfile, Loader=yaml.FullLoader)
takes = get_egoexo4d_takes(configs['exclusion_list'])

# Get all HRs
hrs = []
for take in takes:
    take_name = take['video_paths']['ego'].split('/')[1]
    hrs_temp = np.load(f'/data/bjbraun/Projects/egoPPG/predicted_hrs/egoexo4d/{take_name}_hrs.npy')
    hrs.append(hrs_temp)

print('AH')

