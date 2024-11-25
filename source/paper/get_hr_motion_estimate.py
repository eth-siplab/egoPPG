import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import yaml

from source.sp.sp_helper import calculate_peak_hr, calculate_ibis
from source.utils import get_adjusted_task_times, resample_signal, upsample_video

# Variable parameters
participants = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025']

# Load configuration files
cfg_path = f'./configs/preprocessing/config_preprocessing_extended_egoppg.yml'
with open(cfg_path, 'r') as yamlfile:
    configs = yaml.load(yamlfile, Loader=yaml.FullLoader)
cfg_path = f'./configs/preprocessing/config_preprocessing_egoppg.yml'
with open(cfg_path, 'r') as yamlfile:
    configs_pre = yaml.load(yamlfile, Loader=yaml.FullLoader)

# Get folder where to load data from
configuration_old = f'CL{configs["clip_length_old"]}_W{configs["w"]}_H{configs["h"]}_LabelRaw_VideoTypeRaw'
data_path = configs['dir_preprocessed'] + f'/Data_ML/{configuration_old}/Data'

signal = 'imu_right'
tasks_evaluate = ['video', 'office', 'kitchen', 'dancing', 'bike', 'walking']
magn_tasks = {task: [] for task in tasks_evaluate}
for participant in participants:
    task_times = get_adjusted_task_times(configs_pre['task_times'][participant], configs['clip_length_old'])
    data_all = np.load(data_path + f'/{participant}_label_{signal}.npy')

    # Get IMU magnitude
    data_all = np.sqrt(np.sum(np.square(np.vstack((data_all[:, 0], data_all[:, 1], data_all[:, 2]))), axis=0))
    for task in task_times.keys():
        if task == 'running':
            continue

        data = data_all[task_times[task][0]:task_times[task][1]]
        output = np.sum(abs(np.diff(data)))
        if task in ['walking_1', 'walking_2', 'walking_3']:
            magn_tasks['walking'].append(output)
        else:
            magn_tasks[task].append(output)

signal = 'ppg_nose'
hrs_tasks = {task: [] for task in tasks_evaluate}
hrs_min = []
hrs_max = []
hrs_stds_parts = []
for participant in participants:
    task_times = get_adjusted_task_times(configs_pre['task_times'][participant], configs['clip_length_old'])
    data_all = np.load(data_path + f'/{participant}_label_{signal}.npy')
    hrs_temp = []

    for task in task_times.keys():
        if task in ['walking_1', 'walking_2', 'walking_3']:
            task_check = 'walking'
        else:
            task_check = task
        if task == 'running' or participant in configs_pre['exclusion_list_l'][task_check]:
            continue

        data = data_all[task_times[task][0]:task_times[task][1]]
        data = nk.ppg_clean(data, sampling_rate=configs['fs_all']['et'], method='elgendi')
        data = np.asarray(data)
        data = data - min(data)
        data = data / max(data)

        # NK requires PPG signal to be inverted (have standard PPG form)
        signals_sys, info_sys = nk.ppg_peaks(data, sampling_rate=configs['fs_all']['et'], method="elgendi",
                                             show=False)
        peaks = info_sys["PPG_Peaks"]
        ibis = calculate_ibis(peaks, configs['fs_all']['et'], reject_outliers=True)
        hrs_temp.extend(60 / (ibis / configs['fs_all']['et']))
        peaks_hr = 60 / (np.mean(ibis) / configs['fs_all']['et'])

        if task in ['walking_1', 'walking_2', 'walking_3']:
            hrs_tasks['walking'].append(peaks_hr)
            hrs_min.append(min(60 / (ibis / configs['fs_all']['et'])))
            hrs_max.append(max(60 / (ibis / configs['fs_all']['et'])))
        else:
            hrs_tasks[task].append(peaks_hr)
            hrs_min.append(min(60 / (ibis / configs['fs_all']['et'])))
            hrs_max.append(max(60 / (ibis / configs['fs_all']['et'])))
    hrs_stds_parts.append(np.std(hrs_temp))


# Print all magnitudes per task
print('Magnitudes')
magn_tasks = {task: np.mean(magn_tasks[task]) for task in magn_tasks.keys()}
for task in magn_tasks.keys():
    print(f'{task}: {magn_tasks[task]}')

# Print all normalized magnitudes per task
print('\nNormalized magnitudes')
magn_tasks = {task: (magn_tasks[task] - min(magn_tasks.values())) / (max(magn_tasks.values()) - min(magn_tasks.values()))
              for task in magn_tasks.keys()}
for task in magn_tasks.keys():
    print(f'{task}: {magn_tasks[task]}')

# Print all heart rates per task
print('\nHeart rates')
hrs_tasks = {task: np.mean(hrs_tasks[task]) for task in hrs_tasks.keys()}
for task in hrs_tasks.keys():
    print(f'{task}: {hrs_tasks[task]}')

# Print mean std of the hearts of all participants
print('\nMean std of heart rates')
print(np.mean(hrs_stds_parts))

# Print min and max heart rates
print('\nMin and max heart rates')
print(f'Min: {min(hrs_min)}')
print(f'Max: {max(hrs_max)}')

