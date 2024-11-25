import matplotlib.pyplot as plt
import numpy as np
import yaml

participants = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014',
                '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025']
fs_all = {'et': 30, 'rgb': 15, 'aria_imu_left': 800, 'aria_imu_right': 1000, 'ms_imu': 64, 'ms_ecg': 1024,
          'shimmer': 256, 'md': 128}

# Fixed parameters
with open('./configs/preprocessing/config_preprocessing_egoppg.yml', 'r') as yamlfile:
    configs = yaml.load(yamlfile, Loader=yaml.FullLoader)
task_times_all = configs['task_times']

total_task_times = []
task_times_per_task = {task: [] for task in task_times_all['001'].keys()}

for participant in task_times_all.keys():
    task_times_full = [task_times_all[participant]['video'][0],
                       task_times_all[participant]['walking_3'][1]]
    total_task_times.append((task_times_full[1] - task_times_full[0]) / fs_all['et'])
    for task in task_times_all[participant].keys():
        task_times_per_task[task].append(
            (task_times_all[participant][task][1] - task_times_all[participant][task][0]) / fs_all['et'])

print(f'Mean total task time: {np.mean(total_task_times) / 60} minutes')
print(f'Mean task time per task:')
for task, times in task_times_per_task.items():
    print(f'{task}: {np.mean(times) / 60} minutes')
