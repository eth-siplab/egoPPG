import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import yaml

from source.evaluation.evaluation_sp import evaluation_single_part
from sp_helper import calculate_fft_hr, calculate_peak_rr
from sp_helper import get_closest_task_category
from signal_filtering import filter_imu, get_edr, filter_rppg

from functools import partial
from multiprocessing import Pool
from scipy.signal import butter
from scipy.stats import pearsonr


def mp_main(participant, configs, win_size_rr, stride_rr, low_pass, high_pass, tasks_evaluate, methods):
    # Get general parameters
    data_path = f'{configs["preprocessed_data_path"]}/Data_SP/{participant}'
    task_times_part = configs['task_times'][participant]
    for task in task_times_part:
        task_times_part[task] = [task_times_part[task][0] / configs['fs_all']['et'] * configs['fs_all']['ms_ecg'],
                                 task_times_part[task][1] / configs['fs_all']['et'] * configs['fs_all']['ms_ecg']]
    fs_all = configs['fs_all']

    # Get tasks to evaluate for participant
    tasks_evaluate_part = []
    for task in configs['tasks']:
        if task in tasks_evaluate:
            tasks_evaluate_part.append(task)
    tasks_evaluate_part.append('all')

    # Get methods to evaluate for participant
    methods_part = methods.copy()

    # Get ECG data, channel values, and Aria IMU data
    ecg = np.load(f'{data_path}/ms_ecg.npy')
    imu = np.load(f'{data_path}/ms_imu.npy')[:, 2]  # ToDo: For egorr [:, 2]
    rr = np.load(f'{data_path}/biopac.npy')  # ToDo: For egorr
    channel_values = np.load(f'{data_path}/et_channel_values_eyes.npy')
    channel_values = np.max(channel_values) - channel_values  # invert

    # Calculate RR over x second windows for rPPG and ECG
    task_categories = []
    rr_tasks = {task: {method: list() for method in methods+['rr_peaks']} for task in tasks_evaluate}
    n_wins = int((len(ecg) - win_size_rr * fs_all['ms_ecg']) / (stride_rr * fs_all['ms_ecg'])) + 1
    for i_win in range(n_wins):
        # Get corresponding task for window and continue if task is not in tasks_evaluate_part
        curr_task = get_closest_task_category(i_win * stride_rr * fs_all['ms_ecg'], win_size_rr * fs_all['ms_ecg'],
                                              task_times_part)
        if curr_task not in tasks_evaluate_part:
            continue

        # Save task category for scatter plot
        task_categories.append(curr_task)

        # Get start time of each signal
        # start_ecg = i_win * stride_rr * fs_all['ms_ecg']
        start_imu = i_win * stride_rr * fs_all['ms_imu']
        start_rr = i_win * stride_rr * fs_all['biopac']  # ToDo: For egorr
        start_et = i_win * stride_rr * fs_all['et']

        # Get windows of signals
        # ecg_temp = ecg[start_ecg:start_ecg + win_size_rr * fs_all['ms_ecg']]
        imu_temp = imu[start_imu:start_imu + win_size_rr * fs_all['ms_imu']]
        rr_temp = rr[start_rr:start_rr + win_size_rr * fs_all['biopac']]
        channel_values_temp = channel_values[start_et:start_et + win_size_rr * fs_all['et']]

        # Calculate belt RR
        # ToDo: For egorr
        rr_peaks = calculate_peak_rr(rr_temp, fs_all['biopac'], 'RR belt', None)
        rr_tasks[curr_task]['rr_peaks'].append(rr_peaks)
        rr_tasks['all']['rr_peaks'].append(rr_peaks)

        # Calculate ECG HR
        """edr = get_edr(ecg_temp, fs_all['ms_ecg'])
        edr_peaks = calculate_peak_rr(edr, fs_all['ms_ecg'], 'EDR', None, reject_outliers=False)
        rr_tasks[curr_task]['edr_peaks'].append(edr_peaks)
        rr_tasks['all']['edr_peaks'].append(edr_peaks)"""

        # Calculate IMU RR
        imu_temp = filter_imu(imu_temp, fs_all['ms_imu'], 'IMU')  # ToDo: use this line for egorr
        imu_peaks = calculate_peak_rr(imu_temp, fs_all['ms_imu'], 'IMU', None)
        rr_tasks[curr_task]['imu_peaks'].append(imu_peaks)
        rr_tasks['all']['imu_peaks'].append(imu_peaks)

        # Calculate rRR
        rrr_signal_temp = filter_rppg(channel_values_temp, fs_all['et'], 0.15, 0.6, 'butter', ntaps=128)
        rrr_peaks = calculate_peak_rr(rrr_signal_temp, fs_all['et'], 'rRR', None)
        rr_tasks[curr_task]['rrr_peaks'].append(rrr_peaks)
        rr_tasks['all']['rrr_peaks'].append(rrr_peaks)


    # Calculate MAE and Pearson correlation between Movisens and ECG and plot results
    mae = {task: {method: list() for method in methods} for task in tasks_evaluate}
    rmse = {task: {method: list() for method in methods} for task in tasks_evaluate}
    mape = {task: {method: list() for method in methods} for task in tasks_evaluate}
    pearson = {task: {method: list() for method in methods} for task in tasks_evaluate}
    reference = 'rr_peaks'  # ToDo: Relace 'imu_peaks' with 'rr_peaks' for belt RR
    for task in tasks_evaluate_part:
        for method in methods_part:
            for i in range(len(rr_tasks[task][reference])):
                mae[task][method].append(np.abs(rr_tasks[task][method][i] - rr_tasks[task][reference][i]))
            rmse[task][method].append(np.sqrt(np.nanmean(np.square(np.asarray(rr_tasks[task][method]) -
                                                                np.asarray(rr_tasks[task][reference])))))
            mape[task][method].append(np.nanmean(np.abs(np.asarray(rr_tasks[task][method]) -
                                                     np.asarray(rr_tasks[task][reference])) /
                                              np.asarray(rr_tasks[task][reference])) * 100)
            pearson[task][method].append(pearsonr(rr_tasks[task][method], rr_tasks[task][reference])[0])
    # evaluation_single_part(participant, mae['all'], rr_tasks['all'], 'all', task_categories, methods_part,
     #                       configs['method_names'], tasks_evaluate_part)
    return mae, rmse, mape, pearson


# ToDos: use of prev_hr and acc_magn for hr calculation, troika and curve tracing approach
def main():
    # Variable parameters
    participants = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014',
                    '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025']
    participants = ['001', '002', '003', '004']
    # participants = ['001']
    tasks_evaluate = ['video', 'office', 'kitchen', 'dancing', 'bike', 'walking', 'all']
    tasks_evaluate = ['video', 'office', 'dancing', 'bike', 'walking', 'all']
    # tasks_evaluate = ['dancing', 'bike', 'walking', 'all']
    methods = ['imu_peaks', 'rrr_peaks']  # edr_peaks', 'rrr_peaks', 'imu_peaks'
    win_size_rr = 60
    stride_rr = 60
    low_pass = 0.05
    high_pass = 3.0
    use_mp = False

    # Fixed parameters
    # ToDo: replace with egorr for rr dataset
    with open('./configs/preprocessing/config_preprocessing_egorr.yml', 'r') as yamlfile:
        configs = yaml.load(yamlfile, Loader=yaml.FullLoader)

    # Run main function over all participants
    if use_mp:
        print('Using multiprocessing for data processing!')
        p = Pool(processes=len(participants))
        prod_x = partial(mp_main, configs=configs, win_size_rr=win_size_rr, stride_rr=stride_rr, low_pass=low_pass,
                         high_pass=high_pass, tasks_evaluate=tasks_evaluate, methods=methods)
        results_participants = p.map(prod_x, participants)
    else:
        results_participants = []
        for participant in participants:
            results_participants.append(mp_main(participant, configs, win_size_rr, stride_rr, low_pass, high_pass,
                                                tasks_evaluate, methods))

    # Print results over all participants for each task and method
    rounding_digits = 2
    print('=========================================================================================================\n')
    for task in tasks_evaluate:
        for method in methods:
            mae_all_part = []
            rmse_all_part = []
            mape_all_part = []
            pearson_all_part = []
            mae_per_part = {participant: None for participant in participants}
            rmse_per_part = {participant: None for participant in participants}
            mape_per_part = {participant: None for participant in participants}
            pearson_per_part = {participant: None for participant in participants}
            for results_participant in results_participants:  # loop over all participants
                mae_all_part.extend(results_participant[0][task][method])
                rmse_all_part.extend(results_participant[1][task][method])
                mape_all_part.extend(results_participant[2][task][method])
                pearson_all_part.extend(results_participant[3][task][method])
                mae_per_part[participants[results_participants.index(results_participant)]] = (
                    round(np.mean(results_participant[0][task][method]), rounding_digits))
                rmse_per_part[participants[results_participants.index(results_participant)]] = (
                    round(np.mean(results_participant[1][task][method]), rounding_digits))
                mape_per_part[participants[results_participants.index(results_participant)]] = (
                    round(np.mean(results_participant[2][task][method]), rounding_digits))
                pearson_per_part[participants[results_participants.index(results_participant)]] = (
                    round(np.mean(results_participant[3][task][method]), rounding_digits))
            print(f'All participants, {task}: All MAEs with {configs["method_names"][method]}: '
                  f'{mae_per_part}')
            print(f'All participants, {task}: Mean MAE+-STD with {configs["method_names"][method]}: '
                  f'{round(np.nanmean(mae_all_part), rounding_digits)} +- '
                  f'{round(np.nanstd(mae_all_part), rounding_digits)}')
            print(f'All participants, {task}: All RMSEs with {configs["method_names"][method]}: '
                  f'{rmse_per_part}')
            print(f'All participants, {task}: Mean RMSE+-STD with {configs["method_names"][method]}: '
                  f'{round(np.nanmean(rmse_all_part), rounding_digits)} +- '
                  f'{round(np.nanstd(rmse_all_part), rounding_digits)}')
            print(f'All participants, {task}: All MAPEs with {configs["method_names"][method]}: '
                  f'{mape_per_part}')
            print(f'All participants, {task}: Mean MAPE+-STD with {configs["method_names"][method]}: '
                  f'{round(np.nanmean(mape_all_part), rounding_digits)} +- '
                  f'{round(np.nanstd(mape_all_part), rounding_digits)}')
            print(f'All participants, {task}: All Pearsons with {configs["method_names"][method]}: '
                  f'{pearson_per_part}')
            print(f'All participants, {task}: Mean Pearson correlation+-STD with {configs["method_names"][method]}: '
                  f'{round(np.nanmean(pearson_all_part), rounding_digits)} +- '
                  f'{round(np.nanstd(pearson_all_part), rounding_digits)}\n')
            if method[-3:] == 'fft':
                print('')
        print('\n===================================================================================================\n')


if __name__ == "__main__":
    main()
