import neurokit2 as nk
import numpy as np
import scipy.io
import yaml

from source.evaluation.evaluation_sp import evaluation_single_part
from source.utils import quantile_artifact_removal
from sp_helper import calculate_fft_hr, calculate_peak_hr, calculate_ibis
from sp_helper import get_closest_task_category
from signal_filtering import filter_rppg, filter_ppg

from functools import partial
from multiprocessing import Pool
from scipy.signal import butter
from scipy.stats import pearsonr


def mp_main(participant, configs, win_size_hr, stride_hr, low_pass, high_pass, methods, tasks_evaluate):
    # Get general parameters
    data_path = f'{configs["preprocessed_data_path"]}/Data_SP/{participant}'
    task_times_part = configs['task_times'][participant]
    fs_all = configs['fs_all']

    # Get tasks to evaluate for participant
    tasks_evaluate_part = []
    for task in configs['tasks']:
        if task in tasks_evaluate and participant not in configs['exclusion_list_l'][task]:
            tasks_evaluate_part.append(task)
    tasks_evaluate_part.append('all')

    # Get methods to evaluate for participant
    methods_part = methods.copy()
    if participant in configs['exclusion_list_shimmer']:
        methods_part.remove('ppg_s_peaks')
        methods_part.remove('ppg_s_fft')

    # Get ECG data, channel values, and Aria IMU data
    aria_acc = np.load(f'{data_path}/aria_imu_right.npy', allow_pickle=True)
    aria_acc_magn = np.sqrt(np.sum(np.square(np.vstack((aria_acc[:, 0], aria_acc[:, 1], aria_acc[:, 2]))), axis=0))
    channel_values = np.load(f'{data_path}/et_channel_values_eyes.npy')
    ecg = np.load(f'{data_path}/ms_ecg.npy')
    if participant not in configs['exclusion_list_shimmer']:
        shimmer_ppg = np.load(f'{data_path}/shimmer.npy')
    md_ppg = np.load(f'{data_path}/md.npy')

    # Invert MD and channel values
    md_ppg = np.max(md_ppg) - md_ppg
    channel_values = np.max(channel_values) - channel_values

    # Downsample and then upsample again to imitate 10 Hz
    """channel_values = channel_values[::3]
    channel_values = resample_signal(channel_values, channel_values.shape[0] * 3, 'cubic')"""

    # Calculate HR over 30 second windows for rPPG and ECG
    task_categories = []
    hrs_tasks = {task: {method: list() for method in methods+['ecg']} for task in tasks_evaluate}
    prev_hr_rppg_fft, prev_hr_s_ppg_fft, prev_hr_md_ppg_fft = None, None, None
    n_wins = int((len(channel_values) - win_size_hr * fs_all['et']) / (stride_hr * fs_all['et'])) + 1
    for i_win in range(n_wins):
        # Get corresponding task for window and continue if task is not in tasks_evaluate_part
        curr_task = get_closest_task_category(i_win * stride_hr * fs_all['et'], win_size_hr * fs_all['et'],
                                              task_times_part)
        if curr_task not in tasks_evaluate_part:
            continue

        # Save task category for scatter plot
        task_categories.append(curr_task)
        # if i_win < 55:
        #     continue

        # Get start time of each signal
        start_et = i_win * stride_hr * fs_all['et']
        start_aria_imu = i_win * stride_hr * fs_all['aria_imu_right']
        start_ppg_shimmer = i_win * stride_hr * fs_all['shimmer']
        start_ppg_md = i_win * stride_hr * fs_all['md']
        start_ecg = i_win * stride_hr * fs_all['ms_ecg']

        # Get windows of signals
        aria_acc_mag_temp = aria_acc_magn[start_aria_imu:start_aria_imu + win_size_hr * fs_all['aria_imu_right']]
        channel_values_temp = channel_values[start_et:start_et + win_size_hr * fs_all['et']]
        if participant not in configs['exclusion_list_shimmer']:
            shimmer_ppg_temp = shimmer_ppg[start_ppg_shimmer:start_ppg_shimmer + win_size_hr * fs_all['shimmer']]
        md_ppg_temp = md_ppg[start_ppg_md:start_ppg_md + win_size_hr * fs_all['md']]
        ecg_temp = ecg[start_ecg:start_ecg + win_size_hr * fs_all['ms_ecg']]

        # Filter signals
        rppg_signal_temp = filter_rppg(channel_values_temp, fs_all['et'], low_pass, high_pass, 'butter', ntaps=128)
        [b, a] = butter(4, [low_pass / (fs_all['aria_imu_right'] / 2),
                            high_pass / (fs_all['aria_imu_right'] / 2)], btype='bandpass')
        aria_acc_mag_temp = scipy.signal.filtfilt(b, a, aria_acc_mag_temp)
        if participant not in configs['exclusion_list_shimmer']:
            shimmer_ppg_temp = filter_ppg(shimmer_ppg_temp, fs_all['shimmer'], 'Shimmer')
        md_ppg_temp = filter_ppg(md_ppg_temp, fs_all['md'], 'MD')

        """import matplotlib.pyplot as plt
        from source.utils import resample_signal, normalize
        fig, ax = plt.subplots(figsize=(10, 5))
        st = 0
        end = 12
        ax.plot(normalize(resample_signal(md_ppg_temp[st*fs_all['md']:end*fs_all['md']], (end-st)*fs_all['md'], 'cubic'), 'zero_one'), label='MD')
        ax.plot(normalize(resample_signal(rppg_signal_temp[st*fs_all['et']:end*fs_all['et']], (end-st)*fs_all['md'], 'cubic'), 'zero_one'), label='rPPG')
        ax.set_title('Filtered signals')
        ax.legend()
        fig.show()"""

        # Calculate heart rates from rPPG signal
        rppg_hr_peaks = calculate_peak_hr(rppg_signal_temp, fs_all['et'])
        rppg_hr_fft = calculate_fft_hr(rppg_signal_temp, fs_all['et'], acc_magn=aria_acc_mag_temp,
                                       fs_acc=fs_all['aria_imu_right'], prev_hr=None, peak_hr=rppg_hr_peaks,
                                       low_pass=low_pass, high_pass=high_pass)
        prev_hr_rppg_fft = rppg_hr_fft
        hrs_tasks[curr_task]['rppg_peaks'].append(rppg_hr_peaks)
        hrs_tasks[curr_task]['rppg_fft'].append(rppg_hr_fft)
        hrs_tasks['all']['rppg_peaks'].append(rppg_hr_peaks)
        hrs_tasks['all']['rppg_fft'].append(rppg_hr_fft)

        # Calculate heart rate from Manuel's device PPG signal
        if 'ppg_md_peaks' in methods and 'ppg_md_fft' in methods:
            ppg_hr_md_peaks = calculate_peak_hr(md_ppg_temp, fs_all['md'])
            ppg_hr_md_fft = calculate_fft_hr(md_ppg_temp, fs_all['md'], acc_magn=aria_acc_mag_temp,
                                             fs_acc=fs_all['aria_imu_right'], prev_hr=prev_hr_md_ppg_fft,
                                             peak_hr=None, low_pass=low_pass, high_pass=high_pass)
            hrs_tasks[curr_task]['ppg_md_peaks'].append(ppg_hr_md_peaks)
            hrs_tasks[curr_task]['ppg_md_fft'].append(ppg_hr_md_fft)
            hrs_tasks['all']['ppg_md_peaks'].append(ppg_hr_md_peaks)
            hrs_tasks['all']['ppg_md_fft'].append(ppg_hr_md_fft)
            prev_hr_md_ppg_fft = ppg_hr_md_fft

        # Calculate heart rate from Shimmer PPG signal
        if 'ppg_s_peaks' in methods_part and 'ppg_s_fft' in methods_part:
            ppg_hr_s_peaks = calculate_peak_hr(shimmer_ppg_temp, fs_all['shimmer'])
            ppg_hr_s_fft = calculate_fft_hr(shimmer_ppg_temp, fs_all['shimmer'], acc_magn=aria_acc_mag_temp,
                                            fs_acc=fs_all['aria_imu_right'], prev_hr=prev_hr_s_ppg_fft,
                                            peak_hr=None, low_pass=low_pass, high_pass=high_pass)
            hrs_tasks[curr_task]['ppg_s_peaks'].append(ppg_hr_s_peaks)
            hrs_tasks[curr_task]['ppg_s_fft'].append(ppg_hr_s_fft)
            hrs_tasks['all']['ppg_s_peaks'].append(ppg_hr_s_peaks)
            hrs_tasks['all']['ppg_s_fft'].append(ppg_hr_s_fft)
            prev_hr_s_ppg_fft = ppg_hr_s_fft

        # Calculate ECG HR
        signals_ecg, info_ecg = nk.ecg_process(ecg_temp, sampling_rate=fs_all['ms_ecg'], method='neurokit')
        ibis_ecg = calculate_ibis(info_ecg['ECG_R_Peaks'], fs_all['ms_ecg'], reject_outliers=True)
        hrs_tasks[curr_task]['ecg'].append(60 / (np.mean(ibis_ecg) / fs_all['ms_ecg']))
        hrs_tasks['all']['ecg'].append(60 / (np.mean(ibis_ecg) / fs_all['ms_ecg']))

        show_ecg = False
        if show_ecg:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(signals_ecg['ECG_Clean'])
            y_peaks = np.asarray([signals_ecg['ECG_Clean'][i] for i in info_ecg['ECG_R_Peaks']])
            ax.scatter(info_ecg['ECG_R_Peaks'], y_peaks, color='green')
            ax.set_title(f'ECG: Original signal')
            fig.show()
            print('AH')

        # Curve tracing approach
        """end = channel_values_task.shape[0] // (window_size_hr * fs_all['et']) * window_size_hr * fs_all['et']
        inp = filter_rppg(channel_values_task[:end], fs_all['et'], low_pass, high_pass, plot_signals=plot_signals)
        inp = np.asarray(filtered_signals)
        hrs_rppg['ct'] = calculate_curve_tracing_hr(inp, fs_all['et'], low_pass, high_pass, window_size_hr,
                                            plot_signals=plot_signals)
        # hrs['rppg_ct'] = list(resample_signal(preds_troika, len(hrs['rppg_fft']), 'cubic'))"""

        # Troika approach
        """win_troika = 15
        troika = Troika(win_troika * fs_all['et'], win_duration=win_troika, step_duration=0.5,
                        ppg_sampling_freq=fs_all['et'], acc_sampling_freq=fs_all['aria_imu_right'],
                        cutoff_freqs=[low_pass, high_pass])
        hr_troika = troika.transform(np.copy(channel_values_task), aria_acc_task)
        preds_troika = [pred for pred in hr_troika]"""

    # Calculate MAE and Pearson correlation between Movisens and ECG and plot results
    mae = {task: {method: list() for method in methods} for task in tasks_evaluate}
    rmse = {task: {method: list() for method in methods} for task in tasks_evaluate}
    mape = {task: {method: list() for method in methods} for task in tasks_evaluate}
    pearson = {task: {method: list() for method in methods} for task in tasks_evaluate}
    for task in tasks_evaluate_part:
        hrs_tasks[task]['pa'] = [np.mean(hrs_tasks[task]['ecg'])] * len(hrs_tasks[task]['ecg'])
        for method in methods_part:
            for i in range(len(hrs_tasks[task]['ecg'])):
                mae[task][method].append(np.abs(hrs_tasks[task][method][i] - hrs_tasks[task]['ecg'][i]))
            if method == 'pa':
                continue
            rmse[task][method].append(np.sqrt(np.mean(np.square(np.asarray(hrs_tasks[task][method]) -
                                                                np.asarray(hrs_tasks[task]['ecg'])))))
            mape[task][method].append(np.mean(np.abs(np.asarray(hrs_tasks[task][method]) -
                                                     np.asarray(hrs_tasks[task]['ecg'])) /
                                              np.asarray(hrs_tasks[task]['ecg'])) * 100)
            pearson[task][method].append(pearsonr(hrs_tasks[task][method], hrs_tasks[task]['ecg'])[0])
    evaluation_single_part(participant, mae['all'], hrs_tasks['all'], 'all', task_categories, methods_part,
                           configs['method_names'], tasks_evaluate_part)
    return mae, rmse, mape, pearson


# ToDos: use of prev_hr and acc_magn for hr calculation, troika and curve tracing approach
def main():
    # Variable parameters
    participants = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014',
                    '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025']
    # participants = ['001']
    methods = ['rppg_peaks', 'rppg_fft', 'ppg_s_peaks', 'ppg_s_fft', 'ppg_md_peaks', 'ppg_md_fft', 'pa']
    tasks_evaluate = ['video', 'office', 'kitchen', 'dancing', 'bike', 'walking', 'all']
    # tasks_evaluate = ['video', 'running', 'all']
    win_size_hr = 60
    stride_hr = 60
    low_pass = 0.6
    high_pass = 3.00
    use_mp = True

    # Fixed parameters
    with open('./configs/preprocessing/config_preprocessing_egoppg.yml', 'r') as yamlfile:
        configs = yaml.load(yamlfile, Loader=yaml.FullLoader)

    # Run main function over all participants
    if use_mp:
        print('Using multiprocessing for data processing!')
        p = Pool(processes=2*len(participants))
        prod_x = partial(mp_main, configs=configs, win_size_hr=win_size_hr, stride_hr=stride_hr, low_pass=low_pass,
                         high_pass=high_pass,  methods=methods, tasks_evaluate=tasks_evaluate)
        results_participants = p.map(prod_x, participants)
    else:
        results_participants = []
        for participant in participants:
            results_participants.append(mp_main(participant, configs, win_size_hr, stride_hr, low_pass, high_pass,
                                                methods, tasks_evaluate))

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
                  f'{round(np.nanstd(pearson_all_part), rounding_digits)}')
            if method[-3:] == 'fft':
                print('')
        print('\n===================================================================================================\n')


if __name__ == "__main__":
    main()
