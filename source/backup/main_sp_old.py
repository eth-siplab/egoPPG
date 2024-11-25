import neurokit2 as nk
import numpy as np
import scipy.io
import yaml

from source.evaluation.evaluation_sp import evaluation_single_part
from source.sp.sp_helper import calculate_fft_hr, calculate_peak_hr, calculate_ibis
from source.sp.sp_helper import get_closest_task_category
from source.sp.signal_filtering import filter_rppg, filter_ppg

from functools import partial
from multiprocessing import Pool
from scipy.signal import butter
from scipy.stats import pearsonr


def mp_main(participant, configs_g, tasks, win_size_hr, stride_hr, low_pass, high_pass, downsampling_factor,
            methods, method_names):
    # Get general parameters
    data_path = f'{configs_g["preprocessed_data_path"]}/Data_SP/{participant}'
    task_times_part = configs_g['task_times'][participant]
    fs_all = configs_g['fs_all']
    for key in fs_all.keys():
        fs_all[key] = fs_all[key] // downsampling_factor

    # Get start and end times, ECG data, channel values, and Aria IMU data
    aria_acc = np.load(f'{data_path}/aria_imu_right.npy', allow_pickle=True)[::downsampling_factor]
    aria_acc_mag = np.sqrt(np.sum(np.square(np.vstack((aria_acc[:, 0], aria_acc[:, 1], aria_acc[:, 2]))), axis=0))
    # channel_values = np.asarray(preprocess_egoppg(participant, 'camera-et', configs_g))[::downsampling_factor]
    channel_values = np.load(f'{data_path}/et_channel_values.npy')[::downsampling_factor]
    ecg = np.load(f'{data_path}/ms_ecg.npy')[::downsampling_factor]
    # shimmer_ppg = np.load(f'{data_path}/shimmer.npy')[::downsampling_factor]
    md_ppg = np.load(f'{data_path}/md.npy')[::downsampling_factor]

    # Downsample and then upsample again to imitate 10 Hz
    """if do_downsampling:
        channel_values = channel_values[::3]
        channel_values = resample_signal(channel_values, channel_values.shape[0] * 3, 'cubic')"""

    # Cut data into tasks and calculate HR
    hrs_tasks = {}
    categories = []
    for task in tasks:
        if task == 'full':
            aria_acc_magn_task = aria_acc_mag
            channel_values_task = channel_values
            ecg_task = ecg
            # shimmer_ppg_task = shimmer_ppg
            md_ppg_task = md_ppg
        else:
            cut_start = (task_times_part[task][0] - task_times_part['video'][0]) / fs_all['et']
            cut_end = (task_times_part[task][1] - task_times_part['video'][0]) / fs_all['et']
            aria_acc_magn_task = aria_acc_mag[int(cut_start * fs_all['aria_imu_right']):
                                              int(cut_end * fs_all['aria_imu_right'])]
            channel_values_task = channel_values[int(cut_start * fs_all['et']):int(cut_end * fs_all['et'])]
            ecg_task = ecg[int(cut_start * fs_all['ms_ecg']):int(cut_end * fs_all['ms_ecg'])]
            # shimmer_ppg_task = shimmer_ppg[int(cut_start * fs_all['shimmer']):int(cut_end * fs_all['shimmer'])]
            md_ppg_task = md_ppg[int(cut_start * fs_all['md']):int(cut_end * fs_all['md'])]

        # Calculate HR over 30 second windows for rPPG and ECG
        hrs = {method: list() for method in methods}
        hrs['ecg'] = []
        prev_hr_rppg_fft, prev_hr_s_ppg_fft, prev_hr_md_ppg_fft = None, None, None
        n_wins = int((len(channel_values_task) - win_size_hr * fs_all['et']) / (stride_hr * fs_all['et'])) + 1
        for i_win in range(n_wins):
            # Get start time of each signal
            start_et = i_win * stride_hr * fs_all['et']
            start_aria_imu = i_win * stride_hr * fs_all['aria_imu_right']
            # start_ppg_shimmer = i_win * stride_hr * fs_all['shimmer']
            start_ppg_md = i_win * stride_hr * fs_all['md']
            start_ecg = i_win * stride_hr * fs_all['ms_ecg']

            # Get windows of signals
            aria_acc_mag_temp = aria_acc_magn_task[start_aria_imu:
                                                   start_aria_imu + win_size_hr * fs_all['aria_imu_right']]
            channel_values_temp = channel_values_task[start_et:start_et + win_size_hr * fs_all['et']]
            # shimmer_ppg_temp = shimmer_ppg_task[start_ppg_shimmer:start_ppg_shimmer + win_size_hr * fs_all['shimmer']]
            md_ppg_temp = md_ppg_task[start_ppg_md:start_ppg_md + win_size_hr * fs_all['md']]
            ecg_temp = ecg_task[start_ecg:start_ecg + win_size_hr * fs_all['ms_ecg']]

            # Filter signals
            rppg_signal_temp = filter_rppg(channel_values_temp, fs_all['et'], low_pass, high_pass, 'firwin', ntaps=128)
            [b, a] = butter(4, [low_pass / (fs_all['aria_imu_right'] / 2),
                                high_pass / (fs_all['aria_imu_right'] / 2)], btype='bandpass')
            aria_acc_mag_temp = scipy.signal.filtfilt(b, a, aria_acc_mag_temp)
            # shimmer_ppg_temp = filter_ppg(shimmer_ppg_temp, fs_all['shimmer'], 'Shimmer')
            md_ppg_temp = filter_ppg(md_ppg_temp, fs_all['md'], 'MD')

            # Remove first and last two seconds of each signal to remove filter artifacts
            """crop = 2
            rppg_signal_temp = rppg_signal_temp[crop * fs_all['et']:-crop * fs_all['et']]
            aria_acc_mag_temp = aria_acc_mag_temp[crop * fs_all['aria_imu_right']:-crop * fs_all['aria_imu_right']]
            shimmer_ppg_temp = shimmer_ppg_temp[crop * fs_all['shimmer']:-crop * fs_all['shimmer']]
            md_ppg_temp = md_ppg_temp[crop * fs_all['md']:-crop * fs_all['md']]
            ecg_temp = ecg_temp[crop * fs_all['ms_ecg']:-crop * fs_all['ms_ecg']]"""

            # Calculate heart rates from rPPG signal
            rppg_hr_peaks = calculate_peak_hr(rppg_signal_temp, fs_all['et'])
            rppg_hr_fft = calculate_fft_hr(rppg_signal_temp, fs_all['et'], acc_magn=aria_acc_mag_temp,
                                           fs_acc=fs_all['aria_imu_right'], prev_hr=None, peak_hr=rppg_hr_peaks,
                                           low_pass=low_pass, high_pass=high_pass)
            # rppg_hr_fft = 60
            prev_hr_rppg_fft = rppg_hr_fft
            hrs['rppg_peaks'].append(rppg_hr_peaks)
            hrs['rppg_fft'].append(rppg_hr_fft)

            # Calculate heart rate from Shimmer PPG signal
            """if 'ppg_s_peaks' in methods and 'ppg_s_fft' in methods:
                ppg_hr_s_peaks = calculate_peak_hr(shimmer_ppg_temp, fs_all['shimmer'])
                ppg_hr_s_fft = calculate_fft_hr(shimmer_ppg_temp, fs_all['shimmer'], acc_magn=aria_acc_mag_temp,
                                                fs_acc=fs_all['aria_imu_right'], prev_hr=prev_hr_s_ppg_fft,
                                                peak_hr=None, low_pass=low_pass, high_pass=high_pass)
                hrs['ppg_s_peaks'].append(ppg_hr_s_peaks)
                hrs['ppg_s_fft'].append(ppg_hr_s_fft)
                prev_hr_s_ppg_fft = ppg_hr_s_fft"""

            # Calculate heart rate from Manuel's device PPG signal
            if 'ppg_md_peaks' in methods and 'ppg_md_fft' in methods:
                ppg_hr_md_peaks = calculate_peak_hr(md_ppg_temp, fs_all['md'])
                ppg_hr_md_fft = calculate_fft_hr(md_ppg_temp, fs_all['md'], acc_magn=aria_acc_mag_temp,
                                                 fs_acc=fs_all['aria_imu_right'], prev_hr=prev_hr_md_ppg_fft,
                                                 peak_hr=None, low_pass=low_pass, high_pass=high_pass)
                hrs['ppg_md_peaks'].append(ppg_hr_md_peaks)
                hrs['ppg_md_fft'].append(ppg_hr_md_fft)
                prev_hr_md_ppg_fft = ppg_hr_md_fft

            # Calculate ECG HR
            signals_ecg, info_ecg = nk.ecg_process(ecg_temp, sampling_rate=fs_all['ms_ecg'], method='neurokit')
            ibis_ecg = calculate_ibis(info_ecg['ECG_R_Peaks'], fs_all['ms_ecg'], reject_outliers=True)
            hrs['ecg'].append(60 / (np.mean(ibis_ecg) / fs_all['ms_ecg']))
            # hrs['ecg'].append(np.mean(ecg_temp))  # for Movisens predicted HRs

            # Add window to corresponding task
            # categories.append(get_closest_task_category(start_et, task_times_part))

            # fig, ax = plt.subplots()
            # ax.plot(signals_ecg['ECG_Clean'])
            # fig.show()

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

        # Check that rPPG and ECG have same number of windows
        if (len(hrs['rppg_fft']) != len(hrs['rppg_peaks'])) or (len(hrs['rppg_fft']) != len(hrs['ecg'])):
            raise ValueError('Different number of windows')

        # Calculate MAE and Pearson correlation between Movisens and ECG and plot results
        # ToDo: Think about if this is the correct way with overlapping windows
        mae = {method: list() for method in methods}
        hrs['pa'] = [np.mean(hrs['ecg'])] * len(hrs['ecg'])
        for method in methods:
            for i in range(len(hrs['ecg'])):
                mae[method].append(np.abs(hrs[method][i] - hrs['ecg'][i]))
        evaluation_single_part(participant, mae, hrs, task, methods, method_names)
        hrs_tasks[task] = hrs

    print(f'Finished participant {participant}!')
    return hrs_tasks


# ToDos: Check window size, threshold for filtering, peak detection simple or nk, check for additional filters and
# if Butterworth or FIR better, filter frequencies, footpoints or not, use of peak hr for fft hr, use of prev_hr,
# filter entire signal and then window or filter only windows, diff of PPG signal for FFT,
# check with or without harmonic check in FFT, acc_magn in fft_hr
def main():
    # Variable parameters
    participants = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014',
                    '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026']
    participants = ['004']
    tasks = ['video', 'office', 'kitchen', 'dancing', 'bike', 'running']
    # tasks = ['video', 'office', 'kitchen', 'dancing', 'bike']
    tasks = ['dancing']  # video, office, kitchen, dancing, bike, running, full_wo_r, full
    # tasks = ['video', 'office']

    methods = ['rppg_peaks', 'rppg_fft', 'ppg_s_peaks', 'ppg_s_fft', 'ppg_md_peaks', 'ppg_md_fft', 'pa']
    methods = ['rppg_peaks', 'rppg_fft', 'ppg_md_peaks', 'ppg_md_fft', 'pa']
    win_size_hr = 30
    # stride_hr = win_size_hr // 2
    stride_hr = 30
    low_pass = 0.75  # 0.66 better for peaks, 0.75 better for fft
    high_pass = 3.00  # 3.33
    downsampling_factor = 1
    use_mp = False
    plot_signals = False
    rounding_digits = 3

    # Fixed parameters
    with open('./configs/preprocessing/config_preprocessing_egoppg.yml', 'r') as yamlfile:
        configs_g = yaml.load(yamlfile, Loader=yaml.FullLoader)
    method_names = {'rppg_peaks': 'rPPG Peaks', 'rppg_ct': 'rPPG Curve Tracing', 'rppg_fft': 'rPPG FFT',
                    'troika': 'rPPG Troika', 'ppg_s_peaks': 'PPG Shimmer Peaks', 'ppg_s_fft': 'PPG Shimmer FFT',
                    'ppg_md_peaks': 'PPG MD Peaks', 'ppg_md_fft': 'PPG MD FFT', 'pa': 'Personal Average'}

    # Run main function over all participants
    if use_mp:
        print('Using multiprocessing for data processing!')
        p = Pool(processes=len(participants))
        prod_x = partial(mp_main, configs_g=configs_g, tasks=tasks, win_size_hr=win_size_hr, stride_hr=stride_hr,
                         low_pass=low_pass, high_pass=high_pass, downsampling_factor=downsampling_factor,
                         methods=methods, method_names=method_names)
        results = p.map(prod_x, participants)
    else:
        results = []
        for participant in participants:
            results.append(mp_main(participant, configs_g, tasks, win_size_hr, stride_hr, low_pass, high_pass,
                                   downsampling_factor, methods, method_names))

    # Print results over all participants for each task and method
    print('=========================================================================================================\n')
    for task in tasks:
        for method in methods:
            mae_all_part = []
            pearson_all_part = []
            for result in results:  # loop over all participants
                mae_all_part.extend([abs(result[task][method][i] - result[task]['ecg'][i])
                                for i in range(len(result[task][method]))])
                pearson_temp, _ = pearsonr(result[task][method], result[task]['ecg'])  # Pearson irrelevant
                pearson_all_part.append(pearson_temp)
            print(f'All participants, {task}: Mean MAE+-STD with {method_names[method]}: '
                  f'{round(np.mean(mae_all_part), rounding_digits)} +- '
                  f'{round(np.std(mae_all_part), rounding_digits)}')
            print(f'All participants, {task}: Mean Pearson correlation+-STD with {method_names[method]}: '
                  f'{round(np.mean(pearson_all_part), rounding_digits)} +- '
                  f'{round(np.std(pearson_all_part), rounding_digits)}')
            if method[-3:] == 'fft':
                print('')
        print('\n===================================================================================================\n')

    # Print results over all participants and all tasks for each method
    for method in methods:
        mae_all_part_tasks = []
        all_preds = []
        all_labels = []
        for task in tasks:
            for result in results:
                mae_all_part_tasks.extend([abs(result[task][method][i] - result[task]['ecg'][i])
                                           for i in range(len(result[task][method]))])
                all_preds.extend(result[task][method])
                all_labels.extend(result[task]['ecg'])
        pearson_temp, _ = pearsonr(all_preds, all_labels)
        print(f'All participants, All tasks: Mean MAE+-STD with {method_names[method]}: '
              f'{round(np.mean(mae_all_part_tasks), rounding_digits)} +- '
              f'{round(np.std(mae_all_part_tasks), rounding_digits)}')
        print(f'All participants, All tasks: Mean Pearson correlation+-STD with {method_names[method]}: '
              f'{round(pearson_temp, rounding_digits)} +- '
              f'{round(pearson_temp, rounding_digits)}')
        if method[-3:] == 'fft':
            print('')


if __name__ == "__main__":
    main()
