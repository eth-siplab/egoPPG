import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import yaml

from preprocessing.preprocessing_helper import get_egoexo4d_takes

from scipy.interpolate import CubicSpline


def get_participants_list(label_signals, dataset_name):
    if dataset_name == 'egoppg':
        participants = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                        '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025']
    elif dataset_name == 'egoexo4d':
        with open('./configs/preprocessing/config_preprocessing_egoexo4d.yml', 'r') as yamlfile:
            configs_pre = yaml.load(yamlfile, Loader=yaml.FullLoader)
        participants = get_egoexo4d_takes(configs_pre['exclusion_list'])
        participants = [take['video_paths']['ego'].split('/')[1] for take in participants]
    else:
        print('Dataset has not been implemented yet!')
        raise RuntimeError
    return participants


def get_participants_lists(configs):
    if configs.TRAIN.DATA.DATASET == configs.TEST.DATA.DATASET:
        participants_train_valid = get_participants_list(configs.LABEL_SIGNALS, configs.TRAIN.DATA.DATASET)
        participants_test = None
    else:
        participants_train_valid = get_participants_list(configs.LABEL_SIGNALS, configs.TRAIN.DATA.DATASET)
        participants_test = get_participants_list(configs.LABEL_SIGNALS, configs.TEST.DATA.DATASET)

    return participants_train_valid, participants_test


def quantile_artifact_removal(data, q1, q3, factor):
    sig_change = np.diff(data, axis=0)
    quantile_q1 = np.quantile(sig_change, q1)
    quantile_q3 = np.quantile(sig_change, q3)
    mean_change = (quantile_q1 + quantile_q3) / 2
    std_change = factor * 1.34896 * (quantile_q3 - quantile_q1)
    big_diffs = np.argwhere((sig_change > (mean_change + std_change)) |
                            (sig_change < (mean_change - std_change)))
    for i in range(len(big_diffs)):
        data[big_diffs[i][0] + 1:] = (data[big_diffs[i][0] + 1:] +
                                      (data[big_diffs[i][0]] - data[big_diffs[i][0] + 1]))

    return data


def quantile_artifact_removal_multi(data, q1, q3, factor):
    for dim in range(data.shape[0]):
        sig_change = np.diff(data[dim, :], axis=0)
        quantile_q1 = np.quantile(sig_change, q1)
        quantile_q3 = np.quantile(sig_change, q3)
        mean_change = (quantile_q1 + quantile_q3) / 2
        std_change = factor * 1.34896 * (quantile_q3 - quantile_q1)
        big_diffs = np.argwhere((sig_change > (mean_change + std_change)) |
                                (sig_change < (mean_change - std_change)))
        for i in range(len(big_diffs)):
            data[dim, big_diffs[i][0] + 1:] = (data[dim, big_diffs[i][0] + 1:] +
                                          (data[dim, big_diffs[i][0]] - data[dim, big_diffs[i][0] + 1]))

    return data


def get_adjusted_task_times(task_times):
    task_times = {task: [int(task_times[task][0] - task_times['video'][0]),
                         int(task_times[task][1] - task_times['video'][0])]
                  for task in task_times.keys()}
    return task_times


def get_task_chunk_list(config, participant):
    cfg_path = f'./configs/preprocessing/config_preprocessing_{config.TRAIN.DATA.DATASET}.yml'
    with open(cfg_path, 'r') as yamlfile:
        configs_pre = yaml.load(yamlfile, Loader=yaml.FullLoader)
    task_times_adjusted = get_adjusted_task_times(configs_pre['task_times'][participant])

    task_chunk_list = {'task_names': [], 'keep': []}
    for task in task_times_adjusted.keys():
        n_chunks = (((task_times_adjusted[task][1] - task_times_adjusted[task][0]) //
                     config.TRAIN.DATA.PREPROCESS.DOWNSAMPLE + 1) // config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH)
        if task in ['walking_1', 'walking_2', 'walking_3']:
            task_name = 'walking'
        else:
            task_name = task
        task_chunk_list['task_names'].extend([task_name] * n_chunks)

        if (task_name in config.TASKS_TO_USE) and (participant not in configs_pre['exclusion_list'][task_name]):
            task_chunk_list['keep'].extend([1] * n_chunks)
        else:
            task_chunk_list['keep'].extend([0] * n_chunks)
    return task_chunk_list


def get_ml_config_name(configs_g):
    ml_config_name = (f'CL{configs_g["clip_length"]}_W{configs_g["w"]}_H{configs_g["h"]}_'
                      f'Label{configs_g["label_type"]}_VideoType')
    for video_type in configs_g["video_types"]:
        ml_config_name += video_type
    return ml_config_name


def resample(data, new_len, interp_type):
    if interp_type == 'linear':  # same as sample
        interp_data = np.interp(np.linspace(0, data.shape[0] - 1, new_len), np.arange(data.shape[0]), data)
    elif interp_type == 'cubic':
        spl = CubicSpline(np.arange(data.shape[0]), data)
        interp_data = spl(np.linspace(0, data.shape[0] - 1, new_len))
    else:
        raise ValueError(f'Interpolation type {interp_type} not supported.')
    return interp_data


def resample_signal(signal, new_len, interp_type):
    if not isinstance(signal, np.ndarray):
        signal = np.asarray(signal)
    if new_len == signal.shape[0]:
        return signal
    else:
        if len(signal.shape) == 1:
            interp_signal = resample(signal, new_len, interp_type)
        else:
            interp_signal = np.zeros((new_len, signal.shape[1]))
            for i in range(signal.shape[1]):
                interp_signal[:, i] = resample(signal[:, i], new_len, interp_type)
        return interp_signal


def upsample_video(frames, upsample_factor, interp_type):
    frames_upsampled = np.zeros((frames.shape[0] * upsample_factor - 2, frames.shape[1], frames.shape[2],
                                 frames.shape[3]))
    for row in range(frames.shape[1]):
        frames_upsampled[:, row, :, 0] = resample_signal(frames[:, row, :, 0], frames.shape[0] * upsample_factor - 2,
                                                         interp_type)
    return frames_upsampled


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def normalize(data, type_normalization):
    # No normalization
    if type_normalization is None:
        data = data
    # Around zero, zero mean and std = 1
    elif type_normalization == 'std':
        data = (data - data.mean()) / data.std()
        data[np.isnan(data)] = 0
    # Between 0 and 1
    elif type_normalization == 'zero_one':
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

    return data


def calculate_peaks(ppg_signal, fs):
    ppg_signal = ppg_signal - min(ppg_signal)
    ppg_signal = ppg_signal / max(ppg_signal)

    # NK requires PPG signal to be inverted (have standard PPG form)
    signals_sys, info_sys = nk.ppg_peaks(ppg_signal, sampling_rate=fs, method="elgendi", show=False)
    peaks = info_sys["PPG_Peaks"]

    plot_signals = False
    if plot_signals:
        fig, ax = plt.subplots()
        ax.plot(ppg_signal)
        y_peaks = np.asarray([ppg_signal[i] for i in peaks])
        ax.scatter(peaks, y_peaks, color='green')
        fig.show()

    return peaks


def calculate_peak_hr(ppg_signal, fs, reject_outliers=True, prev_hr=None):
    ppg_signal = ppg_signal - min(ppg_signal)
    ppg_signal = ppg_signal / max(ppg_signal)

    # NK requires PPG signal to be inverted (have standard PPG form)
    signals_sys, info_sys = nk.ppg_peaks(ppg_signal, sampling_rate=fs, method="elgendi", show=False)
    peaks = info_sys["PPG_Peaks"]
    # peaks = get_simple_footpoints(ppg_signal, peaks, 'Min', fs)

    plot_signals = False
    if plot_signals:
        fig, ax = plt.subplots()
        ax.plot(ppg_signal)
        y_peaks = np.asarray([ppg_signal[i] for i in peaks])
        ax.scatter(peaks, y_peaks, color='green')
        fig.show()

    ibis = calculate_ibis(peaks, fs, 1.5, reject_outliers=reject_outliers)
    peaks_hr = 60 / (np.mean(ibis) / fs)

    return peaks_hr


def calculate_ibis(peaks, fs, threshold, reject_outliers=False):
    ibis = np.diff(peaks)

    # Only keep IBIs that are within the interquartile range
    if reject_outliers:
        ibis = ibis[np.where(ibis < (threshold * fs))]
        if len(ibis) == 0:
            return ibis
        quantile_q1 = np.quantile(ibis, 0.25)
        quantile_q3 = np.quantile(ibis, 0.75)
        ibis = ibis[np.where((ibis >= quantile_q1) & (ibis <= quantile_q3))]

    return ibis
