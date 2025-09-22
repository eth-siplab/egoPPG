import json
import glob
import math
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import os
import pandas as pd
import scipy.io
import yaml

from preprocessing_helper import chunk_data, save_chunks
from preprocessing_extended_helper import diff_standardize_extended_video, diff_standardize_video, diff_video
from preprocessing_extended_helper import diff_standardize_label, diff_label, standardize_label, standardize_video
from scipy.signal import butter
from source.sp.sp_helper import calculate_peak_hr, create_perfect_sine, calculate_peaks, calculate_ibis
from source.utils import get_adjusted_task_times, resample_signal, upsample_video

from functools import partial
from multiprocessing import Pool


def preprocess_videos(configs, configs_pre, participant, data_path, save_path):
    file_counter = len(glob.glob1(data_path, f"{participant}_input_et*"))
    if file_counter > 0:
        # Load frames
        frames = list()
        for i in range(file_counter):
            frames.extend(np.load(data_path + f'/{participant}_input_et{i}.npy'))
        frames = np.asarray(frames)
        if len(frames.shape) == 3:
            frames = np.expand_dims(frames, axis=3)

        # Get task times adjusted to video start and chunk length
        task_times = get_adjusted_task_times(configs_pre['task_times'][participant], configs['clip_length_old'])
        chunk_idx = 0
        for task in task_times.keys():
            frames_task = frames[task_times[task][0]:task_times[task][1]]
            frames_task = frames_task[0::configs['downsampling']]  # (task_times[1]-task_times[0]) // downsampling + 1

            # Interpolate between frames if upsampling is used
            if configs['upsampling'] > 1:
                frames_task = upsample_video(frames_task, configs['upsampling'], 'linear')

            # Normalize/standardize frames
            frames_processed = list()
            for video_type in configs['video_types']:
                f_c = frames_task.copy()
                if video_type == "Raw":
                    frames_processed.append(np.asarray(f_c[:, :, :, :], dtype=np.float32))
                elif video_type == "Diff":
                    frames_processed.append(diff_video(np.asarray(f_c, dtype=np.float32)))
                elif video_type == "DiffStandardized":
                    frames_processed.append(diff_standardize_video(np.asarray(f_c, dtype=np.float32)))
                elif video_type == "DiffStandardizedExtended":
                    frames_processed.append(diff_standardize_extended_video(np.asarray(f_c, dtype=np.float32)))
                elif video_type == "Standardized":
                    frames_processed.append(standardize_video(np.asarray(f_c, dtype=np.float32)))
                else:
                    os.rmdir(save_path)
                    raise ValueError("Unsupported data type!")
            frames_processed = np.concatenate(frames_processed, axis=3)

            # Show first preprocessed image for validation
            if task == 'video':
                fig, ax = plt.subplots()
                ax.imshow(frames_processed[0, :, :, :1])
                ax.set_title(f'Participant {participant}')
                fig.show()
                if frames_processed.shape[3] > 1:
                    fig, ax = plt.subplots()
                    ax.imshow(frames_processed[0, :, :, 1:])
                    fig.show()

            # Chunk and save frames
            frame_chunks = chunk_data(frames_processed, configs['clip_length_new'])
            frame_chunks = np.transpose(frame_chunks, (0, 4, 1, 2, 3))   # save as (N, C, D, W, H) for PyTorch
            save_chunks(frame_chunks, save_path + f'/{participant}_input_et', chunk_idx)
            chunk_idx += len(frame_chunks)
    else:
        raise ValueError(f'No video files found for participant {participant}!')


def preprocess_timeseries(configs, configs_pre, participant, data_path, save_path):
    task_times = get_adjusted_task_times(configs_pre['task_times'][participant], configs['clip_length_old'])
    # for signal in ['ppg_nose', 'ppg_ear', 'imu_right', 'eda_raw', 'eda_filtered', 'eda_tonic']:
    # for signal in ['ppg_nose', '2imu_right']:
    for signal in ['2imu_right', 'ppg_nose']:
        if (participant in configs_pre['exclusion_list_shimmer'] and
                signal in ['ppg_ear', 'eda', 'eda_raw', 'eda_filtered', 'eda_tonic']):
            continue

        if signal in ['eda_raw', 'eda_filtered', 'eda_tonic']:
            data_all = np.load(data_path + f'/{participant}_label_eda.npy')
        elif signal == '2imu_right':
            data_all = np.load(data_path + f'/{participant}_label_imu_right.npy')
        else:
            data_all = np.load(data_path + f'/{participant}_label_{signal}.npy')

        if signal == 'imu_right':
            # data_all = data_all[:, :3]  # remove time column
            data_all = np.sqrt(np.sum(np.square(np.vstack((data_all[:, 0], data_all[:, 1], data_all[:, 2]))), axis=0))
        elif signal in ['eda_tonic', 'eda_filtered']:
            signals, info = nk.eda_process(data_all, sampling_rate=configs_pre['fs_all']['et'])
            data_all = np.asarray(signals['EDA_Tonic'])
            if signal == 'eda_filtered':
                [b_eda, a_eda] = butter(2, 0.003 / configs['fs_all']['et'] * 2, btype='highpass')
                data_all = scipy.signal.filtfilt(b_eda, a_eda, data_all)
            fig, ax = plt.subplots()
            ax.plot(data_all, label='raw_eda')
            ax.set_title(f'Participant {participant}')
            ax.legend()
            plt.show()

        chunk_idx = 0
        for task in task_times.keys():
            data = data_all[task_times[task][0]:task_times[task][1]]

            # ToDo: Invert and then invert back?
            # Filter PPG signals to filter out any motion artifacts or low-frequent component (PPG ear sensor has)
            if signal in ['ppg_nose', 'ppg_ear']:
                data = nk.ppg_clean(data, sampling_rate=configs['fs_all']['et'], method='elgendi')

            # Downsample signal
            data = data[0::configs['downsampling']]

            # Interpolate between samples if upsampling is used
            if configs['upsampling'] > 1:
                data = resample_signal(data, data.shape[0] * configs['upsampling'] - 2, 'linear')

            # Preprocess biosignals
            data_orig = data.copy()
            # Extend 1 dim if only 1 dim
            if len(data.shape) == 1:
                data = np.expand_dims(data, axis=1)
            for i in range(data.shape[1]):
                if configs["label_type"] == "Raw":
                    data[:, i] = data[:, i]
                elif configs["label_type"] == "Diff":
                    data[:, i] = diff_label(data[:, i])
                elif configs["label_type"] == "DiffStandardized":
                    data[:, i] = diff_standardize_label(data[:, i])
                elif configs["label_type"] == "Standardized":
                    data[:, i] = standardize_label(data[:, i])
                else:
                    os.rmdir(save_path)
                    raise ValueError("Unsupported label type for EDA!")

            if signal == '2imu_right':
                data = np.sqrt(np.sum(np.square(np.vstack((data[:, 0], data[:, 1], data[:, 2]))), axis=0))
            else:
                if data.shape[1] == 1:
                    data = np.squeeze(data, axis=1)

            # Chunk, check that number of files are the same as for video, and save
            label_chunks = chunk_data(data, configs['clip_length_new'])
            if signal in ['ppg_nose', 'ppg_ear', 'eda_raw', 'eda_filtered', 'eda_tonic', 'rr']:
                save_chunks(label_chunks, save_path + f'/{participant}_label_{signal}', chunk_idx)
            elif signal in ['imu_right', '2imu_right']:
                save_chunks(label_chunks, save_path + f'/{participant}_input_{signal}', chunk_idx)
            else:
                raise ValueError("Unsupported signal!")

            if signal == 'ppg_nose':
                data_orig = np.max(data_orig) - data_orig
                sp_chunk_len = 60 * configs['fs_all']['et']  # Number of samples in 60 seconds
                peaks_all = []
                if len(data_orig) <= sp_chunk_len:
                    peaks_all.extend(calculate_peaks(data_orig, configs['fs_all']['et']))
                else:
                    for i_chunk in range(0, len(data_orig)+1 - sp_chunk_len, sp_chunk_len):
                        if i_chunk > len(data_orig) - 2 * sp_chunk_len:
                            if (len(data_orig) % sp_chunk_len) == 0:
                                end = i_chunk + sp_chunk_len
                            else:
                                end = len(data_orig)
                        else:
                            end = i_chunk + sp_chunk_len
                        window_data = data_orig[i_chunk:end]
                        peaks_temp = calculate_peaks(window_data, configs['fs_all']['et'])
                        peaks_all.extend([peak + i_chunk for peak in peaks_temp])

                peaks_all = np.asarray(peaks_all)
                hrs_chunks = list()
                mapped_hrs_chunks = list()
                n_wins = len(data_orig) // configs['clip_length_new']
                for i_chunk in range(n_wins):
                    start = i_chunk * configs['clip_length_new']
                    end = start + configs['clip_length_new']
                    peaks_in_clip = peaks_all[(peaks_all >= start) & (peaks_all < end)]

                    # ibis = calculate_ibis(peaks_in_clip, configs['fs_all']['et'], 1.5, reject_outliers=False)
                    # hr_chunk = 60 / (np.mean(ibis) / configs['fs_all']['et'])
                    hr_chunk = 60 / (np.mean(np.diff(peaks_in_clip)) / configs['fs_all']['et'])
                    mapped_hr = int(round(hr_chunk))
                    if mapped_hr < 40:
                        mapped_hr = 40
                    elif mapped_hr > 180:
                        mapped_hr = 180
                    # mapped_hr_class = mapped_hr - 40  # now in the range 0 to 140
                    mapped_hr_class = mapped_hr
                    mapped_hrs_chunks.append([mapped_hr_class] * configs['clip_length_new'])
                save_chunks(np.asarray(hrs_chunks), save_path + f'/{participant}_label_hr', chunk_idx)
                save_chunks(np.asarray(mapped_hrs_chunks), save_path + f'/{participant}_label_classhr', chunk_idx)

                """data_orig = np.max(data_orig) - data_orig  # invert signal for nk peak finder
                sine_wave = create_perfect_sine(data_orig, configs['fs_all']['et'])
                data_orig_chunks = chunk_data(data_orig, configs['clip_length_new'])
                sine_wave_chunks = chunk_data(sine_wave, configs['clip_length_new'])
                hrs_chunks = list()
                mapped_hrs_chunks = list()
                for chunk in data_orig_chunks:
                    hr_chunk = calculate_peak_hr(chunk, configs['fs_all']['et'], reject_outliers=False)
                    hrs_chunks.append([hr_chunk] * len(chunk))
                    # Map HR to integer class: clip HR to [40, 180], then subtract 40 so that 40->0 and 180->140.
                    mapped_hr = int(round(hr_chunk))
                    if mapped_hr < 40:
                        mapped_hr = 40
                    elif mapped_hr > 180:
                        mapped_hr = 180
                    mapped_hr_class = mapped_hr  # now in the range 0 to 140
                    mapped_hrs_chunks.append([mapped_hr_class] * len(chunk))
                save_chunks(np.asarray(hrs_chunks), save_path + f'/{participant}_label_hr', chunk_idx)
                save_chunks(np.asarray(mapped_hrs_chunks), save_path + f'/{participant}_label_classhr', chunk_idx)
                save_chunks(sine_wave_chunks, save_path + f'/{participant}_label_sineppg', chunk_idx)"""
            elif signal == 'rr':
                sp_chunk_len = 60 * configs['fs_all']['et']  # Number of samples in 60 seconds
                resp_rates_full = np.empty(len(data_orig))
                if len(data_orig) <= sp_chunk_len:
                    signals_rr, info_rr = nk.rsp_process(data_orig, sampling_rate=configs['fs_all']['et'], show=False)
                    resp_rates_full[:] = np.mean(signals_rr['RSP_Rate'])
                else:
                    prev_rr = None
                    for i_chunk in range(0, len(data_orig) - sp_chunk_len, sp_chunk_len):
                        if i_chunk > len(data_orig) - 2 * sp_chunk_len:
                            if (len(data_orig) % sp_chunk_len) == 0:
                                end = i_chunk + sp_chunk_len
                            else:
                                end = len(data_orig)
                        else:
                            end = i_chunk + sp_chunk_len
                        window_data = data_orig[i_chunk:end]
                        signals_rr, info_rr = nk.rsp_process(window_data, sampling_rate=configs['fs_all']['et'],
                                                             show=False)
                        if math.isnan(np.mean(signals_rr['RSP_Rate'])):
                            rr_temp = prev_rr
                        else:
                            rr_temp = np.mean(signals_rr['RSP_Rate'])
                        resp_rates_full[i_chunk:end] = rr_temp
                        prev_rr = rr_temp

                rrs_chunks = []
                mapped_rrs_chunks = []
                min_rr = 5
                max_rr = 35
                for i_chunk in range(0, len(data_orig) - configs['clip_length_new'] + 1, configs['clip_length_new']):
                    # Calculate the average respiratory rate for the current clip.
                    rr_clip = np.mean(resp_rates_full[i_chunk:i_chunk + configs['clip_length_new']])
                    rrs_chunks.append([rr_clip] * configs['clip_length_new'])

                    # Map RR to an integer class by clipping between min_rr and max_rr and then shifting.
                    mapped_rr = int(round(rr_clip))
                    if mapped_rr < min_rr:
                        mapped_rr = min_rr
                    elif mapped_rr > max_rr:
                        mapped_rr = max_rr
                    mapped_rr_class = mapped_rr
                    mapped_rrs_chunks.append([mapped_rr_class] * configs['clip_length_new'])

                save_chunks(np.asarray(rrs_chunks), save_path + f'/{participant}_label_raterr', chunk_idx)
                save_chunks(np.asarray(mapped_rrs_chunks), save_path + f'/{participant}_label_classrr', chunk_idx)
            chunk_idx += len(label_chunks)

        if signal in ['ppg_nose', 'ppg_ear', 'eda_raw', 'eda_filtered', 'eda_tonic', 'rr']:
            if (len(glob.glob1(save_path, f"{participant}_input_et*")) !=
                    len(glob.glob1(save_path, f"{participant}_label_{signal}*"))):
                raise ValueError(f'Number of video and biosignal files do not match for participant {participant}!')
        elif signal in ['imu_right', '2imu_right']:
            if (len(glob.glob1(save_path, f"{participant}_input_et*")) !=
                    len(glob.glob1(save_path, f"{participant}_input_{signal}*"))):
                raise ValueError(f'Number of video and biosignal files do not match for participant {participant}!')
        else:
            raise ValueError("Unsupported signal!")


def mp_preprocessing_extended(participant, configs, configs_pre, data_path, save_path, do_video, do_timesignal):
    if do_video:
        preprocess_videos(configs, configs_pre, participant, data_path, save_path)
    if do_timesignal:
        preprocess_timeseries(configs, configs_pre, participant, data_path, save_path)
    print(f'Finished participant {participant}\n')


def main():
    # Variable parameters
    participants = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                    '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025']
    # participants = ['005']
    do_video = True
    do_timesignal = True
    use_mp = True

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

    # Create folder to save preprocessed data
    configuration_new = (f'CL{configs["clip_length_new"]}_Down{configs["downsampling"]//configs["upsampling"]}_'
                         f'W{configs["w"]}_H{configs["h"]}_Label{configs["label_type"]}_VideoType')
    for video_type in configs["video_types"]:
        configuration_new += video_type
    if configs['upsampling'] > 1:
        configuration_new += f'_Up{configs["upsampling"]}'
    save_path = configs['dir_preprocessed'] + f'/Data_ML/{configuration_old}/{configuration_new}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f'Saved to path: {save_path}')

    if use_mp:
        print('Using multiprocessing for data processing!')
        p = Pool(processes=len(participants))
        prod_x = partial(mp_preprocessing_extended, configs=configs, configs_pre=configs_pre, data_path=data_path,
                         save_path=save_path, do_video=do_video, do_timesignal=do_timesignal)
        p.map(prod_x, participants)
    else:
        for participant in participants:
            mp_preprocessing_extended(participant, configs, configs_pre, data_path, save_path, do_video, do_timesignal)


if __name__ == "__main__":
    main()

