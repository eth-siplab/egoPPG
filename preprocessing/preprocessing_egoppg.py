import cv2
import glob
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import os
import pandas as pd
import yaml

from preprocessing.preprocessing_helper import chunk_data, save_chunks
from preprocessing.preprocessing_helper import diff_standardize_extended_video, diff_standardize_video, diff_video
from preprocessing.preprocessing_helper import diff_standardize_label, diff_label, standardize_label, standardize_video
from utils import get_adjusted_task_times, resample_signal, upsample_video, calculate_peaks

from functools import partial
from multiprocessing import Pool


def preprocess_videos(configs, participant, save_path):
    # Load frames
    frames = np.load(configs['original_data_path'] + f'/{participant}/{participant}_et.npy')
    frames = [cv2.resize(frames[i], (configs['w'], configs['h']), interpolation=cv2.INTER_AREA) for i in
              range(len(frames))]
    # N_chunks = len(frames) // configs['clip_length']
    frames = np.expand_dims(np.asarray(frames), axis=3)
    # frames = frames[:N_chunks * configs['clip_length'], :, :, :]  # remove first frames until first full chunk

    task_times = get_adjusted_task_times(configs['task_times'][participant])
    chunk_idx = 0
    for task in task_times.keys():
        frames_task = frames[task_times[task][0]:task_times[task][1]]
        frames_task = frames_task[0::configs['downsampling']]  # downsample frames if specified

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
        frame_chunks = chunk_data(frames_processed, configs['clip_length'])
        frame_chunks = np.transpose(frame_chunks, (0, 4, 1, 2, 3))   # save as (N, C, D, W, H) for PyTorch
        save_chunks(frame_chunks, save_path + f'/{participant}_input_et', chunk_idx)
        chunk_idx += len(frame_chunks)


def preprocess_timeseries(configs, participant, save_path):
    task_times = get_adjusted_task_times(configs['task_times'][participant])
    for signal in ['ppg_nose', 'imu_right']:
        data_all = np.load(configs['original_data_path'] + f'/{participant}/{participant}_{signal}.npy')
        chunk_idx = 0
        for task in task_times.keys():
            data = data_all[task_times[task][0]:task_times[task][1]]

            # Filter PPG signal to filter out any motion artifacts or low-frequent component
            if signal == 'ppg_nose':
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

            if signal == 'imu_right':
                data = np.sqrt(np.sum(np.square(np.vstack((data[:, 0], data[:, 1], data[:, 2]))), axis=0))
            else:
                if data.shape[1] == 1:
                    data = np.squeeze(data, axis=1)

            # Chunk and save
            label_chunks = chunk_data(data, configs['clip_length'])
            if signal == 'ppg_nose':
                save_chunks(label_chunks, save_path + f'/{participant}_label_{signal}', chunk_idx)
            elif signal == 'imu_right':
                save_chunks(label_chunks, save_path + f'/{participant}_input_{signal}', chunk_idx)
            else:
                raise ValueError("Unsupported signal!")

            # Calculate and save HR in bpm as class label (40-180 bpm) to directly predict HR class
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
                mapped_hrs_chunks = list()
                n_wins = len(data_orig) // configs['clip_length']
                for i_chunk in range(n_wins):
                    start = i_chunk * configs['clip_length']
                    end = start + configs['clip_length']
                    peaks_in_clip = peaks_all[(peaks_all >= start) & (peaks_all < end)]
                    hr_chunk = 60 / (np.mean(np.diff(peaks_in_clip)) / configs['fs_all']['et'])
                    mapped_hr = int(round(hr_chunk))
                    if mapped_hr < 40:
                        mapped_hr = 40
                    elif mapped_hr > 180:
                        mapped_hr = 180
                    mapped_hr_class = mapped_hr
                    mapped_hrs_chunks.append([mapped_hr_class] * configs['clip_length'])
                save_chunks(np.asarray(mapped_hrs_chunks), save_path + f'/{participant}_label_classhr', chunk_idx)
            chunk_idx += len(label_chunks)

        if signal in ['ppg_nose']:
            if (len(glob.glob1(save_path, f"{participant}_input_et*")) !=
                    len(glob.glob1(save_path, f"{participant}_label_{signal}*"))):
                raise ValueError(f'Number of video and biosignal files do not match for participant {participant}!')
        elif signal in ['imu_right']:
            if (len(glob.glob1(save_path, f"{participant}_input_et*")) !=
                    len(glob.glob1(save_path, f"{participant}_input_{signal}*"))):
                raise ValueError(f'Number of video and biosignal files do not match for participant {participant}!')
        else:
            raise ValueError("Unsupported signal!")


def mp_preprocessing_extended(participant, configs, save_path, do_video, do_timesignal):
    if do_video:
        preprocess_videos(configs, participant, save_path)
    if do_timesignal:
        preprocess_timeseries(configs, participant, save_path)
    print(f'Finished participant {participant}\n')


def main():
    # Variable parameters
    participants = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                    '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025']
    # participants = ['017']
    do_video = True
    do_timesignal = True
    use_mp = True  # use multiprocessing, consider memory consumption

    # Load configuration files
    cfg_path = f'./configs/preprocessing/config_preprocessing_egoppg.yml'
    with open(cfg_path, 'r') as yamlfile:
        configs = yaml.load(yamlfile, Loader=yaml.FullLoader)

    # Create folder to save preprocessed data
    configuration_overall = f'CL{configs["clip_length"]}_W{configs["w"]}_H{configs["h"]}_LabelRaw_VideoTypeRaw'
    configuration_detailed = (f'CL{configs["clip_length"]}_Down{configs["downsampling"]//configs["upsampling"]}_'
                         f'W{configs["w"]}_H{configs["h"]}_Label{configs["label_type"]}_VideoType')
    for video_type in configs["video_types"]:
        configuration_detailed += video_type
    if configs['upsampling'] > 1:
        configuration_detailed += f'_Up{configs["upsampling"]}'
    save_path = configs['preprocessed_data_path'] + f'/{configuration_overall}/{configuration_detailed}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f'Saved to path: {save_path}')

    # Run preprocessing
    if use_mp:
        print('Using multiprocessing for data processing!')
        p = Pool(processes=len(participants))
        prod_x = partial(mp_preprocessing_extended, configs=configs, save_path=save_path, do_video=do_video,
                         do_timesignal=do_timesignal)
        p.map(prod_x, participants)
    else:
        for participant in participants:
            mp_preprocessing_extended(participant, configs, save_path, do_video, do_timesignal)


if __name__ == "__main__":
    main()

