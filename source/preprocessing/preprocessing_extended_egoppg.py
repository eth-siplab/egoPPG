import json
import glob
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import os
import pandas as pd
import yaml

from preprocessing_helper import chunk_data, save_chunks
from preprocessing_extended_helper import diff_standardize_extended_video, diff_standardize_video, diff_video
from preprocessing_extended_helper import diff_standardize_label, diff_label, standardize_label, standardize_video
from source.sp.sp_helper import calculate_peak_hr
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
    # for signal in ['ppg_nose', 'ppg_ear', 'imu_right']:
    for signal in ['imu_right']:
        if participant in configs_pre['exclusion_list_shimmer'] and signal == 'ppg_ear':
            continue
        data_all = np.load(data_path + f'/{participant}_label_{signal}.npy')

        if signal == 'imu_right':
            # data_all = data_all[:, :3]  # remove time column
            data_all = np.sqrt(np.sum(np.square(np.vstack((data_all[:, 0], data_all[:, 1], data_all[:, 2]))), axis=0))

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
            if configs["label_type"] == "Raw":
                data = data
            elif configs["label_type"] == "Diff":
                data = diff_label(data)
            elif configs["label_type"] == "DiffStandardized":
                data = diff_standardize_label(data)
            elif configs["label_type"] == "Standardized":
                data = standardize_label(data)
            else:
                os.rmdir(save_path)
                raise ValueError("Unsupported label type for EDA!")

            # Chunk, check that number of files are the same as for video, and save
            label_chunks = chunk_data(data, configs['clip_length_new'])
            if signal in ['ppg_nose', 'ppg_ear']:
                save_chunks(label_chunks, save_path + f'/{participant}_label_{signal}', chunk_idx)
            elif signal == 'imu_right':
                save_chunks(label_chunks, save_path + f'/{participant}_input_{signal}', chunk_idx)
            else:
                raise ValueError("Unsupported signal!")
            if signal == 'ppg_nose':
                data_orig = np.max(data_orig) - data_orig  # invert signal for nk peak finder
                data_orig_chunks = chunk_data(data_orig, configs['clip_length_new'])
                hrs_chunks = list()
                for chunk in data_orig_chunks:
                    hr_chunk = calculate_peak_hr(chunk, configs['fs_all']['et'], reject_outliers=False)
                    hrs_chunks.append([hr_chunk] * len(chunk))
                save_chunks(np.asarray(hrs_chunks), save_path + f'/{participant}_label_hr', chunk_idx)
            chunk_idx += len(label_chunks)

        if signal in ['ppg_nose', 'ppg_ear']:
            if (len(glob.glob1(save_path, f"{participant}_input_et*")) !=
                    len(glob.glob1(save_path, f"{participant}_label_{signal}*"))):
                raise ValueError(f'Number of video and biosignal files do not match for participant {participant}!')
        elif signal == 'imu_right':
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
    # participants = ['002']
    do_video = False
    do_timesignal = True
    use_mp = False

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

