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

from source.preprocessing.preprocessing_helper import chunk_data, save_chunks
from source.preprocessing.preprocessing_extended_helper import diff_standardize_extended_video, diff_standardize_video, diff_video
from source.preprocessing.preprocessing_extended_helper import diff_standardize_label, diff_label, standardize_label, standardize_video
from source.preprocessing.preprocessing_multi import get_save_name_participant
from scipy.signal import butter
from source.sp.sp_helper import calculate_peak_hr, create_perfect_sine, calculate_peaks, calculate_ibis
from source.utils import get_adjusted_task_times, resample_signal, upsample_video, get_participants_list

from functools import partial
from multiprocessing import Pool
from natsort import natsorted


def get_tasks_dataset(dataset_name, participant, configs):
    if dataset_name in ['ubfc_rppg', 'egoppg']:
        tasks = [0]
    elif dataset_name == 'pure':
        if participant == '06':
            tasks = ['01', '03', '04', '05', '06']
        else:
            tasks = ['01', '02', '03', '04', '05', '06']
    elif dataset_name == 'mmpd':
        mat_files = natsorted(os.listdir(configs['original_data_path'] + f'/{participant}'))
        tasks = [mat_file.split(".")[0][3:] for mat_file in mat_files]
    else:
        raise RuntimeError('Dataset has not been implemented yet!')
    return tasks

def preprocess_videos(dataset_name, configs, configs_pre, participant, data_path, save_path):
    tasks = get_tasks_dataset(dataset_name, participant, configs)
    chunk_idx = 0
    for i_task, task in enumerate(tasks):
        save_name_participant = get_save_name_participant(dataset_name, participant, i_task)
        file_counter = len(glob.glob1(data_path, f"{save_name_participant}_input_{configs['input_name']}*"))
        if file_counter > 0:
            # Load frames
            frames_task = list()
            for i in range(file_counter):
                frames_task.extend(np.load(data_path + f'/{save_name_participant}_input_{configs["input_name"]}{i}.npy'))
            frames_task = np.asarray(frames_task)
            if len(frames_task.shape) == 3:
                frames_task = np.expand_dims(frames_task, axis=3)

            # Get task times adjusted to video start and chunk length
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
            save_chunks(frame_chunks, save_path + f'/{participant}_input_{configs["input_name"]}', chunk_idx)
            chunk_idx += len(frame_chunks)
        else:
            raise ValueError(f'No video files found for participant {participant}!')


def preprocess_timeseries(dataset_name, configs, configs_pre, participant, data_path, save_path):
    tasks = get_tasks_dataset(dataset_name, participant, configs)
    signal = 'ppg'
    chunk_idx = 0
    for i_task, task in enumerate(tasks):
        save_name_participant = get_save_name_participant(dataset_name, participant, i_task)
        data = np.load(data_path + f'/{save_name_participant}_label_{signal}.npy')

        # Filter PPG signals to filter out any motion artifacts or low-frequent component (PPG ear sensor has)
        if signal in ['ppg']:
            data = nk.ppg_clean(data, sampling_rate=configs['fs'], method='elgendi')

        # Downsample signal and interpolate between samples if upsampling is used
        data = data[0::configs['downsampling']]
        if configs['upsampling'] > 1:
            data = resample_signal(data, data.shape[0] * configs['upsampling'] - 2, 'linear')

        # Preprocess biosignals
        data_orig = data.copy()
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=1)  # Extend 1 dim if only 1 dim
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

        if data.shape[1] == 1:
            data = np.squeeze(data, axis=1)

        # Chunk, check that number of files are the same as for video, and save
        label_chunks = chunk_data(data, configs['clip_length_new'])
        if signal in ['ppg']:
            save_chunks(label_chunks, save_path + f'/{participant}_label_{signal}', chunk_idx)
        else:
            raise ValueError("Unsupported signal!")
        chunk_idx += len(label_chunks)

    # Check number of video clips and biosignal clips is the same
    if (len(glob.glob1(save_path, f"{participant}_input_{configs['input_name']}*")) !=
            len(glob.glob1(save_path, f"{participant}_label_{signal}*"))):
        raise ValueError(f'Number of video and biosignal files do not match for participant {participant}!')


def mp_preprocessing_extended(participant, configs, configs_pre, data_path, save_path, do_video, do_timesignal,
                              dataset_name):
    if do_video:
        preprocess_videos(dataset_name, configs, configs_pre, participant, data_path, save_path)
    if do_timesignal:
        preprocess_timeseries(dataset_name, configs, configs_pre, participant, data_path, save_path)
    print(f'Finished participant {participant}\n')


def main():
    # Variable parameters
    dataset_name = 'ubfc_rppg'  # pure, ubfc_rppg, mmpd
    do_video = True
    do_timesignal = True
    use_mp = True

    # Load configuration files
    cfg_path = f'./configs/preprocessing/config_preprocessing_extended_{dataset_name}.yml'
    with open(cfg_path, 'r') as yamlfile:
        configs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfg_path = f'./configs/preprocessing/config_preprocessing_{dataset_name}.yml'
    with open(cfg_path, 'r') as yamlfile:
        configs_pre = yaml.load(yamlfile, Loader=yaml.FullLoader)

    # Get participants depending on dataset
    participants = get_participants_list(dataset_name)

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
                         save_path=save_path, do_video=do_video, do_timesignal=do_timesignal, dataset_name=dataset_name)
        p.map(prod_x, participants)
    else:
        for participant in participants:
            mp_preprocessing_extended(participant, configs, configs_pre, data_path, save_path, do_video, do_timesignal,
                                      dataset_name)


if __name__ == "__main__":
    main()

