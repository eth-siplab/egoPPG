import cv2
import glob
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import os
import pandas as pd
import yaml

from preprocessing_helper import chunk_data, save_chunks
from preprocessing_helper import diff_standardize_extended_video, diff_standardize_video, diff_video
from preprocessing_helper import diff_standardize_label, diff_label, standardize_label, standardize_video
from utils import get_adjusted_task_times, resample_signal, upsample_video, calculate_peaks

from functools import partial
from multiprocessing import Pool


def preprocess_videos(configs, participant, save_path):
    # Load frames and resize to specified size
    frames = np.load(configs['original_data_path'] + f'/{participant}/{participant}_et.npy')
    frames = [cv2.resize(frames[i], (configs['w'], configs['h']), interpolation=cv2.INTER_AREA) for i in
              range(len(frames))]
    N_chunks = len(frames) // configs['clip_length']
    frames = np.expand_dims(np.asarray(frames), axis=3)
    frames = frames[:N_chunks * configs['clip_length'], :, :, :]  # remove extra frames that do not fit into a chunk

    # Ensure that only frames within all tasks are used and exclude, e.g., frames after the study protocol
    # task_times = get_adjusted_task_times(configs['task_times'][participant])
    # frames = frames[task_times['video'][0]:task_times['walking_3'][1], :, :, :]

    # Normalize/standardize frames over entire video of one participant
    frames_processed = list()
    for video_type in configs['video_types']:
        f_c = frames.copy()
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

    # Chunk and save data. Chunking is done using task times to enable clean evaluation per task later on.
    task_times = get_adjusted_task_times(configs['task_times'][participant])
    chunk_idx = 0
    for task in task_times.keys():
        frames_task = frames_processed[task_times[task][0]:task_times[task][1]]
        frames_task = frames_task[0::configs['downsampling']]  # downsample frames if specified

        # Interpolate between frames if upsampling is used
        if configs['upsampling'] > 1:
            frames_task = upsample_video(frames_task, configs['upsampling'], 'linear')

        # Show first preprocessed image for validation
        if task == 'video':
            fig, ax = plt.subplots()
            ax.imshow(frames_task[0, :, :, :1])
            ax.set_title(f'Participant {participant}')
            fig.show()
            if frames_task.shape[3] > 1:
                fig, ax = plt.subplots()
                ax.imshow(frames_task[0, :, :, 1:])
                fig.show()

        # Chunk and save frames
        frame_chunks = chunk_data(frames_task, configs['clip_length'])
        frame_chunks = np.transpose(frame_chunks, (0, 4, 1, 2, 3))   # save as (N, C, D, W, H) for PyTorch
        save_chunks(frame_chunks, save_path + f'/{participant}_input_et', chunk_idx)
        chunk_idx += len(frame_chunks)


def preprocess_timeseries(configs, participant, save_path):
    task_times = get_adjusted_task_times(configs['task_times'][participant])
    for signal in ['ppg_nose', 'imu_right']:
        data_all = np.load(configs['original_data_path'] + f'/{participant}/{participant}_{signal}.npy')

        # Filter PPG signal to filter out any motion artifacts or low-frequent component
        if signal == 'ppg_nose':
            data_all = nk.ppg_clean(data_all, sampling_rate=configs['fs_all']['et'], method='elgendi')

        # Extend 1 dim if only 1 dim
        if len(data_all.shape) == 1:
            data_all = np.expand_dims(data_all, axis=1)

        # Ensure that only data within all tasks are used and exclude, e.g., data after the study protocol
        # data_all = data_all[task_times['video'][0]:task_times['walking_3'][1], :]

        # Normalize/standardize data over entire data of one participant
        for i in range(data_all.shape[1]):
            if configs["label_type"] == "Raw":
                data_all[:, i] = data_all[:, i]
            elif configs["label_type"] == "Diff":
                data_all[:, i] = diff_label(data_all[:, i])
            elif configs["label_type"] == "DiffStandardized":
                data_all[:, i] = diff_standardize_label(data_all[:, i])
            elif configs["label_type"] == "Standardized":
                data_all[:, i] = standardize_label(data_all[:, i])
            else:
                os.rmdir(save_path)
                raise ValueError("Unsupported label type for EDA!")

        # Chunk and save data. Chunking is done using task times to enable clean evaluation per task later on.
        chunk_idx = 0
        for task in task_times.keys():
            data = data_all[task_times[task][0]:task_times[task][1]]

            # Downsample signal
            data = data[0::configs['downsampling']]

            # Interpolate between samples if upsampling is used
            if configs['upsampling'] > 1:
                data = resample_signal(data, data.shape[0] * configs['upsampling'] - 2, 'linear')

            # Calculate magnitude of IMU signal and squeeze PPG signal to have shape (N,)
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

            chunk_idx += len(label_chunks)

        # Check that number of video and biosignal chunks match
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
    do_video = True
    do_timesignal = True
    use_mp = True  # use multiprocessing

    # Load configuration files
    cfg_path = f'../configs/preprocessing/config_preprocessing_egoppg.yml'
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

