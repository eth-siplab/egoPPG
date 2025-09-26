import json
import glob
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import os
import pandas as pd
import yaml

from preprocessing_helper import chunk_data, save_chunks, get_egoexo4d_takes
from preprocessing_helper import diff_standardize_extended_video, diff_standardize_video, diff_video
from preprocessing_helper import diff_standardize_label, diff_label, standardize_label, standardize_video
from utils import upsample_video, resample_signal

from functools import partial
from multiprocessing import Pool


def preprocess_videos(configs, take, data_path, save_path):
    take_name = take['video_paths']['ego'].split('/')[1]
    file_counter = len(glob.glob1(data_path, f"{take_name}_input_et*"))
    if file_counter > 0:
        # Load frames
        frames = list()
        for i in range(file_counter):
            frames.extend(np.load(data_path + f'/{take_name}_input_et{i}.npy'))
        frames = np.asarray(frames)
        if len(frames.shape) == 3:
            frames = np.expand_dims(frames, axis=3)

        # Interpolate between frames if upsampling is used
        if configs['upsampling'] > 1:
            frames = upsample_video(frames, configs['upsampling'], 'linear')

        # Normalize/standardize frames
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

        # Show first preprocessed image for validation
        """fig, ax = plt.subplots()
        ax.imshow(frames_processed[0, :, :, :1], cmap='gray')
        ax.set_title(f'Take {take_name}')
        fig.show()
        if frames_processed.shape[3] > 1:
            fig, ax = plt.subplots()
            ax.imshow(frames_processed[0, :, :, 1:])
            fig.show()"""

        # Chunk and save frames
        frame_chunks = chunk_data(frames_processed, configs['clip_length_new'])
        print(f'Take {take["video_paths"]["ego"].split("/")[1]}\n')
        try:
            frame_chunks = np.transpose(frame_chunks, (0, 4, 1, 2, 3))   # save as (N, C, D, W, H) for PyTorch
        except:
            raise ValueError(f'Error in take {take_name}!')
        save_chunks(frame_chunks, save_path + f'/{take_name}_input_et', 0)
    else:
        raise ValueError(f'No video files found for take {take_name}!')


def preprocess_timeseries(configs, take, data_path, save_path):
    take_name = take['video_paths']['ego'].split('/')[1]
    data = np.load(data_path + f'/{take_name}_input_imu.npy')

    # Interpolate between samples if upsampling is used
    if configs['upsampling'] > 1:
        data = resample_signal(data, data.shape[0] * configs['upsampling'] - 2, 'linear')

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

        # Take magnitude of IMU data
        data = np.sqrt(np.sum(np.square(np.vstack((data[:, 0], data[:, 1], data[:, 2]))), axis=0))

        # Chunk, check that number of files are the same as for video, and save
        data_chunks = chunk_data(data, configs['clip_length_new'])
        save_chunks(data_chunks, save_path + f'/{take_name}_input_2imu_right', 0)

def mp_preprocessing_extended(take, configs, data_path, save_path, do_video, do_timeseries):
    if do_video:
        preprocess_videos(configs, take, data_path, save_path)
    if do_timeseries:
        preprocess_timeseries(configs, take, data_path, save_path)
    print(f'Finished take {take["video_paths"]["ego"].split("/")[1]}\n')


def main():
    use_mp = True
    do_video = False
    do_timeseries = True

    # Load configuration files
    with open('./configs/preprocessing/config_preprocessing_extended_egoexo4d.yml', 'r') as yamlfile:
        configs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    with open('./configs/preprocessing/config_preprocessing_egoexo4d.yml', 'r') as yamlfile:
        configs_pre = yaml.load(yamlfile, Loader=yaml.FullLoader)

    takes = get_egoexo4d_takes(configs_pre['exclusion_list'])

    """f = {}
    for take in takes:
        f[take['video_paths']['ego'].split('/')[1]] = take
    # test = f['uniandes_bouldering_029_76']
    # take_name = test['video_paths']['ego'].split('/')[1]
    takes = [f['uniandes_bouldering_027_65']]"""

    # Get folder where to load data from
    configuration_old = f'CL{configs["clip_length_old"]}_W{configs["w"]}_H{configs["h"]}_LabelRaw_VideoTypeRaw'
    data_path = configs['dir_preprocessed'] + f'/Data_ML/{configuration_old}/Data'

    # Create folder to save preprocessed data
    configuration_new = (f'CL{configs["clip_length_new"]}_Down{configs["downsampling"]}_'
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
        p = Pool(processes=100)
        prod_x = partial(mp_preprocessing_extended, configs=configs, data_path=data_path, save_path=save_path,
                         do_video=do_video, do_timeseries=do_timeseries)
        p.map(prod_x, takes)
    else:
        for take in takes:
            mp_preprocessing_extended(take, configs, data_path, save_path, do_video, do_timeseries)


if __name__ == "__main__":
    main()

