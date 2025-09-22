import cv2
import glob
import json
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy.io
import sqlite3
import yaml

# from src.ml.preprocessing.artifact_removal import artifact_detection
from functools import partial
from multiprocessing import Pool
from natsort import natsorted
from preprocessing_helper import *
from PIL import Image
from scipy.signal import butter
import scipy.io as sio

from functools import partial
from multiprocessing import Pool
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def read_mat(mat_file):
    try:
        mat = sio.loadmat(mat_file)
    except:
        for _ in range(20):
            print(mat_file)
    frames = np.array(mat['video'])
    return frames


def preprocess_videos(configs, participant, dataset_name, save_path):
    # Load and resize frames
    mat_files = natsorted(os.listdir(configs['dir_data'] + f'/{participant}'))
    mat_files = [f'{configs["dir_data"]}/{participant}/{mat_file}' for mat_file in mat_files]
    frames = []
    for mat_file in mat_files:
        mat = sio.loadmat(mat_file)
        frames.extend(np.array(mat['video']))
    frames = np.array(frames)
    frames = (np.round(frames * 255)).astype(np.uint8)
    frames = resize_frames(np.asarray(frames), configs, participant)

    # Chunk and save frames
    clip_num = 0
    frames_clips = chunk_frames(frames, configs['clip_length'])
    for frames_clip in frames_clips:
        # Show resized frames at the beginning and end
        if clip_num == 1 or clip_num == 50:
            fig, ax = plt.subplots()
            frame_show = np.asarray(frames_clip[0], dtype=np.uint8)
            ax.imshow(frame_show)
            fig.show()

        # Save chunk of frames as .npy file for extended preprocessing for ML
        np.save(save_path + f'/{participant}_input_face{clip_num}', frames_clip)
        clip_num += 1


def preprocess_biosignals(configs, participant, dataset_name, save_path):
    mat_files = natsorted(os.listdir(configs['dir_data'] + f'/{participant}'))
    mat_files = [f'{configs["dir_data"]}/{participant}/{mat_file}' for mat_file in mat_files]

    bvps = []
    for mat_file in mat_files:
        mat = sio.loadmat(mat_file)
        bvps_temp = np.array(mat['GT_ppg']).T.reshape(-1)
        bvps.extend(bvps_temp)

    np.save(save_path + f'/{participant}_label_ppg.npy', np.asarray(bvps))

    # Plot for integrity check
    fig, ax = plt.subplots()
    ax.plot(bvps[:1000])
    fig.show()


def mp_preprocessing(participant, configs, dataset_name, do_preprocess_video, do_preprocess_biosignal, save_path_ml):
    if do_preprocess_video:
        preprocess_videos(configs, participant, dataset_name, save_path_ml)
    if do_preprocess_biosignal:
        preprocess_biosignals(configs, participant, dataset_name, save_path_ml)

    print(f'Finished participant {participant}\n')


def main():
    # Specify settings for processing
    dataset_names = ['mmpd']  # eda_dataset, biovid, ubfc, rafael, bp4d+, pure, ubfc_rppg
    use_mp = True
    do_preprocess_video = True
    do_preprocess_biosignal = True

    # Run through different dataset
    for dataset_name in dataset_names:
        cfg_path = f'../configs/preprocessing/config_preprocessing_{dataset_name}.yml'
        with open(cfg_path, 'r') as yamlfile:
            configs = yaml.load(yamlfile, Loader=yaml.FullLoader)

        # Load subject names
        if dataset_name == 'mmpd':
            participants = ['1', '2', '3', '4', '5', '6', '7', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                            '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33']
            participants = ['subject' + participant for participant in participants]
        else:
            print('Dataset has not been implemented yet!')
            raise RuntimeError

        # Create folder to save preprocessed data for ML
        configuration_name = f'CL{configs["clip_length"]}_DL{configs["detection_length"]}_W{configs["w"]}_H{configs["h"]}_' \
                             f'LabelRaw_DataTypeRaw'
        save_path_ml = configs['dir_preprocessed'] + f'/Data_ML/{configuration_name}/Data'
        if not os.path.exists(save_path_ml):
            os.makedirs(save_path_ml)
            print(f'Saved to path: {save_path_ml}')

        # Process videos and EDA for each clip length and each subject
        if use_mp:
            print('Using multiprocessing for data processing!')
            p = Pool(processes=len(participants))
            prod_x = partial(mp_preprocessing, configs=configs, dataset_name=dataset_name,
                             do_preprocess_video=do_preprocess_video, do_preprocess_biosignal=do_preprocess_biosignal,
                             save_path_ml=save_path_ml)
            p.map(prod_x, participants)
        else:
            for participant in participants:
                mp_preprocessing(participant, configs, dataset_name, do_preprocess_video, do_preprocess_biosignal,
                                 save_path_ml)


if __name__ == "__main__":
    main()
