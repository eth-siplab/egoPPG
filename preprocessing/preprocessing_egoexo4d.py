import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

from preprocessing_helper import get_egoexo4d_takes, chunk_data, save_chunks
from preprocessing_helper import diff_standardize_extended_video, diff_standardize_video, diff_video, standardize_video
from preprocessing_helper import diff_standardize_label, diff_label, standardize_label
from utils import resample_signal, upsample_video

from functools import partial
from multiprocessing import Pool
from pathlib import Path
from projectaria_tools.core import data_provider


def preprocess_data(take, configs, save_path):
    # Get general parameters
    take_name = take['video_paths']['ego'].split('/')[1]
    video_file = take['video_paths']['ego'].split('/')[3][:-10]
    input_vid = cv2.VideoCapture(configs['original_data_path'] + f'/takes/{take_name}/frame_aligned_videos/'
                                                                 f'downscaled/448/{video_file}_211-1.mp4')
    input_vid.set(cv2.CAP_PROP_POS_MSEC, 0)
    total_frames = int(input_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    success, frame_orig = input_vid.read()

    # Filter out takes with less than 160 frames to be able to run predictions on 128 frames later
    if total_frames < 192:
        raise ValueError(f'Take {take_name} has less than 192 frames!')

    # Check that no health care or bike repair videos are included as they are not included for the benchmark
    if take['scenario_name'] in ['Health', 'Bike Repair']:
        raise ValueError(f'Take {take_name} has a scenario that is not included in the benchmark!')

    # Process video frames
    continues_black_frames = 0
    frames = []
    while success:
        # As black images were added for EgoExo4D to have 30 fps, we skip these images
        if np.mean(frame_orig) == 0:
            success, frame_orig = input_vid.read()
            continues_black_frames += 1
            if continues_black_frames > 2:
                raise ValueError(f'{continues_black_frames} continuous black frames in take {take_name}!')
            continue
        continues_black_frames = 0

        # Resize image and append to list
        frame_orig = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2GRAY)
        frames.append(cv2.resize(frame_orig, (configs['w'], configs['h']), interpolation=cv2.INTER_AREA))

        # Read new frame and print status
        success, frame_orig = input_vid.read()

    # Release all video related objects
    input_vid.release()
    cv2.destroyAllWindows()

    # Downsample and upsample frames if specified
    frames = np.asarray(frames)
    if len(frames.shape) == 3:
        frames = np.expand_dims(frames, axis=3)

    if configs['downsampling'] > 1:
        frames = frames[0::configs['downsampling']]  # downsample frames if specified

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

    # Chunk and save frames
    frame_chunks = chunk_data(frames_processed, configs['clip_length'])
    try:
        frame_chunks = np.transpose(frame_chunks, (0, 4, 1, 2, 3))  # save as (N, C, D, W, H) for PyTorch
    except:
        raise ValueError(f'Error in take {take_name}!')
    save_chunks(frame_chunks, save_path + f'/{take_name}_input_et', 0)

    # Process IMU data
    # Get IMU data from VRS file
    vrs_folder = f'{configs["original_data_path"]}/takes/{take_name}'
    vrs_file_name = list(Path(vrs_folder).glob("*.vrs"))[0].name
    provider = data_provider.create_vrs_data_provider(vrs_folder + '/' + vrs_file_name)
    stream_id = provider.get_stream_id_from_label('imu-right')
    imu_data = {'accel_x': [], 'accel_y': [], 'accel_z': []}
    for index in range(0, provider.get_num_data(stream_id)):
        imu_temp = provider.get_imu_data_by_index(stream_id, index)
        imu_data['accel_x'].append(imu_temp.accel_msec2[0])
        imu_data['accel_y'].append(imu_temp.accel_msec2[1])
        imu_data['accel_z'].append(imu_temp.accel_msec2[2])
    for key in imu_data.keys():
        imu_data[key] = np.asarray(imu_data[key])

    # Resample IMU data to match video clip length and save
    imu_data = np.swapaxes(np.array(list(imu_data.values())), 0, 1)
    imu_data = resample_signal(imu_data, int(frames.shape[0]), 'linear')

    # Process IMU data based on label type
    for i in range(imu_data.shape[1]):
        if configs["label_type"] == "Raw":
            imu_data[:, i] = imu_data[:, i]
        elif configs["label_type"] == "Diff":
            imu_data[:, i] = diff_label(imu_data[:, i])
        elif configs["label_type"] == "DiffStandardized":
            imu_data[:, i] = diff_standardize_label(imu_data[:, i])
        elif configs["label_type"] == "Standardized":
            imu_data[:, i] = standardize_label(imu_data[:, i])
        else:
            os.rmdir(save_path)
            raise ValueError("Unsupported label type for EDA!")

    # Take magnitude of IMU data
    imu_data = np.sqrt(np.sum(np.square(np.vstack((imu_data[:, 0], imu_data[:, 1], imu_data[:, 2]))), axis=0))

    # Chunk and save IMU data
    imu_data_chunks = chunk_data(imu_data, configs['clip_length'])
    save_chunks(imu_data_chunks, save_path + f'/{take_name}_input_imu_right', 0)

    print(f'Finished take {take["video_paths"]["ego"].split("/")[1]}!')


def main():
    use_mp = False

    # Load configuration files
    cfg_path = './configs/preprocessing/config_preprocessing_egoexo4d.yml'
    with open(cfg_path, 'r') as yamlfile:
        configs = yaml.load(yamlfile, Loader=yaml.FullLoader)

    # Create folder to save preprocessed data
    configuration_overall = f'CL{configs["clip_length"]}_W{configs["w"]}_H{configs["h"]}_LabelRaw_VideoTypeRaw'
    configuration_detailed = (f'CL{configs["clip_length"]}_Down{configs["downsampling"]}_'
                              f'W{configs["w"]}_H{configs["h"]}_Label{configs["label_type"]}_VideoType')
    for video_type in configs["video_types"]:
        configuration_detailed += video_type
    if configs['upsampling'] > 1:
        configuration_detailed += f'_Up{configs["upsampling"]}'
    save_path = configs['preprocessed_data_path'] + f'/{configuration_overall}/{configuration_detailed}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f'Saved to path: {save_path}')

    takes = get_egoexo4d_takes(configs['exclusion_list'])

    if use_mp:
        print('Using multiprocessing for data processing!')
        p = Pool(processes=30)
        prod_x = partial(preprocess_data, configs=configs, save_path=save_path)
        p.map(prod_x, takes)
    else:
        for take in takes:
            preprocess_data(take, configs, save_path)


if __name__ == "__main__":
    main()

