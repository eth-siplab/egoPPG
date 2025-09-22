import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yaml

from preprocessing_helper import chunk_data, load_aria_imu, load_movisens_data, load_shimmer_data, load_biopac_data
from preprocessing_helper import save_chunks
from source.utils import resample_signal, quantile_artifact_removal
from source.sp.signal_filtering import get_edr

from functools import partial
from multiprocessing import Pool
from projectaria_tools.core import data_provider


def preprocess_video(provider, participant, configs, save_path_sp, start_end_times):
    # Get general parameters
    task_times_full = [configs['task_times'][participant]['video'][0],
                       configs['task_times'][participant]['running'][1]]
    stream_id = provider.get_stream_id_from_label('camera-et')
    fs = configs['fs_all']['et']

    # Only keep frames between start and end of IMU synchronization
    frame_indices = np.arange(0, provider.get_num_data(stream_id))
    frame_indices = frame_indices[np.where((frame_indices >= start_end_times['aria'][0] * fs) &
                                           (frame_indices <= start_end_times['aria'][1] * fs))]

    # Cut frames to only include frames between beginning of first task and ending of last task
    start_after_synch = task_times_full[0] / configs['fs_all']['et'] - start_end_times['aria'][0]
    end_after_synch = start_after_synch + (task_times_full[1] - task_times_full[0]) / configs['fs_all']['et']
    frame_indices = frame_indices[int(start_after_synch * fs):int(end_after_synch * fs)]

    ml_frames, sp_channel_values, exposure_duration = [], [], []
    i_read, i_clip, old_ts, n_dropped_frames = 0, 0, 0, 0

    # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i_all in range(0, provider.get_num_data(stream_id)):
        if i_all not in frame_indices:
            continue
        img_data = provider.get_image_data_by_index(stream_id, i_all)
        img = img_data[0].to_numpy_array().copy()
        exposure_duration.append(img_data[1].exposure_duration)

        # Check if frames were dropped by camera, e.g., due to overheating and not by clock drift
        time_diff = (img_data[1].capture_timestamp_ns - old_ts) / 1000000  # in ms
        old_ts = img_data[1].capture_timestamp_ns
        if i_read > 0:
            if time_diff > (1/fs*1000 + 10):
                arg_frame = np.argwhere(frame_indices == i_all)[0][0]
                if (frame_indices[arg_frame] - frame_indices[arg_frame - 1]) != 2:
                    print('Time diff in ms:', time_diff)
                    n_dropped_frames += 1

        # Process eye tracking images for signal-processing-based approach
        crop_coords = configs['crop_coords'][participant]
        # crop_coords = configs['crop_coords_eyes'][participant]  # [200, 240, 0, 640]  # [25, 75, 0, 640], [10, 50, 0, 640]
        cropped_img = img[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
        # cropped_img = img[crop_coords[0]:crop_coords[1]+20, 50:200]
        sp_channel_values.append(np.mean(cropped_img))

        # Plot image every 5 minutes
        if i_read == 0:
            # np.save(f'/local/home/bjbraun/img_{participant}.npy', img)
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Participant {participant}: First ET frame')
            fig.show()
        if i_read == 0 or (i_read % (15 * 60 * fs) == 0):
            fig, ax = plt.subplots()
            ax.imshow(cropped_img, cmap='gray')
            ax.set_title(f'Participant {participant}: ET frame {i_read}')
            fig.show()
        i_read += 1
    print(f'Total number of dropped frames for participant {participant}: {n_dropped_frames}')

    # Check if difference of exposure was constant throughout recording
    if len(np.unique(exposure_duration)) > 1:
        raise RuntimeError('Exposure duration was not constant throughout recording')

    np.save(f'{save_path_sp}/et_channel_values_eyes.npy', sp_channel_values)


def preprocess_timeseries(provider, participant, configs, data_path, save_path_sp, save_path_ml, start_end_times):
    # Get general parameters
    fs_all = configs['fs_all']
    task_times_full = [configs['task_times'][participant]['video'][0],
                       configs['task_times'][participant]['walking_3'][1]]

    # Load IMU data from Aria glasses, ECG and IMU data from Movisens, IMU and PPG data from Shimmer,
    # and IMU and PPG from Manuel's device
    ms_ecg, ms_imu = load_movisens_data(data_path)
    ms_ecg = ms_ecg[int(start_end_times['ms'][0] * fs_all['ms_ecg']):
                    int(start_end_times['ms'][1] * fs_all['ms_ecg'])]
    ms_imu = ms_imu[int(start_end_times['ms'][0] * fs_all['ms_imu']):
                    int(start_end_times['ms'][1] * fs_all['ms_imu'])]
    biopac = load_biopac_data(data_path)
    biopac = biopac[int(start_end_times['biopac'][0] * fs_all['biopac']):
                    int(start_end_times['biopac'][1] * fs_all['biopac'])]

    sensors = {'biopac': biopac, 'ms_ecg': ms_ecg, 'ms_imu': ms_imu}
    for sensor in sensors:
        data = sensors[sensor]

        # First, account for clock drift between sensors by resampling signals to same length as Aria glasses
        # Then, cut data to only include time between beginning of first task and ending of last task
        len_aria = start_end_times['aria'][1] - start_end_times['aria'][0]
        start_after_synch = task_times_full[0] / fs_all['et'] - start_end_times['aria'][0]
        end_after_synch = start_after_synch + (task_times_full[1] - task_times_full[0]) / fs_all['et']
        data = resample_signal(data, int(len_aria*fs_all[sensor]), 'linear')
        data = data[int(start_after_synch * fs_all[sensor]):int(end_after_synch * fs_all[sensor])]

        # Plot example segment of each signal
        if sensor in ['ms_ecg', 'biopac', 'ms_imu']:
            fig, ax = plt.subplots()
            ax.plot(data[:20 * fs_all[sensor]])
            ax.set_title(f'Participant {participant}: First 20 seconds of {sensor}')
            fig.show()

        # Save synchronized data as .npy for SP usage and as .csv file for sharing
        np.save(f'{save_path_sp}/{sensor}.npy', data, allow_pickle=True)


def preprocess_egorr(participant, configs, save_path_ml):
    data_path = f'{configs["original_data_path"]}/{participant}'
    save_path_sp = configs['preprocessed_data_path'] + f'/Data_SP/{participant}'

    # Load start and end times and data provider
    start_end_times = np.load(f'{save_path_sp}/start_end_times.npy', allow_pickle=True).item()
    provider = data_provider.create_vrs_data_provider(f'{data_path}/aria/{participant}.vrs')

    if not os.path.exists(save_path_sp):
        os.makedirs(save_path_sp)

    preprocess_video(provider, participant, configs, save_path_sp, start_end_times)
    preprocess_timeseries(provider, participant, configs, data_path, save_path_sp, save_path_ml, start_end_times)

    print(f'Finished participant {participant}!')


def main():
    # Variable parameters
    participants = ['001', '002', '003', '004']
    # participants = ['004']
    use_mp = True

    # Fixed parameters
    with open('./configs/preprocessing/config_preprocessing_egorr.yml', 'r') as yamlfile:
        configs = yaml.load(yamlfile, Loader=yaml.FullLoader)

    ml_config_name = (f'CL{configs["clip_length"]}_W{configs["w"]}_H{configs["h"]}_'
                      f'LabelRaw_VideoTypeRaw/Data')
    save_path_ml = f'{configs["preprocessed_data_path"]}/Data_ML/{ml_config_name}'
    if not os.path.exists(f'{save_path_ml}'):
        os.makedirs(f'{save_path_ml}')

    if use_mp:
        print('Using multiprocessing for data processing!')
        p = Pool(processes=len(participants))
        prod_x = partial(preprocess_egorr, configs=configs, save_path_ml=save_path_ml)
        p.map(prod_x, participants)
    else:
        for participant in participants:
            preprocess_egorr(participant, configs, save_path_ml)


if __name__ == "__main__":
    main()

