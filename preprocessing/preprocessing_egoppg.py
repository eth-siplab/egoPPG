import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yaml

from preprocessing_helper import chunk_data, load_aria_imu, load_movisens_data, load_shimmer_data, load_md_data
from preprocessing_helper import save_chunks
from source.utils import resample_signal, quantile_artifact_removal
from source.sp.signal_filtering import filter_imu

from functools import partial
from multiprocessing import Pool
from projectaria_tools.core import data_provider


def preprocess_video(provider, participant, configs, save_path_sp, save_path_ml, synch_path, start_end_times,
                     do_output_synch=False):
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

    # Output synchronized, cut videos of ET for sharing
    if do_output_synch:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        img_data = provider.get_image_data_by_index(stream_id, 0)
        height = img_data[0].to_numpy_array().astype(float).shape[0]
        width = img_data[0].to_numpy_array().astype(float).shape[1]
        output_vid = cv2.VideoWriter(synch_path + f'/et.mp4', fourcc, fs, (width, height), isColor=False)

    ml_frames, sp_channel_values, exposure_duration = [], [], []
    i_read, i_clip, old_ts, n_dropped_frames = 0, 0, 0, 0

    # with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for i_all in range(0, provider.get_num_data(stream_id)):
        if i_all not in frame_indices:
            continue
        img_data = provider.get_image_data_by_index(stream_id, i_all)
        img = img_data[0].to_numpy_array().copy()
        exposure_duration.append(img_data[1].exposure_duration)

        if i_read == 0:
            np.save(f'/local/home/bjbraun/img_{participant}.npy', img)

        # Check if frames were dropped by camera, e.g., due to overheating and not by clock drift
        time_diff = (img_data[1].capture_timestamp_ns - old_ts) / 1000000  # in ms
        old_ts = img_data[1].capture_timestamp_ns
        if i_read > 0:
            if time_diff > (1/fs*1000 + 10):
                arg_frame = np.argwhere(frame_indices == i_all)[0][0]
                if (frame_indices[arg_frame] - frame_indices[arg_frame - 1]) != 2:
                    print('Time diff in ms:', time_diff)
                    n_dropped_frames += 1
                    # raise RuntimeError('Time diff too high. Frame must have been dropped.')

        # Write output videos
        if do_output_synch:
            output_vid.write(img)

        """if prev_img is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_img, img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mask[..., 0] = angle * 180 / np.pi / 2
            mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            mask2.append(mask[..., 2])
        prev_img = img"""

        # Append frames for ML
        ml_frames.append(img)

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

    """magn_map = np.mean(np.asarray(mask2), axis=0)
    fig, ax = plt.subplots()
    ax.imshow(magn_map)
    fig.show()"""

    # Check if difference of exposure was constant throughout recording
    if len(np.unique(exposure_duration)) > 1:
        raise RuntimeError('Exposure duration was not constant throughout recording')

    # Process ML frames, chunk in clips and save as .npy files
    ml_frames = [cv2.resize(ml_frames[i], (configs['w'], configs['h']), interpolation=cv2.INTER_AREA) for i in
                 range(len(ml_frames))]
    ml_frames_chunks = chunk_data(np.asarray(ml_frames), configs['clip_length'])
    save_chunks(ml_frames_chunks, f'{save_path_ml}/{participant}_input_et', 0)

    # Plot one ML example frame
    fig, ax = plt.subplots()
    ax.imshow(ml_frames[0], cmap='gray')
    ax.set_title(f'Participant {participant}: First ML frame')
    fig.show()

    if do_output_synch:
        output_vid.release()
        cv2.destroyAllWindows()

    np.save(f'{save_path_sp}/et_channel_values_eyes.npy', sp_channel_values)


def preprocess_timeseries(provider, participant, configs, data_path, save_path_sp, save_path_ml, synch_path,
                          start_end_times, do_output_synch=False):
    # Get general parameters
    fs_all = configs['fs_all']
    task_times_full = [configs['task_times'][participant]['video'][0],
                       configs['task_times'][participant]['running'][1]]

    # Load IMU data from Aria glasses, ECG and IMU data from Movisens, IMU and PPG data from Shimmer,
    # and IMU and PPG from Manuel's device
    aria_imu_left, aria_imu_right = load_aria_imu(provider, start_end_times, fs_all)
    ms_ecg, ms_imu = load_movisens_data(data_path)
    ms_ecg = ms_ecg[int(start_end_times['ms'][0] * fs_all['ms_ecg']):
                    int(start_end_times['ms'][1] * fs_all['ms_ecg'])]
    ms_imu = ms_imu[int(start_end_times['ms'][0] * fs_all['ms_imu']):
                    int(start_end_times['ms'][1] * fs_all['ms_imu']), :]
    md_df = load_md_data(data_path)
    md_ppg = md_df['PPG_G'].to_numpy()[int(start_end_times['md'][0] * fs_all['md']):
                                       int(start_end_times['md'][1] * fs_all['md'])]

    # sensors = {'md': md_ppg, 'aria_imu_left': aria_imu_left, 'aria_imu_right': aria_imu_right, 'ms_ecg': ms_ecg,
    #            'ms_imu': ms_imu}
    sensors = {'md': md_ppg}
    """if participant not in configs['exclusion_list_shimmer']:
        shimmer_df = load_shimmer_data(data_path, participant)
        shimmer_ppg = shimmer_df['PPG'].to_numpy()[int(start_end_times['shimmer'][0] * fs_all['shimmer_ppg']):
                                                   int(start_end_times['shimmer'][1] * fs_all['shimmer_ppg'])]
        shimmer_eda = shimmer_df['EDA'].to_numpy()[int(start_end_times['shimmer'][0] * fs_all['shimmer_ppg']):
                                                   int(start_end_times['shimmer'][1] * fs_all['shimmer_ppg'])]
        sensors['shimmer_ppg'] = shimmer_ppg
        sensors['shimmer_eda'] = shimmer_eda"""

    columns_sensors = {'aria_imu_left': ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'],
                       'aria_imu_right': ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'],
                       'ms_ecg': ['ecg'], 'ms_imu': ['accel_z'], 'shimmer_ppg': ['ppg'],
                       'shimmer_eda': ['eda']}
    for sensor in sensors:
        data = sensors[sensor]

        # Filter PPG signal artifact caused by IR sensor from ET camera before resampling
        if sensor == 'md':
            data = quantile_artifact_removal(data, 0.25, 0.75, 3)
        elif sensor == 'ms_imu':
            data = filter_imu(data[:, 2], fs_all[sensor], 'IMU')

        # First, account for clock drift between sensors by resampling signals to same length as Aria glasses
        # Then, cut data to only include time between beginning of first task and ending of last task
        len_aria = start_end_times['aria'][1] - start_end_times['aria'][0]
        start_after_synch = task_times_full[0] / fs_all['et'] - start_end_times['aria'][0]
        end_after_synch = start_after_synch + (task_times_full[1] - task_times_full[0]) / fs_all['et']
        data = resample_signal(data, int(len_aria*fs_all[sensor]), 'linear')
        data = data[int(start_after_synch * fs_all[sensor]):int(end_after_synch * fs_all[sensor])]

        # Plot example segment of each signal
        if sensor in ['ms_ecg', 'shimmer_ppg', 'shimmer_eda', 'md', 'aria_imu_right', 'ms_imu']:
            fig, ax = plt.subplots()
            ax.plot(data[:40 * fs_all[sensor]])
            ax.set_title(f'Participant {participant}: First 40 seconds of {sensor}')
            fig.show()

        # Save synchronized data as .npy for SP usage and as .csv file for sharing
        np.save(f'{save_path_sp}/{sensor}.npy', data, allow_pickle=True)
        if do_output_synch:
            pd.DataFrame(data=data, columns=columns_sensors[sensor]).to_csv(f'{synch_path}/{sensor}.csv', index=False)

        # Resample signals to same fps than Aria glasses and then save as .npy file
        if sensor in ['shimmer_ppg', 'md', 'aria_imu_right', 'shimmer_eda', 'ms_imu']:
            data_downsampled = resample_signal(data, int(data.shape[0]/fs_all[sensor]*fs_all['et']), 'linear')
            if sensor == 'shimmer_ppg':
                label_name = 'ppg_ear'
            elif sensor == 'shimmer_eda':
                label_name = 'eda'
            elif sensor == 'md':
                label_name = 'ppg_nose'
            elif sensor == 'aria_imu_right':
                label_name = 'imu_right'
            elif sensor == 'ms_imu':
                label_name = 'rr'
            else:
                raise ValueError('Sensor not implemented yet!')
            np.save(f'{save_path_ml}/{participant}_label_{label_name}.npy', data_downsampled, allow_pickle=True)


def preprocess_egoppg(participant, configs, save_path_ml, do_output_synch=False):
    data_path = f'{configs["original_data_path"]}/{participant}'
    save_path_sp = configs['preprocessed_data_path'] + f'/Data_SP/{participant}'
    synch_path = configs['synchronized_data_path'] + f'/{participant}'
    if not os.path.exists(f'{synch_path}'):
        os.makedirs(f'{synch_path}')

    # Load start and end times and data provider
    start_end_times = np.load(f'{save_path_sp}/start_end_times.npy', allow_pickle=True).item()
    provider = data_provider.create_vrs_data_provider(f'{data_path}/aria/{participant}.vrs')

    # Preprocess eye tracking video and timeseries data
    preprocess_video(provider, participant, configs, save_path_sp, save_path_ml, synch_path, start_end_times,
                     do_output_synch=do_output_synch)
    # preprocess_timeseries(provider, participant, configs, data_path, save_path_sp, save_path_ml, synch_path,
    #                       start_end_times, do_output_synch=do_output_synch)

    print(f'Finished participant {participant}!')


def main():
    # Variable parameters
    participants = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014',
                    '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025']
    participants = ['001']
    do_output_synch = True
    use_mp = True

    # Fixed parameters
    with open('./configs/preprocessing/config_preprocessing_egoppg.yml', 'r') as yamlfile:
        configs = yaml.load(yamlfile, Loader=yaml.FullLoader)

    ml_config_name = (f'CL{configs["clip_length"]}_W{configs["w"]}_H{configs["h"]}_'
                      f'LabelRaw_VideoTypeRaw/Data')
    save_path_ml = f'{configs["preprocessed_data_path"]}/Data_ML/{ml_config_name}'
    if not os.path.exists(f'{save_path_ml}'):
        os.makedirs(f'{save_path_ml}')

    if use_mp:
        print('Using multiprocessing for data processing!')
        p = Pool(processes=len(participants))
        prod_x = partial(preprocess_egoppg, configs=configs, save_path_ml=save_path_ml, do_output_synch=do_output_synch)
        p.map(prod_x, participants)
    else:
        for participant in participants:
            preprocess_egoppg(participant, configs, save_path_ml, do_output_synch=do_output_synch)


if __name__ == "__main__":
    main()

