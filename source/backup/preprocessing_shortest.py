import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import yaml

from preprocessing_helper import chunk_data, create_cv_splits, load_aria_imu, load_movisens_data, load_shimmer_data
from preprocessing_helper import save_data, preprocess_frames
from utils import get_ml_config_name, resample_signal

from functools import partial
from multiprocessing import Pool
from projectaria_tools.core import data_provider


# ToDo: Adjust as timeseries to only look for length of Aria glasses (do not drop them!)
def preprocess_video_aria(provider, participant, camera_label, configs_g, save_path_sp, save_path_ml, synch_path,
                          start_end_times, do_output_synch=False):
    # Get general parameters
    task_times_full = configs_g['task_times'][participant]['full']
    stream_id = provider.get_stream_id_from_label(camera_label)
    fs = configs_g['fs_all'][configs_g['abbrev'][camera_label]]

    # Only keep frames between start and end of IMU synchronization
    frame_indices = np.arange(0, provider.get_num_data(stream_id))
    frame_indices = frame_indices[np.where((frame_indices >= start_end_times['aria'][0] * fs) &
                                           (frame_indices <= start_end_times['aria'][1] * fs))]

    # Account for clock drift between sensors by dropping frames evenly spaced between synchronization events
    # Only drop frames if Aria glasses recorded longer than Movisens
    sig_lens = {'aria': None, 'ms': None, 'shimmer': None}
    for device in sig_lens.keys():
        sig_lens[device] = start_end_times[device][1] - start_end_times[device][0]
    min_dev = min(sig_lens, key=sig_lens.get)
    diff_len = ((start_end_times['aria'][1] - start_end_times['aria'][0]) -
                (start_end_times[min_dev][1] - start_end_times[min_dev][0]))
    n_frames = int(start_end_times['aria'][1] * fs) - int(start_end_times['aria'][0] * fs)
    if diff_len > 0:
        drop_indices = np.arange(0, n_frames, int(n_frames / (diff_len * fs))).astype(int)[1:]
        frame_indices = np.delete(frame_indices, drop_indices)

    # Cut frames to only include frames between beginning of first task and ending of last task
    start_after_synch = task_times_full[0] / configs_g['fs_all']['et'] - start_end_times['aria'][0]
    end_after_synch = start_after_synch + (task_times_full[1] - task_times_full[0]) / configs_g['fs_all']['et']
    frame_indices = frame_indices[int(start_after_synch * fs):int(end_after_synch * fs)]

    # Output synchronized, cut videos of ET for sharing
    if do_output_synch:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        img_data = provider.get_image_data_by_index(stream_id, 0)
        height = img_data[0].to_numpy_array().astype(float).shape[0]
        width = img_data[0].to_numpy_array().astype(float).shape[1]
        if camera_label == 'camera-et':
            output_vid = cv2.VideoWriter(synch_path + f'/{configs_g["abbrev"][camera_label]}.mp4', fourcc, fs,
                                         (width, height), isColor=False)
        else:
            output_vid = cv2.VideoWriter(synch_path + f'/{configs_g["abbrev"][camera_label]}.mp4', fourcc, fs,
                                         (width, height), isColor=True)

    ml_frames, sp_channel_values, exposure_duration = [], [], []
    i_read, i_clip, old_ts, n_dropped_frames = 0, 0, 0, 0
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
                    # raise RuntimeError('Time diff too high. Frame must have been dropped.')

        if camera_label == 'camera-rgb':
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Write output videos
        if do_output_synch:
            output_vid.write(img)

        # Append frames for ML
        ml_frames.append(img)

        # Process eye tracking images for signal-processing-based approach
        if camera_label == 'camera-et':
            crop_coords = [25, 75, 0, 640]
            cropped_img = img[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]
            sp_channel_values.append(np.mean(cropped_img))

        # Plot image every 5 minutes
        if i_read == 0 or (i_read % (5 * 60 * fs) == 0):
            fig, ax = plt.subplots()
            if camera_label == 'camera-et':
                ax.imshow(cropped_img, cmap='gray')
            else:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            fig.show()
        i_read += 1

    print(f'Total number of dropped frames for participant {participant}: {n_dropped_frames}')

    # Process ML frames, chunk in clips and save as .npy files
    ml_frames_processed = preprocess_frames(np.asarray(ml_frames), configs_g, save_path_ml)
    ml_frames_clips = chunk_data(ml_frames_processed, configs_g['clip_length'])
    save_data(ml_frames_clips, save_path_ml, participant, 'input_' + configs_g['abbrev'][camera_label])

    # Plot frames before and after preprocessing
    fig, ax = plt.subplots()
    ax.imshow(ml_frames[0], cmap='gray')
    ax.set_title('First frame before preprocessing')
    fig.show()
    fig, ax = plt.subplots()
    ax.imshow(ml_frames_processed[0, :, :, 0], cmap='gray')
    ax.set_title('First frame after preprocessing')
    fig.show()

    if do_output_synch:
        output_vid.release()
        cv2.destroyAllWindows()

    if camera_label == 'camera-et':
        # Check if difference of exposure was constant throughout recording
        if len(np.unique(exposure_duration)) > 1:
            raise RuntimeError('Exposure duration was not constant throughout recording')
        np.save(f'{save_path_sp}/et_channel_values.npy', sp_channel_values)
        return ml_frames_clips.shape[0], sp_channel_values
    else:
        return ml_frames_clips.shape[0], None


def preprocess_timeseries_data(provider, participant, configs_g, data_path, save_path_sp, save_path_ml, synch_path,
                               start_end_times, do_output_synch=False):
    # Get general parameters
    fs_all = configs_g['fs_all']
    task_times_full = configs_g['task_times'][participant]['full']

    # Load IMU data from Aria glasses, ECG and IMU data from Movisens, and IMU and PPG data from Shimmer
    aria_imu_left, aria_imu_right = load_aria_imu(provider, start_end_times, fs_all)
    ms_ecg, ms_imu = load_movisens_data(data_path)
    ms_ecg = ms_ecg[int(start_end_times['ms'][0] * fs_all['ms_ecg']):
                    int(start_end_times['ms'][1] * fs_all['ms_ecg'])]
    ms_imu = ms_imu[int(start_end_times['ms'][0] * fs_all['ms_imu']):
                    int(start_end_times['ms'][1] * fs_all['ms_imu']), :]
    # ToDo: Implement for PPG once available
    shimmer_df = load_shimmer_data(data_path)
    shimmer_ppg = shimmer_df['PPG'].to_numpy()[int(start_end_times['shimmer'][0] * fs_all['shimmer']):
                                               int(start_end_times['shimmer'][1] * fs_all['shimmer'])]

    n_ppg_clips = None
    sensors = {'aria_imu_left': aria_imu_left, 'aria_imu_right': aria_imu_right, 'ms_ecg': ms_ecg, 'ms_imu': ms_imu,
               'shimmer': shimmer_ppg}
    columns_sensors = {'aria_imu_left': ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'],
                       'aria_imu_right': ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'],
                       'ms_ecg': ['ecg'], 'ms_imu': ['accel_x', 'accel_y', 'accel_z'], 'shimmer': ['ppg']}
    sensor_device_mapping = {'aria_imu_left': 'aria', 'aria_imu_right': 'aria', 'ms_ecg': 'ms', 'ms_imu': 'ms',
                             'shimmer': 'shimmer'}

    # Get sensor with the shortest signal length
    sig_lens = {'aria': None, 'ms': None, 'shimmer': None}
    for device in sig_lens.keys():
        sig_lens[device] = start_end_times[device][1] - start_end_times[device][0]
    min_dev = min(sig_lens, key=sig_lens.get)

    for sensor in sensors:
        data = sensors[sensor]

        # Account for clock drift by dropping samples evenly spaced between synchronization events for all
        # signals which are longer than the signal with the shortest length
        sensor_name = sensor_device_mapping[sensor]
        diff_len = ((start_end_times[sensor_name][1] - start_end_times[sensor_name][0]) -
                    (start_end_times[min_dev][1] - start_end_times[min_dev][0]))
        if diff_len > 0:
            drop_indices = np.arange(0, data.shape[0],
                                     int(data.shape[0] / (abs(diff_len) * fs_all[sensor]))).astype(int)[1:]
            data = np.delete(data, drop_indices, axis=0)

        # Cut data to only include time between beginning of first task and ending of last task
        start_after_synch = task_times_full[0] / fs_all['et'] - start_end_times['aria'][0]
        end_after_synch = start_after_synch + (task_times_full[1] - task_times_full[0]) / fs_all['et']
        data = data[int(start_after_synch * fs_all[sensor]):int(end_after_synch * fs_all[sensor])]

        # Save synchronized data as .npy for own usage and as .csv file for sharing
        np.save(f'{save_path_sp}/{sensor}.npy', data, allow_pickle=True)
        if do_output_synch:
            pd.DataFrame(data=data, columns=columns_sensors[sensor]).to_csv(f'{synch_path}/{sensor}.csv', index=False)

        # Resample PPG signal to same fps than Aria glasses and save as .npy batches for ML
        if sensor == 'shimmer':
            ppg_downsampled = resample_signal(data, int(data.shape[0]/fs_all[sensor]*fs_all['et']), 'cubic')
            ppg_clips = chunk_data(ppg_downsampled, configs_g['clip_length'])
            save_data(ppg_clips, save_path_ml, participant, 'labels_shimmer')
            n_ppg_clips = ppg_clips.shape[0]

    return n_ppg_clips


def preprocess_egoppg(participant, camera_label, configs_g, do_timeseries=False, do_output_synch=False):
    data_path = f'{configs_g["original_data_path"]}/{participant}'
    save_path_sp = configs_g['preprocessed_data_path'] + f'/Data_SP/{participant}'
    ml_config_name = get_ml_config_name(configs_g)
    save_path_ml = f'{configs_g["preprocessed_data_path"]}/Data_ML/{ml_config_name}'
    synch_path = configs_g['synchronized_data_path'] + f'/{participant}'
    if not os.path.exists(f'{save_path_ml}'):
        os.makedirs(f'{save_path_ml}')
    if not os.path.exists(f'{synch_path}'):
        os.makedirs(f'{synch_path}')

    # Load start and end times and data provider
    start_end_times = np.load(f'{save_path_sp}/start_end_times.npy', allow_pickle=True).item()
    provider = data_provider.create_vrs_data_provider(f'{data_path}/aria/{participant}.vrs')

    # Preprocess eye tracking video
    n_vid_files, channel_values = preprocess_video_aria(provider, participant, camera_label, configs_g, save_path_sp,
                                                        save_path_ml, synch_path, start_end_times,
                                                        do_output_synch=do_output_synch)
    # ToDo: Change!
    # channel_values = None

    # Preprocess timeseries data
    if do_timeseries:
        n_ts_files = preprocess_timeseries_data(provider, participant, configs_g, data_path, save_path_sp,
                                                save_path_ml, synch_path, start_end_times,
                                                do_output_synch=do_output_synch)
        """if n_ts_files != n_vid_files:
            raise RuntimeError('Different number of clips for video and timeseries data!')"""

    print(f'Finished participant {participant}!')
    return channel_values


def main():
    # Variable parameters
    participants = ['variable_1']  # 'variable_1', 'variable_2', 'variable_3', synch-test2, test30
    camera_label = 'camera-et'  # camera-et, camera-rgb
    do_output_synch = False
    do_timeseries = True
    use_mp = False

    # Fixed parameters
    with open('../configs/preprocessing/config_preprocessing_egoppg.yml', 'r') as yamlfile:
        configs_g = yaml.load(yamlfile, Loader=yaml.FullLoader)

    # ToDo: Check if multiprocessing works properly
    if use_mp:
        print('Using multiprocessing for data processing!')
        p = Pool(processes=len(participants))
        prod_x = partial(preprocess_egoppg, camera_label=camera_label, configs_g=configs_g, do_timeseries=do_timeseries,
                         do_output_synch=do_output_synch)
        _ = p.map(prod_x, participants)
    else:
        for participant in participants:
            _ = preprocess_egoppg(participant, camera_label, configs_g, do_timeseries=do_timeseries,
                                  do_output_synch=do_output_synch)

    # create_cv_splits('egoPPG', participants, camera_label, configs_g)


if __name__ == "__main__":
    main()


# Tries for VRS but could potentially work on Ubunutu!
# new_arr = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype('uint8')
# gray = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY)
# calib_et_left = provider.get_device_calibration().get_camera_calib('camera-et-left')
# calib_et_left = provider.get_device_calibration().get_camera_calib('camera-et-right')
# pinhole = calibration.get_linear_camera_calibration(512, 512, 150)
# undistorted_image = calibration.distort_by_calibration(img_left, pinhole, calib_et_left)

# img = cv2.imread(img_files[index], cv2.IMREAD_GRAYSCALE)

# Old synchronization
# Account for clock drift between sensors by dropping frames evenly spaced
# n_frames = int(start_end_times['aria'][1] * fs) - int(start_end_times['aria'][0] * fs)
# diff_len = ((start_end_times['aria'][1] - start_end_times['aria'][0]) -
#             (start_end_times['ms'][1] - start_end_times['ms'][0]))
# if diff_len > 0:
#     drop_indices = np.arange(0, n_frames, int(n_frames/(diff_len*fs))).astype(int)[1:]
#     frames = np.delete(frames, drop_indices, axis=0)
# Cut frames to only include frames between beginning of first task and ending of last task
# start_after_synch = task_times_full[0] / fs - start_end_times['aria'][0]
# end_after_synch = start_after_synch + (task_times_full[1] - task_times_full[0]) / fs
# frames = frames[int(start_after_synch * fs):int(end_after_synch * fs), :, :]


"""# Another old synchronization
# Videos
# Account for clock drift between sensors by dropping frames evenly spaced between synchronization events
# Only drop frames if Aria glasses recorded longer than Movisens
diff_len = ((start_end_times['aria'][1] - start_end_times['aria'][0]) -
            (start_end_times['ms'][1] - start_end_times['ms'][0]))
n_frames = int(start_end_times['aria'][1] * fs) - int(start_end_times['aria'][0] * fs)
if diff_len > 0:
    drop_indices = np.arange(0, n_frames, int(n_frames / (diff_len * fs))).astype(int)[1:]
    frame_indices = np.delete(frame_indices, drop_indices)
    
# Timeseries
diff_len = ((start_end_times['aria'][1] - start_end_times['aria'][0]) -
                (start_end_times['ms'][1] - start_end_times['ms'][0]))
if diff_len < 0:
    drop_indices_ecg = np.arange(0, ms_ecg.shape[0],
                                 int(ms_ecg.shape[0]/(abs(diff_len)*fs_all['ms_ecg']))).astype(int)[1:]
    ms_ecg = np.delete(ms_ecg, drop_indices_ecg, axis=0)
    drop_indices_imu = np.arange(0, ms_imu.shape[0],
                                 int(ms_imu.shape[0]/(abs(diff_len)*fs_all['ms_imu']))).astype(int)[1:]
    ms_imu = np.delete(ms_imu, drop_indices_imu, axis=0)
else:
    for key in aria_imu_left:
        drop_indices_left = np.arange(0, aria_imu_left[key].shape[0],
                                      int(aria_imu_left[key].shape[0] /
                                          (abs(diff_len) * fs_all['aria_imu_left']))).astype(int)[1:]
        aria_imu_left[key] = np.delete(aria_imu_left[key], drop_indices_left, axis=0)
        drop_indices_right = np.arange(0, aria_imu_right[key].shape[0],
                                       int(aria_imu_right[key].shape[0] /
                                           (abs(diff_len) * fs_all['aria_imu_right']))).astype(int)[1:]
        aria_imu_right[key] = np.delete(aria_imu_right[key], drop_indices_right, axis=0)

# Cut data to only include time between beginning of first task and ending of last task
start_after_synch = task_times_full[0] / fs_all['et'] - start_end_times['aria'][0]
end_after_synch = start_after_synch + (task_times_full[1] - task_times_full[0]) / fs_all['et']
ms_ecg = ms_ecg[int(start_after_synch * fs_all['ms_ecg']):int(end_after_synch * fs_all['ms_ecg'])]
ms_imu = ms_imu[int(start_after_synch * fs_all['ms_imu']):int(end_after_synch * fs_all['ms_imu']), :]
for key in aria_imu_left:
    aria_imu_left[key] = aria_imu_left[key][int(start_after_synch * fs_all['aria_imu_left']):
                                            int(end_after_synch * fs_all['aria_imu_left'])]
    aria_imu_right[key] = aria_imu_right[key][int(start_after_synch * fs_all['aria_imu_right']):
                                              int(end_after_synch * fs_all['aria_imu_right'])]

# Save synchronized data as .npy for own usage
np.save(f'{save_path_sp}/ms_ecg.npy', ms_ecg)
np.save(f'{save_path_sp}/ms_imu.npy', ms_imu)
np.save(f'{save_path_sp}/aria_imu_left.npy', aria_imu_left, allow_pickle=True)
np.save(f'{save_path_sp}/aria_imu_right.npy', aria_imu_right, allow_pickle=True)

# Save synchronized data as .npy files in chunks for ML
# ToDo: Do for PPG signal and think about use of IMU for rPPG
n_ppg_clips = 0

# Save synchronized data as .csv file for sharing
if do_output_synch:
    pd.DataFrame(data=ms_ecg, columns=['ecg']).to_csv(f'{synch_path}/ms_ecg.csv', index=False)
    pd.DataFrame(data=ms_imu, columns=['accel_x', 'accel_y', 'accel_z']
                 ).to_csv(f'{synch_path}/ms_imu.csv', index=False)
    pd.DataFrame(data=aria_imu_left, columns=['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
                 ).to_csv(f'{synch_path}/aria_imu_left.csv', index=False)
    pd.DataFrame(data=aria_imu_right, columns=['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
                 ).to_csv(f'{synch_path}/aria_imu_right.csv', index=False)"""

# Stable mask accumulator
"""stable_mask_accumulator = None
for index in range(len(img_files)):
    if index not in frame_indices:
        continue
    img = cv2.imread(img_files[index], cv2.IMREAD_GRAYSCALE)

    crop_coords = [25, 75, 0, 640]
    img = img[crop_coords[0]:crop_coords[1], crop_coords[2]:crop_coords[3]]

    _, skin_mask = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    # skin_mask = cv2.inRange(img, 30, 150)
    if stable_mask_accumulator is None:
        stable_mask_accumulator = skin_mask.astype(float)
    else:
        stable_mask_accumulator = cv2.accumulateWeighted(skin_mask, stable_mask_accumulator, 0.1)
stable_mask = cv2.convertScaleAbs(stable_mask_accumulator)
# img = cv2.bitwise_and(img, img, mask=stable_mask)"""

