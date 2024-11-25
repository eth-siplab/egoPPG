import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io
import yaml

from preprocessing_helper import load_movisens_data, load_shimmer_data, load_md_data
from source.utils import normalize

from projectaria_tools.core import data_provider
from scipy.signal import butter, find_peaks


def main():
    # Variable parameters
    participants = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014',
                    '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025']
    participants = ['022']
    start_cut_1 = 40  # 40
    start_cut_2 = 60  # 40, 60
    end_cut_1 = 60  # 50
    end_cut_2 = 20  # 30

    # Fixed parameters
    with open('./configs/preprocessing/config_preprocessing_egoppg.yml', 'r') as yamlfile:
        configs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    fs_all = configs['fs_all']
    data_path = configs['original_data_path']
    save_path = configs['preprocessed_data_path']

    for participant in participants:
        print('Starting participant: ', participant)
        if participant in configs['exclusion_list_shimmer']:
            imu_devices = ['aria', 'ms', 'md']
        else:
            imu_devices = ['aria', 'ms', 'shimmer', 'md']
        data_path_temp = f'{data_path}/{participant}'
        save_path_temp = f'{save_path}/Data_SP/{participant}'

        # Creat save folder if it does not exist
        if not os.path.exists(f'{save_path_temp}'):
            os.makedirs(f'{save_path_temp}')

        # Get IMU data from Aria glasses
        provider = data_provider.create_vrs_data_provider(f'{data_path_temp}/aria/{participant}.vrs')
        stream_id = provider.get_stream_id_from_label('imu-right')
        imu_data_aria = {'accel_x': [], 'accel_y': [], 'accel_z': [], 'gyro_x': [], 'gyro_y': [], 'gyro_z': []}
        for index in range(0, provider.get_num_data(stream_id)):
            imu_temp_aria = provider.get_imu_data_by_index(stream_id, index)
            imu_data_aria['accel_x'].append(imu_temp_aria.accel_msec2[0])
            imu_data_aria['accel_y'].append(imu_temp_aria.accel_msec2[1])
            imu_data_aria['accel_z'].append(imu_temp_aria.accel_msec2[2])
            imu_data_aria['gyro_x'].append(imu_temp_aria.gyro_radsec[0])
            imu_data_aria['gyro_y'].append(imu_temp_aria.gyro_radsec[1])
            imu_data_aria['gyro_z'].append(imu_temp_aria.gyro_radsec[2])

        # Get IMU data from Movisens, Shimmer, and Manuel's device
        ecg, imu_data_movisens = load_movisens_data(data_path_temp)
        if 'shimmer' in imu_devices:
            shimmer_df = load_shimmer_data(data_path_temp, participant)
        if 'md' in imu_devices:
            md_df = load_md_data(data_path_temp)

        # Find start and end times for each IMU sensor
        # ToDo: Check if minus or plus gives better results
        possible_peaks = {imu_device: {'start': [], 'end': []} for imu_device in imu_devices}
        start_end_times = {imu_device: list() for imu_device in imu_devices}
        for imu_device in imu_devices:
            if imu_device == 'aria':
                fs_temp = fs_all['aria_imu_right']
                imu_temp = -np.asarray(imu_data_aria['accel_y'])
            elif imu_device == 'ms':
                fs_temp = fs_all['ms_imu']
                imu_temp = -np.asarray(imu_data_movisens[:, 1])
            elif imu_device == 'shimmer':
                fs_temp = fs_all['shimmer']
                if participant == '022':
                    imu_temp = -shimmer_df['ACC_Y_WR'].to_numpy()
                else:
                    imu_temp = -shimmer_df['ACC_Y'].to_numpy()
            elif imu_device == 'md':
                fs_temp = fs_all['md']
                imu_temp = -md_df['ACC_Z_H'].to_numpy()
            else:
                raise RuntimeError(f'IMU device {imu_device} not implemented yet!')

            imu_segments = [imu_temp[start_cut_1 * fs_temp:start_cut_2 * fs_temp],
                            imu_temp[-end_cut_1 * fs_temp:-end_cut_2 * fs_temp:]]
            for i_segment, imu_segment in enumerate(imu_segments):
                segment = 'start' if i_segment == 0 else 'end'
                [b, a] = butter(2, 10 / fs_temp * 2, btype='highpass')
                imu_filt = scipy.signal.filtfilt(b, a, imu_segment, axis=0)
                imu_filt = normalize(imu_filt, 'zero_one')
                peaks = find_peaks(imu_filt, height=0.8)[0]  # distance=0.5 * fs_temp)[0]
                y_peaks = np.asarray([imu_filt[i] for i in peaks])

                # Plot filtered signal and found peaks
                fig, ax = plt.subplots()
                ax.plot(imu_filt)
                ax.scatter(peaks, y_peaks, color='green')
                ax.set_title(f'Participant {participant}: {imu_device} - {segment}')
                fig.show()

                # Take the highest peak for the Aria device. For the other devices, take all peaks and select afterward.
                if imu_device == 'aria':
                    possible_peaks[imu_device][segment].append(peaks[np.argmax(y_peaks)])
                else:
                    possible_peaks[imu_device][segment].append(peaks)

                # Add cut time from start to possible peaks time/subtract possible peaks time from length of signals
                if segment == 'start':
                    possible_peaks[imu_device][segment] = [start_cut_1*fs_temp + possible_peaks[imu_device][segment][i]
                                                           for i in range(len(possible_peaks[imu_device][segment]))]
                else:
                    possible_peaks[imu_device][segment] = [imu_temp.shape[0] - (end_cut_1 * fs_temp -
                                                                                possible_peaks[imu_device][segment][i])
                                                           for i in range(len(possible_peaks[imu_device][segment]))]

                # Get start and end times
                if imu_device == 'aria':
                    start_end_times[imu_device].append(possible_peaks[imu_device][segment][0] / fs_temp)
                else:
                    peak_times_temp = [possible_peaks[imu_device][segment][i] / fs_temp for
                                       i in range(len(possible_peaks[imu_device][segment]))]
                    diff_peaks = abs((possible_peaks['aria'][segment][0] / fs_all['aria_imu_right']) - peak_times_temp)
                    start_end_times[imu_device].append(possible_peaks[imu_device][segment][0][np.argmin(diff_peaks)] /
                                                       fs_temp)

        # Check if distance between start and end times of devices differ too much from Aria glasses
        threshold_diff = 10
        print(f'Participant {participant}: Start and end times of devices: {start_end_times}')
        for imu_device in imu_devices:
            if (abs(start_end_times['aria'][0] - start_end_times[imu_device][0]) > threshold_diff or
                    abs(start_end_times['aria'][1] - start_end_times[imu_device][1]) > threshold_diff or
                    abs(abs(start_end_times['aria'][1]-start_end_times['aria'][0]) -
                        abs(start_end_times[imu_device][1]-start_end_times[imu_device][0])) > threshold_diff):
                raise RuntimeError(f'Distance between start and end times of devices differ too much')

        # Save start end times
        np.save(f'{save_path_temp}/start_end_times.npy', start_end_times, allow_pickle=True)


if __name__ == "__main__":
    main()
