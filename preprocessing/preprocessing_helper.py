import cv2
import glob
import json
import numpy as np
import os
import pandas as pd
import unisens


# %% General functions
def chunk_data(data, clip_length):
    """Chunks the data into clips."""
    clip_num = data.shape[0] // clip_length
    data_clips = [data[i * clip_length:(i + 1) * clip_length] for i in range(clip_num)]
    return np.asarray(data_clips)


def save_chunks(data, path, start_index):
    for i in range(len(data)):
        output_path = path + f'{i+start_index}'
        np.save(output_path, data[i])


def take_last_ele(ele):
    ele = ele.split('.')[0][-3:]
    try:
        return int(ele[-3:])
    except ValueError:
        try:
            return int(ele[-2:])
        except ValueError:
            return int(ele[-1:])


def load_aria_imu(provider, start_end_times, fs_all):
    labels = ["imu-left", "imu-right"]
    aria_imu = {}
    for label in labels:
        stream_id = provider.get_stream_id_from_label(label)
        imu_data = {'accel_x': [], 'accel_y': [], 'accel_z': [], 'gyro_x': [], 'gyro_y': [], 'gyro_z': []}
        if label == 'imu-left':
            fs_temp = fs_all['aria_imu_left']
        else:
            fs_temp = fs_all['aria_imu_right']
        for index in range(0, provider.get_num_data(stream_id)):
            if (index < start_end_times['aria'][0] * fs_temp or
                    index > start_end_times['aria'][1] * fs_temp):
                continue
            imu_temp = provider.get_imu_data_by_index(stream_id, index)
            imu_data['accel_x'].append(imu_temp.accel_msec2[0])
            imu_data['accel_y'].append(imu_temp.accel_msec2[1])
            imu_data['accel_z'].append(imu_temp.accel_msec2[2])
            imu_data['gyro_x'].append(imu_temp.gyro_radsec[0])
            imu_data['gyro_y'].append(imu_temp.gyro_radsec[1])
            imu_data['gyro_z'].append(imu_temp.gyro_radsec[2])

        for key in imu_data.keys():
            imu_data[key] = np.asarray(imu_data[key])
        imu_data = np.swapaxes(np.array(list(imu_data.values())), 0, 1)
        aria_imu[label] = imu_data

    return aria_imu['imu-left'], aria_imu['imu-right']


def load_movisens_data(data_path):
    u = unisens.Unisens(f'{data_path}/movisens')

    # Alternatively, use predicted HRs from Movisens
    """a = u.ecg_bin.get_data()[0, :]
    ab = u.bpmbxb_live_csv.get_data()
    abn = np.asarray(ab)
    abc = u.hrvisvalid_live_bin.get_data()
    hbs = []
    for i in range(len(ab)):
        if i == 0:
            hbs.extend([ab[i][1]] * ab[i][0])
        elif i == (len(ab) - 1):
            hbs.extend([ab[i][1]] * (ab[i][0] - ab[i - 1][0]))
            hbs.extend([ab[i][1]] * (len(a) - ab[i][0]))
        else:
            hbs.extend([ab[i][1]] * (ab[i][0] - ab[i-1][0]))

    import matplotlib.pyplot as plt
    import neurokit2 as nk
    min = 23
    fig, ax = plt.subplots()
    signals_ecg, info_ecg = nk.ecg_process(a[int(min * 60 * 1024) : int((min+0.5) * 60 * 1024)], sampling_rate=1024, method='neurokit')
    ax.plot(signals_ecg['ECG_Clean'])
    # ax.plot(a[int(min * 60 * 1024) : int((min+0.5) * 60 * 1024)])
    fig.show()
    return hbs, np.swapaxes(u.acc_bin.get_data(), 0, 1)  # angularrate_bin"""

    return u.ecg_bin.get_data()[0, :], np.swapaxes(u.acc_bin.get_data(), 0, 1)  # angularrate_bin


def load_shimmer_data(data_path, participant):
    if participant in ['001', '002', '003', '004', '005', '007', '008', '009', '010', '011', '012', '013', '014',
                       '015', '016']:
        use_cols = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13]
    else:
        use_cols = [0, 1, 2, 3, 4, 5, 6, 9, 11, 12, 13, 14]
    shimmer_df = pd.read_csv(data_path + '/shimmer.csv', sep='\t', skiprows=[0, 1, 2], header=None, usecols=use_cols,
                             names=['timestamps', 'ACC_X', 'ACC_Y', 'ACC_Z', 'ACC_X_WR', 'ACC_Y_WR', 'ACC_Z_WR', 'EDA',
                                    'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'PPG'])
    return shimmer_df


def load_md_data(data_path):
    md_df = pd.read_csv(data_path + '/md.csv', sep=',', skiprows=[0], header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                        names=['PPG_G', 'PPG_R', 'PPG_IR', 'ACC_X_H', 'ACC_Y_H', 'ACC_Z_H', 'ACC_X_S', 'ACC_Y_S',
                               'ACC_Z_S'])
    return md_df


def load_biopac_data(data_path):
    # Get recordings from BIOPAC
    with open(data_path + f'/biopac.txt') as f:
        lines = f.readlines()

    column_names = ['timestamps', 'ECG', 'RR']
    biopac_df = pd.DataFrame([np.double(line.split('\t')[:-1]) for line in lines[11:]], columns=column_names)

    return biopac_df.loc[:, 'RR']


def get_egoexo4d_takes(exclustion_list):
    path_base = '/local/home/bjbraun/Datasets/OriginalData/EgoExo4D'

    with open(path_base + '/annotations/proficiency_demonstrator_train.json', 'r') as f:
        train_splits_official = json.load(f)['annotations']
    with open(path_base + '/annotations/proficiency_demonstrator_val.json', 'r') as f:
        val_splits_official = json.load(f)['annotations']

    takes_out = []
    for take in train_splits_official:
        take_name = take['video_paths']['ego'].split('/')[1]
        if (take_name in os.listdir(path_base + '/takes')) and (take_name not in exclustion_list):
            takes_out.append(take)
    for take in val_splits_official:
        take_name = take['video_paths']['ego'].split('/')[1]
        if take_name in os.listdir(path_base + '/takes') and (take_name not in exclustion_list):
            takes_out.append(take)

    return takes_out
