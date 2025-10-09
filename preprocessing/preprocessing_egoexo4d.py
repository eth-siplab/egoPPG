import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

from preprocessing_helper import get_egoexo4d_takes
from utils import resample_signal

from functools import partial
from multiprocessing import Pool
from projectaria_tools.core import data_provider


def preprocess_data(take, configs, save_path_ml):
    # Get general parameters
    take_name = take['video_paths']['ego'].split('/')[1]
    video_file = take['video_paths']['ego'].split('/')[3][:-10]
    input_vid = cv2.VideoCapture(configs['original_data_path'] + f'/EgoExo4D_full/takes/{take_name}/frame_aligned_videos/'
                                                                 f'{video_file}_211-1.mp4')
    input_vid.set(cv2.CAP_PROP_POS_MSEC, 0)
    total_frames = int(input_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    success, frame_orig = input_vid.read()

    # Filter out takes with less than 160 frames to be able to run predictions on 128 frames later
    if total_frames < 192:
        raise ValueError(f'Take {take_name} has less than 192 frames!')

    # Check that no health care or bike repair videos are included as they are not included for the benchmark
    if take['scenario_name'] in ['Health', 'Bike Repair']:
        raise ValueError(f'Take {take_name} has a scenario that is not included in the benchmark!')

    # Plot first image of each participant for verification
    """fig, ax = plt.subplots()
    ax.imshow(np.asarray(frame_orig, dtype=np.uint8), cmap='grey')
    fig.show()"""

    # Process video frames
    clip_num = 0
    counter_frames_preprocessed = 0
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
        frames.append(frame_orig)
        counter_frames_preprocessed += 1

        if (counter_frames_preprocessed % configs['clip_length']) == 0 and counter_frames_preprocessed != 0:
            frames = [cv2.resize(frames[i], (configs['w'], configs['h']), interpolation=cv2.INTER_AREA) for i
                      in range(len(frames))]

            # Save chunk of frames as .npy file for extended preprocessing for ML
            np.save(save_path_ml + f'/{take_name}_input_et{clip_num}', frames)
            frames = []
            clip_num += 1

        # Read new frame and print status
        success, frame_orig = input_vid.read()

    # Release all video related objects
    input_vid.release()
    cv2.destroyAllWindows()

    # Process VRS file
    vrs_folder = f'{configs["original_data_path"]}/EgoExo4D_VRS/takes/{take_name}'
    vrs_file_name = os.listdir(vrs_folder)[0]
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
    imu_data = np.swapaxes(np.array(list(imu_data.values())), 0, 1)
    imu_data = resample_signal(imu_data, int(clip_num * configs['clip_length']), 'linear')
    np.save(f'{save_path_ml}/{take_name}_input_imu.npy', imu_data, allow_pickle=True)

    print(f'Finished take {take["video_paths"]["ego"].split("/")[1]}!')


def main():
    use_mp = True

    # Fixed parameters
    with open('./configs/preprocessing/config_preprocessing_egoexo4d.yml', 'r') as yamlfile:
        configs = yaml.load(yamlfile, Loader=yaml.FullLoader)

    ml_config_name = (f'CL{configs["clip_length"]}_W{configs["w"]}_H{configs["h"]}_'
                      f'LabelRaw_VideoTypeRaw/Data')
    save_path_ml = f'{configs["preprocessed_data_path"]}/Data_ML/{ml_config_name}'
    if not os.path.exists(f'{save_path_ml}'):
        os.makedirs(f'{save_path_ml}')

    takes = get_egoexo4d_takes(configs['exclusion_list'])

    """f = {}
    for take in takes:
        f[take['video_paths']['ego'].split('/')[1]] = take
    takes = [f['uniandes_bouldering_027_65']]"""

    if use_mp:
        print('Using multiprocessing for data processing!')
        p = Pool(processes=30)
        prod_x = partial(preprocess_data, configs=configs, save_path_ml=save_path_ml)
        p.map(prod_x, takes)
    else:
        for take in takes:
            preprocess_data(take, configs, save_path_ml)


if __name__ == "__main__":
    main()

