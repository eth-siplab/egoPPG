import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml

from preprocessing_helper import get_egoexo4d_takes

from functools import partial
from multiprocessing import Pool


def preprocess_video(take, configs, save_path_ml):
    # Get general parameters
    take_name = take['video_paths']['ego'].split('/')[1]
    video_file = take['video_paths']['ego'].split('/')[3][:-10]
    input_vid = cv2.VideoCapture(configs['original_data_path'] + f'/takes/{take_name}/frame_aligned_videos/'
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

            # Plot some frames during preprocessing for verification
            """if counter_frames_preprocessed in [configs['clip_length'], configs['clip_length'] * 100]:
                fig, ax = plt.subplots()
                ax.imshow(frames[0], cmap='grey')
                fig.show()"""

            # Save chunk of frames as .npy file for extended preprocessing for ML
            np.save(save_path_ml + f'/{take_name}_input_et{clip_num}', frames)
            frames = []
            clip_num += 1

        # Read new frame and print status
        success, frame_orig = input_vid.read()

    # Release all video related objects
    input_vid.release()
    cv2.destroyAllWindows()


def preprocess_egoppg(take, configs, save_path_ml):
    preprocess_video(take, configs, save_path_ml)
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
        p = Pool(processes=150)
        prod_x = partial(preprocess_egoppg, configs=configs, save_path_ml=save_path_ml)
        p.map(prod_x, takes)
    else:
        for take in takes:
            preprocess_egoppg(take, configs, save_path_ml)


if __name__ == "__main__":
    main()

