import cv2
import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio
import yaml

from functools import partial
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from multiprocessing import Pool
from natsort import natsorted

from source.utils import get_participants_list
from source.preprocessing.preprocessing_multi_helper import resize_face_frames_mediapipe, sample, resize_frames


def get_save_name_participant(dataset_name, participant, i):
    if dataset_name == 'ubfc_rppg':
        save_name_participant = participant.replace('subject', 's')
    elif dataset_name == 'egoppg':
        save_name_participant = participant
    elif dataset_name == 'pure':
        if participant == '06':
            tasks = ['01', '03', '04', '05', '06']
            save_name_participant = participant + f'_T{tasks[i]}'
        else:
            save_name_participant = participant + f'_T0{i + 1}'
    elif dataset_name == 'mmpd':
        save_name_participant = participant.replace('subject', 's') + f'_T{str(i)}'
    else:
        raise RuntimeError('Dataset has not been implemented yet!')

    return save_name_participant


def preprocess_video(configs, dataset_name, participant, clip_length, save_path, plot_images=True):
    video_files = []
    if dataset_name == 'ubfc_rppg':
        video_files.append(configs['original_data_path'] + f'/{participant}/vid.avi')
    elif dataset_name == 'pure':
        if participant == '06':
            tasks = ['01', '03', '04', '05', '06']
        else:
            tasks = ['01', '02', '03', '04', '05', '06']
        for task in tasks:
            img_files = glob.glob(configs['original_data_path'] + f'/{participant}-{task}/{participant}-{task}/*.png')
            img_files.sort()
            video_files.append(img_files)
    elif dataset_name == 'mmpd':
        mat_files = natsorted(os.listdir(configs['original_data_path'] + f'/{participant}'))
        video_files = [f'{configs["original_data_path"]}/{participant}/{mat_file}' for mat_file in mat_files]
    else:
        raise RuntimeError('Dataset has not been implemented yet!')

    # Read in video files and save chunks
    old_coords = None
    for i, video_file in enumerate(video_files):
        pure_counter = 0
        mmpd_counter = 0

        # Get video capture object
        if dataset_name == 'ubfc_rppg':
            VidObj = cv2.VideoCapture(video_file, cv2.CAP_FFMPEG)
            VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
            success, frame = VidObj.read()  # h, w, c
        elif dataset_name == 'pure':
            frame = cv2.imread(video_file[pure_counter])
            success = 1
        elif dataset_name == 'mmpd':
            mat = sio.loadmat(video_file)
            frames_mat = np.array(mat['video'])
            frame = frames_mat[0]
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
            if frames_mat.shape[0] > 0:
                success = 1
        else:
            raise RuntimeError('Dataset has not been implemented yet!')

        # Define detection model
        """base_options = python.BaseOptions(model_asset_path=configs['landmark_model_path'])
        options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True, num_faces=1)
        detector = vision.FaceLandmarker.create_from_options(options)"""

        frames = list()
        clip_num = 0
        counter = 0
        counter_frames = 0
        while success:
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            if dataset_name in ['mmpd']:
                frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            frames.append(frame)

            if dataset_name == 'ubfc_rppg':
                success, frame = VidObj.read()
            elif dataset_name == 'pure':
                pure_counter += 1
                if pure_counter == len(video_file):
                    success = 0
                if success:
                    frame = cv2.imread(video_file[pure_counter])
                    if frame is None:
                        pure_counter += 1
                        frame = cv2.imread(video_file[pure_counter])
            elif dataset_name == 'mmpd':
                mmpd_counter += 1
                if mmpd_counter < frames_mat.shape[0]:
                    frame = frames_mat[mmpd_counter]
                else:
                    success = 0
                frame = (frame * 255).clip(0, 255).astype(np.uint8)

            counter_frames += 1
            counter += 1

            # Process and save every clip_length frames
            if (counter_frames % clip_length) == 0:
                print('Finished frame number: ', counter_frames)
                frames = resize_frames(np.asarray(frames), configs, participant)  # old approach
                # frames, old_coords = resize_face_frames_mediapipe(frames, configs, detector, old_coords, participant)

                # Show resized frames at the beginning and end
                if plot_images:
                    if clip_num == 1 or clip_num == 50:
                        fig, ax = plt.subplots()
                        frame_show = np.asarray(frames[0], dtype=np.uint8)
                        ax.imshow(frame_show)
                        fig.show()

                # Save frames as .npy file for extended preprocessing for ML
                save_name_participant = get_save_name_participant(dataset_name, participant, i)
                np.save(save_path + f'/{save_name_participant}_input_{configs["input_name"]}{clip_num}', frames)
                frames = []
                clip_num += 1
        if dataset_name == 'ubfc_rppg':
            VidObj.release()

def preprocess_biosignals(configs, dataset_name, participant, clip_length, save_path):
    files = []
    counters_pure = []
    if dataset_name == 'ubfc_rppg':
        files.append(configs['original_data_path'] + f'/{participant}/ground_truth.txt')
    elif dataset_name == 'pure':
        if participant == '06':
            tasks = ['01', '03', '04', '05', '06']
        else:
            tasks = ['01', '02', '03', '04', '05', '06']
        for task in tasks:
            counters_pure.append(len(glob.glob(configs['original_data_path'] + f'/{participant}-{task}/{participant}-{task}/*.png')))
            files_temp = configs['original_data_path'] + f'/{participant}-{task}/{participant}-{task}.json'
            files.append(files_temp)
    elif dataset_name == 'mmpd':
        mat_files = natsorted(os.listdir(configs['original_data_path'] + f'/{participant}'))
        files = [f'{configs["original_data_path"]}/{participant}/{mat_file}' for mat_file in mat_files]
    else:
        raise RuntimeError('Dataset has not been implemented yet!')

    for i, file in enumerate(files):
        # Load data of each dataset
        if dataset_name == 'ubfc_rppg':
            ppg = np.loadtxt(file)
            ppg = ppg[0, :]  # 0: BVP signal, 1: HR, 2: time
            VidObj = cv2.VideoCapture(configs['original_data_path'] + f'/{participant}/vid.avi')
            num_frames = int(VidObj.get(cv2.CAP_PROP_FRAME_COUNT))
            if ppg.shape[0] != num_frames:
                raise RuntimeError('Number of frames in video and ppg signal do not match!')
        elif dataset_name == 'pure':
            with open(file, "r") as f:
                labels = json.load(f)
                waves = [label["Value"]["waveform"]
                         for label in labels["/FullPackage"]]
            ppg = np.asarray(waves)
            ppg = sample(ppg, counters_pure[i])
        elif dataset_name == 'mmpd':
            mat = sio.loadmat(file)
            ppg = np.array(mat['GT_ppg']).T.reshape(-1)
        else:
            raise RuntimeError('Dataset has not been implemented yet!')

        # Plot one example signal
        if i == 0:
            fig, ax = plt.subplots()
            ax.plot(ppg[:1000])
            ax.set_title(f'PPG Signal of dataset {dataset_name} for participant {participant}')
            fig.show()

        # Save data
        save_name_participant = get_save_name_participant(dataset_name, participant, i)
        np.save(save_path + f'/{save_name_participant}_label_ppg', ppg)


def mp_preprocessing(participant, configs, dataset_name, clip_length, save_path, do_preprocess_biosignal,
                     do_preprocess_video):
    if do_preprocess_video:
        preprocess_video(configs, dataset_name, participant, clip_length, save_path)
    if do_preprocess_biosignal:
        preprocess_biosignals(configs, dataset_name, participant, clip_length, save_path)

    print(f"Finished preprocessing participant {participant}!")


def main():
    # Specify settings for processing
    dataset_names = ['ubfc_rppg']  # pure, ubfc_rppg, mmpd
    use_mp = True
    do_preprocess_video = True
    do_preprocess_biosignal = True

    # Run through different dataset
    for dataset_name in dataset_names:
        cfg_path = f'./configs/preprocessing/config_preprocessing_{dataset_name}.yml'
        with open(cfg_path, 'r') as yamlfile:
            configs = yaml.load(yamlfile, Loader=yaml.FullLoader)

        # Load participants list
        participants = get_participants_list(dataset_name)
        # participants = [participants[1]]

        # Process videos and biosignals for each clip length and each subject
        for clip_length in configs['clip_lengths']:
            configurations_name = f'CL{clip_length}_W{configs["w"]}_H{configs["h"]}_' \
                                  f'LabelRaw_VideoTypeRaw'
            save_path = configs['preprocessed_data_path'] + f'/Data_ML/{configurations_name}/Data'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                print(f'Saved to path: {save_path}')

            if use_mp:
                p = Pool(processes=len(participants))
                prod_x = partial(mp_preprocessing, configs=configs, dataset_name=dataset_name, clip_length=clip_length,
                                 save_path=save_path, do_preprocess_biosignal=do_preprocess_biosignal,
                                 do_preprocess_video=do_preprocess_video)
                p.map(prod_x, participants)
            else:
                for i_participant, participant in enumerate(participants):
                    mp_preprocessing(participant, configs, dataset_name, clip_length, save_path,
                                     do_preprocess_biosignal, do_preprocess_video)


if __name__ == "__main__":
    main()
