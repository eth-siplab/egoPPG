import glob
import numpy as np

from utils import get_task_chunk_list

from functools import partial
from multiprocessing import Pool
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold


def take_last_ele(ele):
    ele = ele.split('.')[0][-3:]
    try:
        return int(ele[-3:])
    except ValueError:
        try:
            return int(ele[-2:])
        except ValueError:
            return int(ele[-1:])


def get_tvt_splits(train_index, valid_index, test_index, participants_temp, input_file_names):
    print(f"  Train: index={train_index}")
    print(f"  Valid: index={valid_index}")
    print(f"  Test:  index={test_index}")

    input_files_train = []
    for index in train_index:
        input_files_train.extend(input_file_names[participants_temp[index]])
    input_files_valid = []
    for index in valid_index:
        input_files_valid.extend(input_file_names[participants_temp[index]])
    input_files_test = []
    for index in test_index:
        input_files_test.extend(input_file_names[participants_temp[index]])

    return input_files_train, input_files_valid, input_files_test


def get_input_paths(participant, data_path, config):
    inputs_part = glob.glob1(data_path, f"{participant}_input_{config.INPUT_SIGNALS[0]}*")
    if len(inputs_part) == 0:
        raise ValueError(f'No video files found for participant {participant}!')
    inputs_part = sorted(inputs_part, key=take_last_ele)
    if len(config.TASKS_TO_USE) > 0:
        task_chunk_list = get_task_chunk_list(config, participant)
        inputs_part = [inputs_part[i] for i in range(len(inputs_part)) if task_chunk_list['keep'][i] == 1]
    inputs_part = [data_path + '/' + input_file for input_file in inputs_part]
    return inputs_part


def get_split_files(config, participants, data_path, random_seed, cross_dataset=False):
    input_file_names = {}
    use_mp = True
    """if use_mp:
        print('Using multiprocessing for data processing!')
        p = Pool(processes=50)
        prod_x = partial(get_input_paths, data_path=data_path, config=config)
        results = p.map(prod_x, participants)
    else:
        results = []
        for participant in participants:
            results.append(get_input_paths(participant, data_path, config))
    for i_result in range(len(results)):
        input_file_names[participants[i_result]] = results[i_result]"""
    for participant in participants:
        inputs_part = glob.glob1(data_path, f"{participant}_input_{config.INPUT_SIGNALS[0]}*")
        if len(inputs_part) == 0:
            print(f'Folder: {data_path}')
            raise ValueError(f'No video files found for participant {participant}!')
        inputs_part = sorted(inputs_part, key=take_last_ele)
        if len(config.TASKS_TO_USE) > 0:
            task_chunk_list = get_task_chunk_list(config, participant)
            # inputs_part = inputs_part[:len(task_chunk_list['task_names'])]
            inputs_part = [inputs_part[i] for i in range(len(inputs_part)) if task_chunk_list['keep'][i] == 1]
        inputs_part = [data_path + '/' + input_file for input_file in inputs_part]
        input_file_names[participant] = inputs_part

    split_files = {'train': {}, 'valid': {}, 'test': {}}
    if cross_dataset:
        print('Cross dataset split:')
        train_index, valid_index = train_test_split(np.arange(len(participants)), random_state=random_seed,
                                                    test_size=0.1)
        test_index = np.arange(len(participants))
        train_index.sort()
        valid_index.sort()
        split_files['train']['cross_dataset'] = [input_file_names[participants[i]] for i in train_index]
        split_files['valid']['cross_dataset'] = [input_file_names[participants[i]] for i in valid_index]
        split_files['test']['cross_dataset'] = [input_file_names[participants[i]] for i in test_index]
    else:
        if config.SPLIT_METHOD == 'loso':
            splitter = LeaveOneOut()
        elif config.SPLIT_METHOD == 'kfold':
            splitter = KFold(n_splits=config.K_FOLD_SPLITS)
        else:
            raise ValueError('Split method not implemented yet!')

        splitter.get_n_splits(participants)
        for i_split, (train_index, test_index) in enumerate(splitter.split(participants)):
            train_index, valid_index = train_test_split(train_index, random_state=random_seed, test_size=2)
            train_index.sort()
            valid_index.sort()

            if config.SPLIT_METHOD == 'loso':
                split_name = 'loso_' + participants[i_split]
            elif config.SPLIT_METHOD == 'kfold':
                split_name = f'kfold{str(config.K_FOLD_SPLITS)}_' + str(i_split)
            else:
                raise ValueError('Splitter name not implemented yet!')
            split_files['train'][split_name] = [input_file_names[participants[i]] for i in train_index]
            split_files['valid'][split_name] = [input_file_names[participants[i]] for i in valid_index]
            split_files['test'][split_name] = [input_file_names[participants[i]] for i in test_index]

    return split_files['train'], split_files['valid'], split_files['test']


def get_cv_split_files(config, participants_train_valid, participants_test, random_seed):
    cv_split_files = {}
    if config.TRAIN.DATA.DATASET == config.TEST.DATA.DATASET:
        cv_split_files['train'], cv_split_files['valid'], cv_split_files['test'] = (
            get_split_files(config, participants_train_valid, config.TRAIN.DATA.CACHED_PATH, random_seed))
    else:
        cv_split_files['train'], cv_split_files['valid'], _ = (
            get_split_files(config, participants_train_valid, config.TRAIN.DATA.CACHED_PATH, random_seed, cross_dataset=True))
        _, _, cv_split_files['test'] = (
            get_split_files(config, participants_test, config.TEST.DATA.CACHED_PATH, random_seed, cross_dataset=True))

    return cv_split_files
