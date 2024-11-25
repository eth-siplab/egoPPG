import argparse
import json
import numpy as np
import os
import random
import torch
import yaml

from source.ml import data_loader, trainer, ml_helper
from source.ml.config import get_config
from source.preprocessing.preprocessing_helper import get_egoexo4d_takes
from torch.utils.data import DataLoader

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Create a general generator for use with the validation dataloader,
    # the test dataloader, and the unsupervised dataloader
    general_generator = torch.Generator()
    general_generator.manual_seed(random_seed)
    # Create a training generator to isolate the train dataloader from
    # other dataloaders and better control non-deterministic behavior
    train_generator = torch.Generator()
    train_generator.manual_seed(random_seed)

    return general_generator, train_generator


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_and_test(config, data_loader_dict, trainer_params):
    if config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'TSCAN':
        model_trainer = trainer.TSCANTrainer.TSCANTrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'PhysNet':
        model_trainer = trainer.PhysNetTrainer.PhysNetTrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'PhysNetNew':
        model_trainer = trainer.PhysNetNewTrainer.PhysNetNewTrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'PhysNetNewIMU':
        model_trainer = trainer.PhysNetNewIMUTrainer.PhysNetNewIMUTrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'PhysNetNewIMUCA':
        model_trainer = trainer.PhysNetNewIMUCATrainer.PhysNetNewIMUCATrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'PhysNetIMU':
        model_trainer = trainer.PhysNetIMUTrainer.PhysNetIMUTrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'PhysNetNewTwo':
        model_trainer = trainer.PhysNetNewTwoTrainer.PhysNetNewTwoTrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'PhysMamba':
        model_trainer = trainer.PhysMambaTrainer.PhysMambaTrainer(config, data_loader_dict, trainer_params)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.train(data_loader_dict)
    return model_trainer.test(data_loader_dict)


def test(config, data_loader_dict, trainer_params):
    if config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'TSCAN':
        model_trainer = trainer.TSCANTrainer.TSCANTrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'PhysNet':
        model_trainer = trainer.PhysNetTrainer.PhysNetTrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'PhysNetNew':
        model_trainer = trainer.PhysNetNewTrainer.PhysNetNewTrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'PhysNetNewIMU':
        model_trainer = trainer.PhysNetNewIMUTrainer.PhysNetNewIMUTrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'PhysNetNewIMUCA':
        model_trainer = trainer.PhysNetNewIMUCATrainer.PhysNetNewIMUCATrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'PhysNetIMU':
        model_trainer = trainer.PhysNetIMUTrainer.PhysNetIMUTrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'PhysNetNewTwo':
        model_trainer = trainer.PhysNetNewTwoTrainer.PhysNetNewTwoTrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'PhysMamba':
        model_trainer = trainer.PhysMambaTrainer.PhysMambaTrainer(config, data_loader_dict, trainer_params)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    return model_trainer.test(data_loader_dict)


def main():
    # %% Data input & output
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, help='Name of the configuration file')
    args = parser.parse_args()
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

    # %% Load configs
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')

    # Parameters to change
    if config.TRAIN.DATA.DATASET == "egoppg":
        # For now, 001, 003, 004, 005, 016, and 020 are excluded
        participants = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                        '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025']
        # participants = ['001', '002', '003', '004', '007', '008', '010', '011', '012', '013', '014', '015', '019',
        #                 '022']
        # participants = ['001', '002', '003', '004', '005', '006', '007', '008']
        # participants = ['005', '006', '007', '008', '009']
        # participants = ['002']
    # ToDo: Implement for EgoExo4D dataset
    else:
        raise ValueError("Unsupported dataset! Currently supporting EDA_Dataset.")

    if config.TEST.DATA.DATASET == "egoexo4d":
        # Fixed parameters
        with open('./configs/preprocessing/config_preprocessing_egoexo4d.yml', 'r') as yamlfile:
            configs_pre = yaml.load(yamlfile, Loader=yaml.FullLoader)
        takes = get_egoexo4d_takes(configs_pre['exclusion_list'])
        takes = [take['video_paths']['ego'].split('/')[1] for take in takes]
        # takes = takes[0:10]
    else:
        takes = None

    random_seeds = [0, 10, 100]
    random_seeds = [0]

    # Define metrics and tasks to evaluate
    test_metrics = {'ppg_nose': ['MAE', 'RMSE', 'MAPE', 'Pearson'], 'ppg_ear': ['MAE', 'RMSE', 'MAPE'],
                    'hr': ['MAE', 'RMSE', 'MAPE', 'Pearson']}

    # Loop through random seeds
    results_all_seeds_mean = {label_signal: {task: {metric: list() for metric in test_metrics[label_signal]}
                              for task in config.TASKS_EVALUATE} for label_signal in config.LABEL_SIGNALS}
    results_all_seeds_std = {label_signal: {task: {metric: list() for metric in test_metrics[label_signal]}
                             for task in config.TASKS_EVALUATE} for label_signal in config.LABEL_SIGNALS}
    for random_seed in random_seeds:
        # Set seed for different random seeds
        print(f'\nRandom seed: {random_seed}')
        general_generator, train_generator = set_seed(random_seed)

        # Get cv splits depending on defined cv split method and dataset
        cv_split_files = ml_helper.get_cv_split_files(config, participants, takes, random_seed)

        # Create dirs for logging models and training/testing outputs
        model_dir = config.MODEL.PATH_MODEL + f'/seed_{random_seed}'
        log_training_dir = config.LOG.PATH_TRAINING + f'/seed_{random_seed}'
        log_testing_dir = config.LOG.PATH_TESTING + f'/seed_{random_seed}'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(log_training_dir):
            os.makedirs(log_training_dir)
        if not os.path.exists(log_testing_dir):
            os.makedirs(log_testing_dir)

        # Figure to plot results of all participants (only relevant for BP to validate results visually)
        fig_all = plt.figure(figsize=(200, 100))
        fig_all.suptitle(f'Predicted vs GT signals', fontsize=20)
        spec_all = gridspec.GridSpec(ncols=7, nrows=6, figure=fig_all)

        # Loop through cross-validation splits
        results_seed = {label_signal: {task: {metric: list() for metric in test_metrics[label_signal]}
                                       for task in config.TASKS_EVALUATE} for label_signal in config.LABEL_SIGNALS}
        for i_split, cv_split in enumerate(cv_split_files['train'].keys()):
            # ToDo: Only for quick testing
            # if i_split != 4: continue
            # if i_split == 1: break
            print(f'Starting cross validation split: {cv_split}')
            data_loader_dict = dict()  # dictionary of data loaders
            ax_all = fig_all.add_subplot(spec_all[i_split])

            # %% Train and validate
            if config.TOOLBOX_MODE == "train_and_test":
                # Define train data loader
                train_data_loader = data_loader.DatasetLoader.DatasetLoader(
                    dataset_name="Train",
                    cv_split_files=cv_split_files['train'][cv_split],
                    input_signals=config.INPUT_SIGNALS,
                    label_signals=config.LABEL_SIGNALS,
                    config_data=config.TRAIN.DATA)
                data_loader_dict['train'] = DataLoader(
                    dataset=train_data_loader,
                    num_workers=16,
                    batch_size=config.TRAIN.BATCH_SIZE,
                    shuffle=True,
                    worker_init_fn=seed_worker,
                    generator=train_generator)

                # Define valid data loader
                valid_data_loader = data_loader.DatasetLoader.DatasetLoader(
                    dataset_name="Valid",
                    cv_split_files=cv_split_files['valid'][cv_split],
                    input_signals=config.INPUT_SIGNALS,
                    label_signals=config.LABEL_SIGNALS,
                    config_data=config.TRAIN.DATA)
                data_loader_dict['valid'] = DataLoader(
                    dataset=valid_data_loader,
                    num_workers=16,
                    batch_size=config.TRAIN.BATCH_SIZE,
                    shuffle=False,
                    worker_init_fn=seed_worker,
                    generator=general_generator)

            # %% Test
            if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "only_test":
                # Define test data loader
                test_data_loader = data_loader.DatasetLoader.DatasetLoader(
                    dataset_name="test",
                    cv_split_files=cv_split_files['test'][cv_split],
                    input_signals=config.INPUT_SIGNALS,
                    label_signals=config.LABEL_SIGNALS,
                    config_data=config.TEST.DATA,
                    no_labels=config.INFERENCE.NO_LABELS)
                data_loader_dict["test"] = DataLoader(
                    dataset=test_data_loader,
                    num_workers=16,
                    batch_size=config.INFERENCE.BATCH_SIZE,
                    shuffle=False,
                    worker_init_fn=seed_worker,
                    generator=general_generator)

                if config.TOOLBOX_MODE == "train_and_test" and config.TEST.USE_LAST_EPOCH:
                    print("Testing uses last epoch, validation dataset is not required.", end='\n\n')

            # Train/test model
            trainer_params = {'cv_split': cv_split, 'ax_all': ax_all, 'model_dir': model_dir,
                              'test_metrics': test_metrics, 'log_training_dir': log_training_dir,
                              'log_testing_dir': log_testing_dir}
            if config.TOOLBOX_MODE == "train_and_test":
                results = train_and_test(config, data_loader_dict, trainer_params)
            elif config.TOOLBOX_MODE == "only_test":
                results = test(config, data_loader_dict, trainer_params)
            else:
                raise ValueError("TOOLBOX_MODE only support train_and_test or only_test!")

            # Save results
            for label_signal in config.LABEL_SIGNALS:
                for task in config.TASKS_EVALUATE:
                    for metric in test_metrics[label_signal]:
                        results_seed[label_signal][task][metric].append(results[label_signal][task][metric])
            print(f'Finished cross validation split: {cv_split}')
            print('=================================================================================================\n')

        # Print results of one seed
        print('\n====================================================================================================')
        print(f'Used random seed: {random_seed}')
        print(f'Used participants: {participants}\n')
        for label_signal in config.LABEL_SIGNALS:
            print(f'Label: {label_signal}\n')
            for task in config.TASKS_EVALUATE:
                print(f'Task: {task}')
                for metric in test_metrics[label_signal]:
                    print(f'All {metric}: '
                          f'{[round(result, 2) for result in results_seed[label_signal][task][metric]]}')
                    print(f'Mean {metric} over all cv splits: '
                          f'{round(np.nanmean(results_seed[label_signal][task][metric]), 2)}')
                    print(f'STD of the {metric} over all cv splits: '
                          f'{round(np.nanstd(results_seed[label_signal][task][metric]), 2)}\n')
                    results_all_seeds_mean[label_signal][task][metric].append(
                        np.mean(results_seed[label_signal][task][metric]))
                    results_all_seeds_std[label_signal][task][metric].append(
                        np.std(results_seed[label_signal][task][metric]))

        # Write results of one seed
        with open(log_testing_dir + '/all_evaluations.txt', 'w') as f:
            f.write(f'Random seed: {random_seed}\n')
            f.write(f'Used participants: {participants}\n\n')
            for label_signal in config.LABEL_SIGNALS:
                f.write(f'Label signal: {label_signal}\n')
                for task in config.TASKS_EVALUATE:
                    f.write(f'Task: {task}\n')
                    for metric in test_metrics[label_signal]:
                        f.write(f'All {metric}: '
                                f'{[round(result, 2) for result in results_seed[label_signal][task][metric]]}\n')
                        f.write(f'Mean {metric} over all cv splits: '
                                f'{round(np.nanmean(results_seed[label_signal][task][metric]), 2)}\n')
                        f.write(f'STD of the {metric} over all cv splits: '
                                f'{round(np.nanstd(results_seed[label_signal][task][metric]), 2)}\n\n')

        # Plot results of one seed
        if 'sys' in config.LABEL_SIGNALS or 'dia' in config.LABEL_SIGNALS:
            fig_all.legend(loc='upper right', fontsize="50")
            fig_all.savefig(log_testing_dir + '/all_plots.png')

    # Print results of all seeds
    print('\n=====================================================================================================')
    print(f'Used random seeds: {random_seeds}')
    print(f'Used participants: {participants}\n')
    for label_signal in config.LABEL_SIGNALS:
        print(f'Label: {label_signal}\n')
        for task in config.TASKS_EVALUATE:
            print(f'Task: {task}')
            for metric in test_metrics[label_signal]:
                print(f'All {metric} of all seeds: '
                      f'{[round(result, 2) for result in results_all_seeds_mean[label_signal][task][metric]]}')
                print(f'Mean {metric} over all seeds: '
                      f'{round(np.mean(results_all_seeds_mean[label_signal][task][metric]), 2)}')
                print(f'STD of the {metric} over all seeds: '
                      f'{round(np.std(results_all_seeds_mean[label_signal][task][metric]), 2)}')
                print(f'Mean STD of the {metric} over all seeds: '
                      f'{round(np.mean(results_all_seeds_std[label_signal][task][metric]), 2)}')
                print(f'STD of the STD of the {metric} over all seeds: '
                      f'{round(np.std(results_all_seeds_std[label_signal][task][metric]), 2)}\n')

    # Write results of all seeds
    with open(config.LOG.PATH_TESTING + f'/all_seeds_evaluations.txt', 'w') as f:
        f.write(f'Used random seeds: {random_seeds}\n')
        f.write(f'Used participants: {participants}\n\n')
        for label_signal in config.LABEL_SIGNALS:
            f.write(f'Label signal: {label_signal}\n')
            for task in config.TASKS_EVALUATE:
                f.write(f'\nTask: {task}\n')
                for metric in test_metrics[label_signal]:
                    f.write(f'All {metric} of all seeds: '
                            f'{[round(result, 2) for result in results_all_seeds_mean[label_signal][task][metric]]}\n')
                    f.write(f'Mean {metric} over all seeds: '
                            f'{round(np.mean(results_all_seeds_mean[label_signal][task][metric]), 2)}\n')
                    f.write(f'STD of the {metric} over all seeds: '
                            f'{round(np.std(results_all_seeds_mean[label_signal][task][metric]), 2)}\n')
                    f.write(f'Mean STD of the {metric} over all seeds: '
                            f'{round(np.mean(results_all_seeds_std[label_signal][task][metric]), 2)}\n')
                    f.write(f'STD of the STD of the {metric} over all seeds: '
                            f'{round(np.std(results_all_seeds_std[label_signal][task][metric]), 2)}\n\n')

    if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
        print(f'ATTENTION: SMALLER WINDOW SIZES OF {config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE} ARE USED!')


if __name__ == "__main__":
    main()
