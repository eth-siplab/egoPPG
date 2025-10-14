import argparse
import json
import numpy as np
import os
import random
import torch

from ml import data_loader, trainer, ml_helper
from ml.config import get_config
from utils import get_participants_lists
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
    if config.MODEL.NAME == 'PhysNet':
        model_trainer = trainer.PhysNetTrainer.PhysNetTrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'PhysNetSA':
        model_trainer = trainer.PhysNetSATrainer.PhysNetSATrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'PulseFormer':
        model_trainer = trainer.PulseFormerTrainer.PulseFormerTrainer(config, data_loader_dict, trainer_params)
    else:
        raise ValueError('Your Model is Not Supported Yet!')
    model_trainer.train(data_loader_dict)
    return model_trainer.test(data_loader_dict)


def test(config, data_loader_dict, trainer_params):
    if config.MODEL.NAME == 'PhysNet':
        model_trainer = trainer.PhysNetTrainer.PhysNetTrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'PhysNetSA':
        model_trainer = trainer.PhysNetSATrainer.PhysNetSATrainer(config, data_loader_dict, trainer_params)
    elif config.MODEL.NAME == 'PulseFormer':
        model_trainer = trainer.PulseFormerTrainer.PulseFormerTrainer(config, data_loader_dict, trainer_params)
    else:
        raise ValueError('Your Model is Not Supported Yet!')
    return model_trainer.test(data_loader_dict)


def main():
    # Data input & output
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', type=str, help='Name of the configuration file')
    args = parser.parse_args()
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))  # pretty print args

    # Load configs
    config = get_config(args)
    print('Configuration:')
    print(config, end='\n\n')
    participants, participants_test = get_participants_lists(config)

    # Define random seeds to evaluate
    random_seeds = [0, 10, 100]
    # random_seeds = [0]

    # Define metrics and tasks to evaluate
    test_metrics = {'ppg_nose': ['MAE', 'RMSE', 'MAPE', 'Pearson'],
                    'hr': ['MAE', 'RMSE', 'MAPE', 'Pearson'],
                    'classhr': ['MAE', 'RMSE', 'MAPE', 'Pearson']}

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
        cv_split_files = ml_helper.get_cv_split_files(config, participants, participants_test, random_seed)

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
        spec_all = gridspec.GridSpec(ncols=5, nrows=5, figure=fig_all)

        # Loop through cross-validation splits
        results_seed = {label_signal: {task: {metric: list() for metric in test_metrics[label_signal]+['MAE_all']}
                                       for task in config.TASKS_EVALUATE} for label_signal in config.LABEL_SIGNALS}
        for i_split, cv_split in enumerate(cv_split_files['train'].keys()):
            print(f'Starting cross validation split: {cv_split}')
            data_loader_dict = dict()  # dictionary of data loaders

            # %% Train and validate
            if config.TOOLBOX_MODE == "train_and_test":
                # Define train data loader
                train_loader = data_loader.DatasetLoader.DatasetLoader
                train_data_loader = train_loader(
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
                valid_loader = data_loader.DatasetLoader.DatasetLoader
                valid_data_loader = valid_loader(
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
                test_loader = data_loader.DatasetLoader.DatasetLoader
                if config.TRAIN.DATA.DATASET == config.TEST.DATA.DATASET:
                    input_signals, label_signals = config.INPUT_SIGNALS, config.LABEL_SIGNALS
                else:
                    input_signals, label_signals = config.INPUT_SIGNALS_TEST, config.LABEL_SIGNALS_TEST
                test_data_loader = test_loader(
                    dataset_name="test",
                    cv_split_files=cv_split_files['test'][cv_split],
                    input_signals=input_signals,
                    label_signals=label_signals,
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
            trainer_params = {'cv_split': cv_split, 'fig_all': fig_all, 'spec_all': spec_all, 'model_dir': model_dir,
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
                    if label_signal in ['ppg_nose', 'ppg_ear']:
                        results_seed[label_signal][task]['MAE_all'].append(results[label_signal][task]['MAE_all'])
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
                        np.nanmean(results_seed[label_signal][task][metric]))
                    results_all_seeds_std[label_signal][task][metric].append(
                        np.nanstd(results_seed[label_signal][task][metric]))

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
        if ('sys' in config.LABEL_SIGNALS or 'dia' in config.LABEL_SIGNALS or 'eda_filtered' in config.LABEL_SIGNALS or
                'eda_raw' in config.LABEL_SIGNALS or 'eda_tonic' in config.LABEL_SIGNALS):
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
                      f'{round(np.nanmean(results_all_seeds_mean[label_signal][task][metric]), 2)}')
                print(f'STD of the {metric} over all seeds: '
                      f'{round(np.nanstd(results_all_seeds_mean[label_signal][task][metric]), 2)}')
                print(f'Mean STD of the {metric} over all seeds: '
                      f'{round(np.nanmean(results_all_seeds_std[label_signal][task][metric]), 2)}')
                print(f'STD of the STD of the {metric} over all seeds: '
                      f'{round(np.nanstd(results_all_seeds_std[label_signal][task][metric]), 2)}\n')

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
                            f'{round(np.nanmean(results_all_seeds_mean[label_signal][task][metric]), 2)}\n')
                    f.write(f'STD of the {metric} over all seeds: '
                            f'{round(np.nanstd(results_all_seeds_mean[label_signal][task][metric]), 2)}\n')
                    f.write(f'Mean STD of the {metric} over all seeds: '
                            f'{round(np.nanmean(results_all_seeds_std[label_signal][task][metric]), 2)}\n')
                    f.write(f'STD of the STD of the {metric} over all seeds: '
                            f'{round(np.nanstd(results_all_seeds_std[label_signal][task][metric]), 2)}\n\n')

    if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
        print(f'ATTENTION: SMALLER WINDOW SIZES OF {config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE} ARE USED!')


if __name__ == "__main__":
    main()
