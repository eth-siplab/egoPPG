import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from matplotlib.ticker import ScalarFormatter, MaxNLocator

from source.evaluation.metrics_bp import calculate_metrics_bp
from source.evaluation.metrics_eda import calculate_metrics_eda
from source.evaluation.metrics_hr import calculate_metrics_hr
from source.evaluation.metrics_ppg_rr import calculate_metrics_ppg_rr


class BaseTrainer:
    @staticmethod
    def add_trainer_args(parser):
        """Adds arguments to Parser for training process"""
        parser.add_argument('--lr', default=None, type=float)
        parser.add_argument('--model_file_name', default=None, type=float)
        return parser

    def __init__(self):
        pass

    def train(self, data_loader):
        pass

    def valid(self, data_loader):
        pass

    def test(self, data_loader):
        pass

    @staticmethod
    def get_model_path(config, model_dir, log_training_dir, cv_split, max_epoch_num, best_epoch):
        if config.TOOLBOX_MODE == "only_test":
            if config.INFERENCE.USE_BEST_EPOCH:
                best_epochs_path = log_training_dir + f'/cv{cv_split}_best_epoch.npy'
                used_epoch = int(np.load(best_epochs_path))
                print("Testing uses best epoch selected during validation as pretrained model!")
            else:
                used_epoch = config.INFERENCE.MODEL_PATH.split(".")[0]
                try:
                    used_epoch = int(used_epoch[-2:])
                except ValueError:
                    used_epoch = int(used_epoch[-1:])
                print("Testing uses model from specified epoch!")
        else:
            if config.TEST.USE_LAST_EPOCH:
                used_epoch = max_epoch_num - 1
                print("Testing uses last epoch as pretrained model!")
            else:
                used_epoch = best_epoch
                print("Testing uses best epoch selected during validation as pretrained model!")

        model_path = model_dir + f'/cv{cv_split}_Epoch{str(used_epoch)}.pth'
        if not os.path.exists(model_path):
            raise ValueError("Inference model path does not exist! Please check inference_model_path.")
        print(f'Used epoch: {used_epoch}\n')
        return model_path, used_epoch

    @staticmethod
    def evaluate_predictions(predictions, labels, config, log_testing_dir, cv_split, used_epoch, test_metrics,
                             fig_all, spec_all):
        results = {label_signal: None for label_signal in config.LABEL_SIGNALS}
        evaluation_file_path = log_testing_dir + f'/cv{cv_split}_Evaluation.txt'
        for i_label, label_signal in enumerate(config.LABEL_SIGNALS):
            if label_signal in ['ppg_finger', 'ppg_ear', 'ppg_nose', 'sineppg', 'ppg']:
                results[label_signal] = calculate_metrics_ppg_rr(predictions[label_signal], labels[label_signal],
                                                                 config, 'ppg', evaluation_file_path, used_epoch,
                                                                 test_metrics[label_signal], i_label)
            elif label_signal in ['rr', 'classrr']:
                results[label_signal] = calculate_metrics_ppg_rr(predictions[label_signal], labels[label_signal],
                                                                 config, 'rr', evaluation_file_path, used_epoch,
                                                                 test_metrics[label_signal], i_label)
            elif label_signal in ['hr', 'classhr']:
                results[label_signal] = calculate_metrics_hr(predictions[label_signal], labels[label_signal], config,
                                                             evaluation_file_path, used_epoch,
                                                             test_metrics[label_signal], i_label)
            elif label_signal in ['sys', 'dia']:
                results[label_signal] = calculate_metrics_bp(predictions[label_signal], labels[label_signal], config,
                                                             evaluation_file_path, used_epoch, fig_all, spec_all,
                                                             test_metrics[label_signal], i_label)
            elif label_signal in ['eda_raw', 'eda_filtered', 'eda_tonic']:
                results[label_signal] = calculate_metrics_eda(predictions[label_signal], labels[label_signal], config,
                                                              evaluation_file_path, used_epoch,
                                                              test_metrics[label_signal], fig_all, spec_all, i_label)
            else:
                raise ValueError("Label signal error! Please check label signal.")
        return results

    @staticmethod
    def save_model(save_dir, model, cv_split, index):
        torch.save(model.state_dict(), save_dir + f'/cv{cv_split}_Epoch{str(index)}.pth')

    @staticmethod
    def save_test_outputs(predictions, labels, config):
        output_dir = config.TEST.OUTPUT_SAVE_DIR
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        data = {'predictions': predictions, 'labels': labels, 'label_type': config.TEST.DATA.PREPROCESS.LABEL_TYPE,
                'fs': config.TEST.DATA.FS}
        with open(output_dir + '/outputs.pickle', 'wb') as handle:  # save out frame dict pickle file
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saving outputs to:', output_dir + '/outputs.pickle')

    @staticmethod
    def plot_losses_and_lrs(train_loss, valid_loss, lrs, config, log_training_dir, cv_split):

        # Filename ID to be used in plots that get saved
        if not config.TOOLBOX_MODE == 'train_and_test':
            raise ValueError('Metrics.py evaluation only supports train_and_test and only_test!')
        
        # Create a single plot for training and validation losses
        plt.figure(figsize=(10, 6))
        epochs = range(0, len(train_loss))  # Integer values for x-axis
        plt.plot(epochs, train_loss, label='Training Loss')
        if len(valid_loss) > 0:
            plt.plot(epochs, valid_loss, label='Validation Loss')
        else:
            print("The list of validation losses is empty. The validation loss will not be plotted!")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Losses')
        plt.legend()
        plt.xticks(epochs)

        # Set y-axis ticks with more granularity
        ax = plt.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=False, prune='both'))

        loss_plot_filename = os.path.join(log_training_dir, f'cv{cv_split}_Losses.pdf')
        plt.savefig(loss_plot_filename, dpi=300)
        plt.close()

        # Create a separate plot for learning rates
        plt.figure(figsize=(6, 4))
        scheduler_steps = range(0, len(lrs))
        plt.plot(scheduler_steps, lrs, label='Learning Rate')
        plt.xlabel('Scheduler Step')
        plt.ylabel('Learning Rate')
        plt.title(f'LR Schedule')
        plt.legend()

        # Set y-axis values in scientific notation
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=False))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))  # Force scientific notation

        lr_plot_filename = os.path.join(log_training_dir, f'cv{cv_split}_Learning_rates.pdf')
        plt.savefig(lr_plot_filename, bbox_inches='tight', dpi=300)
        plt.close()

        print('Saving plots of losses and learning rates to:', log_training_dir)


