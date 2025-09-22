import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from source.evaluation.metrics_ppg_rr import _reform_data_from_dict
from source.evaluation.post_process import calculate_metric_per_video_bp
from tqdm import tqdm


def plot_all_signals_bp(prediction, label, participant, fs, ax):
    x = np.linspace(0, len(prediction) / fs, len(prediction))
    t = pd.to_datetime(x, unit='s')
    ax.plot(t, label, color='black', label='Label')
    ax.plot(t, prediction, color='orange', label='Prediction')
    ax.set_title(f'{participant}')
    ax.set_xlabel('Time [min]')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%M'))


def calculate_metrics_bp(predictions, labels, config, evaluation_file_path, used_epoch, fig_all, spec_all, metrics,
                         i_label):
    SC_all = list()
    MAE_all = list()

    for index in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[index])
        label = _reform_data_from_dict(labels[index])

        if config.TEST.DATA.PREPROCESS.LABEL_TYPE in ["Raw", "Standardized"]:
            diff_flag_test = False
        elif config.TEST.DATA.PREPROCESS.LABEL_TYPE in ["Diff", "DiffStandardized"]:
            diff_flag_test = True
        else:
            raise ValueError("Unsupported label type in testing!")

        np.save(evaluation_file_path.replace('Evaluation.txt', f'results.npy'), np.asarray([prediction.numpy(),
                                                                                            label.numpy()]))

        # Plot results of entire signal without windowing
        _, _, prediction_full, label_full = calculate_metric_per_video_bp(
            np.copy(prediction), np.copy(label), config, do_norm=False, diff_flag=diff_flag_test,
            fs=config.TEST.DATA.FS, use_highpass=False)
        ax_all = fig_all.add_subplot(spec_all[len(fig_all.axes)])
        plot_all_signals_bp(prediction_full, label_full, index, config.TEST.DATA.FS, ax_all)

        # Get windows
        video_frame_size = prediction.shape[0]
        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > video_frame_size:
                window_frame_size = video_frame_size
        else:
            window_frame_size = video_frame_size

        # Calculate MAE and SC for each window
        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            SC_temp, MAE_temp, _, _ = calculate_metric_per_video_bp(
                np.asarray(pred_window), np.asarray(label_window), config, do_norm=False, diff_flag=diff_flag_test,
                fs=config.TEST.DATA.FS, use_highpass=False)
            if not np.isnan(SC_temp):
                SC_all.append(SC_temp)
            else:
                raise ValueError(f"\nParticipant: {index}, NaN value omitted!")
            SC_all.append(SC_temp)
            MAE_all.append(MAE_temp)

    if i_label == 0:
        write_mode = 'w'
    else:
        write_mode = 'a'
    with open(evaluation_file_path, write_mode) as f:
        f.write(f'Used Epoch: {used_epoch}')
        SC_all = np.array(SC_all)
        MAE_all = np.array(MAE_all)
        num_test_samples = len(SC_all)
        metrics_out = {metric: None for metric in metrics}
        for metric in metrics:
            if metric == "SC":
                mean_SPC = np.mean(SC_all)
                metrics_out["SC"] = mean_SPC
                standard_error = np.std(SC_all) / np.sqrt(num_test_samples)
                print("Mean spearman correlation: {0} +/- {1}".format(mean_SPC, standard_error))
                f.write("\nMean spearman correlation: {0} +/- {1}".format(mean_SPC, standard_error))
            elif metric == "MAE":
                mean_MAE = np.mean(MAE_all)
                metrics_out["MAE"] = mean_MAE
                standard_error = np.std(MAE_all) / np.sqrt(num_test_samples)
                print("Mean absolute error: {0} +/- {1}".format(mean_MAE, standard_error))
                f.write("\nMean absolute error: {0} +/- {1}".format(mean_MAE, standard_error))
            else:
                raise ValueError("Wrong Test Metric Type")

    return metrics_out
