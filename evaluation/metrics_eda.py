import matplotlib.dates as mdates
import pandas as pd

from source.evaluation.metrics_ppg_rr import _reform_data_from_dict
from source.evaluation.post_process import *
from source.utils import get_task_chunk_list, resample_signal
from tqdm import tqdm


def plot_all_signals_eda(prediction, label, participant, fs, ax):
    x = np.linspace(0, len(prediction) / fs, len(prediction))
    t = pd.to_datetime(x, unit='s')
    ax.plot(t, label, color='black', label='Label')
    ax.plot(t, prediction, color='orange', label='Prediction')
    ax.set_title(f'{participant}')
    ax.set_xlabel('Time [min]')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%M'))


def calculate_metrics_eda(predictions, labels, config, evaluation_file_path, used_epoch, metrics, fig_all, spec_all,
                          i_label):

    SPC_ALL = {task: [] for task in config.TASKS_EVALUATE}
    PC_ALL = {task: [] for task in config.TASKS_EVALUATE}
    MAE_all = {task: [] for task in config.TASKS_EVALUATE}
    for i_part, participant in enumerate(tqdm(predictions.keys(), ncols=80)):
        prediction = _reform_data_from_dict(predictions[participant])
        label = _reform_data_from_dict(labels[participant])
        np.save(evaluation_file_path.replace('Evaluation.txt', f'results.npy'), np.asarray([prediction, label]))

        # Get GT and predicted HR
        if config.TEST.DATA.PREPROCESS.LABEL_TYPE in ['Diff', 'DiffStandardized', 'DiffStandardizedExtended']:
            diff_flag = True
        else:
            diff_flag = False

        # Get overall results and plot them
        SPC, PC, MAE, prediction_full, label_full = calculate_metric_per_video_eda(
            np.copy(prediction), np.copy(label), config,  diff_flag=diff_flag, fs=config.TEST.DATA.FS)
        SPC_ALL['overall'].append(SPC)
        PC_ALL['overall'].append(PC)
        MAE_all['overall'].append(MAE)
        ax_all = fig_all.add_subplot(spec_all[len(fig_all.axes)])
        plot_all_signals_eda(prediction_full, label_full, participant, config.TEST.DATA.FS, ax_all)

        # Get task list for each participant
        task_chunk_list = get_task_chunk_list(config, participant)
        task_list_flattened = []
        for i_chunk in range(len(task_chunk_list['keep'])):
            if task_chunk_list['keep'][i_chunk] == 1:
                task_list_flattened.extend(
                    [task_chunk_list['task_names'][i_chunk]] * config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH)
        task_list_flattened = np.array(task_list_flattened)
        change_indices = np.where(task_list_flattened[:-1] != task_list_flattened[1:])[0] + 1
        change_indices = np.insert(change_indices, 0, 0)
        change_indices = np.append(change_indices, len(task_list_flattened))
        change_classes = task_list_flattened[change_indices[:-1]]

        # Calculate metrics for each task
        for i in range(len(change_indices) - 1):
            pred_task = prediction[change_indices[i]:change_indices[i + 1]]
            label_task = label[change_indices[i]:change_indices[i + 1]]

            if len(pred_task) < 9:
                print(f"Window frame size of {len(pred_task)} is smaller than minimum pad length of 9. Window ignored!")
                continue

            SPC_task, PC_task, MAE_task, pred_task, label_task = calculate_metric_per_video_eda(
                pred_task, label_task, config, diff_flag=diff_flag, fs=config.TEST.DATA.FS)
            if not np.isnan(SPC_task):
                SPC_ALL[change_classes[i]].append(SPC_task)
            else:
                print(f"\nParticipant: {participant}, NaN value omitted for SPC!")
            if not np.isnan(PC_task):
                PC_ALL[change_classes[i]].append(PC_task)
            else:
                print(f"\nParticipant: {participant}, NaN value omitted for PC!")
            MAE_all[change_classes[i]].append(MAE_task)

    if i_label == 0:
        write_mode = 'w'
    else:
        write_mode = 'a'
    metrics_out = {task: {metric: None for metric in metrics} for task in config.TASKS_EVALUATE}
    with open(evaluation_file_path, write_mode) as f:
        f.write(f'Used Epoch: {used_epoch}')
        print(f'Used Epoch: {used_epoch}')
        for task in config.TASKS_EVALUATE:
            f.write(f'\nTask: {task}')
            print(f'\nTask: {task}')
            for metric in metrics:
                if metric == "SPC":
                    result = np.nanmean(SPC_ALL[task])
                    standard_error = np.std(SPC_ALL[task]) / np.sqrt(len(SPC_ALL[task]))
                elif metric == "PC":
                    result = np.nanmean(PC_ALL[task])
                    standard_error = np.std(PC_ALL[task]) / np.sqrt(len(PC_ALL[task]))
                elif metric == "MAE":
                    result = np.nanmean(MAE_all[task])
                    standard_error = np.std(MAE_all[task]) / np.sqrt(len(MAE_all[task]))
                else:
                    raise ValueError("Wrong Test Metric Type")
                metrics_out[task][metric] = result
                print(f"{metric}: {round(result, 2)} +/- {round(standard_error, 2)}")
                f.write(f"\n{metric}: {round(result, 2)} +/- {round(standard_error, 2)}")

    return metrics_out
