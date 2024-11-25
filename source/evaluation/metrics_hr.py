import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from scipy.stats import pearsonr

from source.evaluation.BlandAltmanPy import BlandAltman
from source.evaluation.post_process import calculate_metric_per_video_ppg
from source.utils import get_task_chunk_list, resample_signal


def _reform_data_from_dict(data, flatten=True):
    """Helper func for calculate metrics: reformat predictions and labels from dicts. """
    sort_data = sorted(data.items(), key=lambda x: x[0])
    sort_data = [i[1] for i in sort_data]
    sort_data = torch.cat(sort_data, dim=0)

    if flatten:
        sort_data = np.reshape(sort_data.cpu(), (-1))
    else:
        sort_data = np.array(sort_data.cpu())

    return sort_data


def calculate_metrics_hr(predictions, labels, config, evaluation_file_path, used_epoch, metrics):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_all = {task: [] for task in config.TASKS_EVALUATE}
    gt_hr_all = {task: [] for task in config.TASKS_EVALUATE}
    pearson_all = {task: [] for task in config.TASKS_EVALUATE}
    for participant in tqdm(predictions.keys(), ncols=80):
        prediction = _reform_data_from_dict(predictions[participant])
        label = _reform_data_from_dict(labels[participant])
        np.save(evaluation_file_path.replace('Evaluation.txt', f'results.npy'), np.asarray([prediction, label]))

        if config.INFERENCE.EVALUATION_WINDOW.USE_SMALLER_WINDOW:
            window_frame_size = config.INFERENCE.EVALUATION_WINDOW.WINDOW_SIZE * config.TEST.DATA.FS
            if window_frame_size > prediction.shape[0]:
                window_frame_size = prediction.shape[0]
        else:
            window_frame_size = prediction.shape[0]

        # Get task list for each participant
        task_chunk_list = get_task_chunk_list(config, participant)
        task_list_flattened = []
        for i_chunk in range(len(task_chunk_list['keep'])):
            if task_chunk_list['keep'][i_chunk] == 1:
                task_list_flattened.extend(
                    [task_chunk_list['task_names'][i_chunk]] * config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH)

        pred_part = {task: [] for task in config.TASKS_EVALUATE}
        label_part = {task: [] for task in config.TASKS_EVALUATE}
        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            # Ignore windows at the end that are too small
            if len(pred_window) < window_frame_size:
                print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Ignored!")
                continue

            pred_hr = np.mean(np.asarray(pred_window))
            gt_hr = np.mean(np.asarray(label_window))
            gt_hr_all['overall'].append(gt_hr)
            predict_hr_all['overall'].append(pred_hr)
            pred_part['overall'].append(pred_hr)
            label_part['overall'].append(gt_hr)

            if len(config.TASKS_TO_USE) > 0:
                task = task_list_flattened[i + window_frame_size // 2]
                gt_hr_all[task].append(gt_hr)
                predict_hr_all[task].append(pred_hr)
                pred_part[task].append(pred_hr)
                label_part[task].append(gt_hr)

        # Calculate Pearson Coefficient for each participant
        hrs_pred_plot = []
        hrs_gt_plot = []
        task_categories = []
        for task in config.TASKS_EVALUATE:
            pred_part_temp = np.asarray(pred_part[task])
            label_part_temp = np.asarray(label_part[task])
            pred_part_temp = pred_part_temp[~np.isnan(pred_part_temp)]
            label_part_temp = label_part_temp[~np.isnan(pred_part_temp)]
            if task != 'overall':
                hrs_pred_plot.extend(pred_part_temp)
                hrs_gt_plot.extend(label_part_temp)
                task_categories.extend([task] * len(pred_part_temp))
            if pred_part_temp.shape[0] > 1:
                pearson_all[task].append(pearsonr(label_part_temp, pred_part_temp)[0])

        fig, ax = plt.subplots()
        colors = {'video': 'orange', 'office': 'purple', 'kitchen': 'yellow', 'dancing': 'blue',
                  'bike': 'red', 'walking': 'black', 'running': 'green'}
        color_list = [colors[group] for group in task_categories]
        ax.scatter(hrs_gt_plot, hrs_pred_plot, c=color_list, lw=0.5)
        ax.set_xlabel('Ground truth HR [bpm]')
        ax.set_ylabel('Predicted HR (rPPG) [bpm]')
        ax.set_title(f'{participant}: Scatter plot of participant {participant}')
        tasks_plot = list(np.unique(task_categories))
        legend_handles = [mpatches.Patch(color=colors[task], label=task) for task in tasks_plot]
        ax.legend(handles=legend_handles, loc='upper left')
        fig.show()

    # Calculate different metrics from predicted HR
    metrics_out = {task: {metric: None for metric in metrics} for task in config.TASKS_EVALUATE}
    with open(evaluation_file_path, 'w') as f:
        f.write(f'Used Epoch: {used_epoch}')
        print(f'Used Epoch: {used_epoch}')
        for task in config.TASKS_EVALUATE:
            f.write(f'\nTask: {task}')
            print(f'\nTask: {task}')
            gt_hr_task = np.array(gt_hr_all[task])
            predict_hr_task = np.array(predict_hr_all[task])
            n_samples_task = len(predict_hr_task)
            for metric in metrics:
                if metric == "MAE":
                    result = np.nanmean(np.abs(predict_hr_task - gt_hr_task))
                elif metric == "RMSE":
                    result = np.sqrt(np.nanmean(np.square(predict_hr_task - gt_hr_task)))
                elif metric == "MAPE":
                    result = np.nanmean(np.abs((predict_hr_task - gt_hr_task) / gt_hr_task)) * 100
                elif metric == "Pearson":
                    # For mean Pearson Coefficient over all predictions
                    # Pearson = np.corrcoef(predict_hr_task, gt_hr_task)
                    # result = Pearson[0][1]

                    # For mean Pearson Coefficient of all participants. Only calculated for overall task
                    result = np.nanmean(pearson_all[task])
                else:
                    raise ValueError("Wrong Test Metric Type")

                metrics_out[task][metric] = result
                if metric == 'Pearson':
                    standard_error = np.sqrt((1 - result ** 2) / (n_samples_task - 2))
                else:
                    standard_error = np.std(np.abs(predict_hr_task - gt_hr_task)) / np.sqrt(n_samples_task)
                print(f"{metric} ({config.INFERENCE.EVALUATION_METHOD}): "
                      f"{round(result, 2)} +/- {round(standard_error, 2)}")
                f.write(f"\n{metric} ({config.INFERENCE.EVALUATION_METHOD}): "
                        f"{round(result, 2)} +/- {round(standard_error, 2)}")

    return metrics_out


# ToDo: Implement Bland-Altman for PPG
"""if "AU" in metric:
    pass
elif "BA" in metric:
    filename_id = 'TEST'
    compare = BlandAltman(gt_hr_all, predict_hr_all, config, averaged=True)
    compare.scatter_plot(
        x_label='GT PPG HR [bpm]',
        y_label='rPPG HR [bpm]',
        show_legend=True, figure_size=(5, 5),
        the_title=f'{filename_id}_FFT_BlandAltman_ScatterPlot',
        file_name=f'{filename_id}_FFT_BlandAltman_ScatterPlot.pdf')
    compare.difference_plot(
        x_label='Difference between rPPG HR and GT PPG HR [bpm]',
        y_label='Average of rPPG HR and GT PPG HR [bpm]',
        show_legend=True, figure_size=(5, 5),
        the_title=f'{filename_id}_FFT_BlandAltman_DifferencePlot',
        file_name=f'{filename_id}_FFT_BlandAltman_DifferencePlot.pdf')"""