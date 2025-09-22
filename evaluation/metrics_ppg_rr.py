import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from tqdm import tqdm

from scipy.stats import pearsonr

from source.evaluation.BlandAltmanPy import BlandAltman
from source.evaluation.post_process import calculate_metric_per_video_ppg, calculate_metric_per_video_rr
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


def calculate_metrics_ppg_rr(predictions, labels, config, signal_name, evaluation_file_path, used_epoch, metrics,
                             i_label):
    """Calculate rPPG Metrics (MAE, RMSE, MAPE, Pearson Coef.)."""
    predict_hr_all = {task: [] for task in config.TASKS_EVALUATE}
    gt_hr_all = {task: [] for task in config.TASKS_EVALUATE}
    SNR_all = {task: [] for task in config.TASKS_EVALUATE}
    MSE_all = {task: [] for task in config.TASKS_EVALUATE}
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
        if len(config.TASKS_TO_USE) > 0:
            task_chunk_list = get_task_chunk_list(config, participant)
            task_list_flattened = []
            for i_chunk in range(len(task_chunk_list['keep'])):
                if task_chunk_list['keep'][i_chunk] == 1:
                    task_list_flattened.extend(
                        [task_chunk_list['task_names'][i_chunk]] * config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH)

        pred_part = {task: [] for task in config.TASKS_EVALUATE}
        label_part = {task: [] for task in config.TASKS_EVALUATE}
        pred_part_save = []
        hrv_part_save = []
        for i in range(0, len(prediction), window_frame_size):
            pred_window = prediction[i:i+window_frame_size]
            label_window = label[i:i+window_frame_size]

            # Ignore windows at the end that are too small
            if len(pred_window) < window_frame_size:
                if config.TEST.DATA.DATASET == 'egoppg':
                    print(f"Window frame size of {len(pred_window)} is smaller than minimum pad length of 9. Ignored!")
                    continue
            if len(pred_window) < 27:
                print(f"Window frame size of take {participant} is {len(pred_window)}, which is smaller than minimum "
                      f"pad length of 27.")
                continue

            # Get GT and predicted HR
            if config.TEST.DATA.PREPROCESS.LABEL_TYPE in ['Diff', 'DiffStandardized', 'DiffStandardizedExtended']:
                diff_flag = True
            else:
                diff_flag = False
            if signal_name == 'ppg':
                gt_hr, pred_hr, SNR, MSE, pred_hrv = calculate_metric_per_video_ppg(
                    pred_window, label_window, config.INFERENCE.EVALUATION_METHOD, diff_flag, config.TEST.DATA.FS)
                hrv_part_save.extend([pred_hrv] * pred_window.shape[0])
            elif signal_name == 'rr':
                gt_hr, pred_hr, SNR, MSE = calculate_metric_per_video_rr(
                    pred_window, label_window, config.INFERENCE.EVALUATION_METHOD, diff_flag, config.TEST.DATA.FS)
            else:
                raise ValueError("Unsupported signal name in testing!")
            gt_hr_all['overall'].append(gt_hr)
            predict_hr_all['overall'].append(pred_hr)
            SNR_all['overall'].append(SNR)
            MSE_all['overall'].append(MSE)
            pred_part['overall'].append(pred_hr)
            pred_part_save.extend([pred_hr] * pred_window.shape[0])
            label_part['overall'].append(gt_hr)

            if len(config.TASKS_TO_USE) > 0:
                task = task_list_flattened[i + window_frame_size // 2]
                gt_hr_all[task].append(gt_hr)
                predict_hr_all[task].append(pred_hr)
                SNR_all[task].append(SNR)
                MSE_all[task].append(MSE)
                pred_part[task].append(pred_hr)
                label_part[task].append(gt_hr)

        # Calculate Pearson Coefficient for each participant
        if config.TEST.DATA.DATASET == 'egoppg':
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
            ax.set_xlabel(f'Ground truth {signal_name} [bpm]')
            ax.set_ylabel(f'Predicted {signal_name} (rPPG) [bpm]')
            ax.set_title(f'{participant}: Scatter plot of participant {participant}')
            tasks_plot = list(np.unique(task_categories))
            legend_handles = [mpatches.Patch(color=colors[task], label=task) for task in tasks_plot]
            ax.legend(handles=legend_handles, loc='upper left')
            fig.show()
        else:
            if np.asarray(pred_part['overall']).shape[0] > 1:
                pearson_all['overall'].append(pearsonr(label_part['overall'], pred_part['overall'])[0])

        # Save and plot predicted HR for each participant over time
        if config.TEST.DATA.DATASET == 'egoexo4d':
            if signal_name == 'ppg':
                feature_name, wave_name = 'hrs', 'rppgs'
                # If dir does not exist, create it
                if not os.path.exists(f'/local/home/bjbraun/Projects/egoPPG/predicted_{feature_name}/{config.MODEL.NAME}/egoexo4d'):
                    os.makedirs(f'/local/home/bjbraun/Projects/egoPPG/predicted_{feature_name}/{config.MODEL.NAME}/egoexo4d')
                if not os.path.exists(f'/local/home/bjbraun/Projects/egoPPG/predicted_hrvs/{config.MODEL.NAME}/egoexo4d'):
                    os.makedirs(f'/local/home/bjbraun/Projects/egoPPG/predicted_hrvs/{config.MODEL.NAME}/egoexo4d')
                if not os.path.exists(f'/local/home/bjbraun/Projects/egoPPG/predicted_{wave_name}/{config.MODEL.NAME}/egoexo4d'):
                    os.makedirs(f'/local/home/bjbraun/Projects/egoPPG/predicted_{wave_name}/{config.MODEL.NAME}/egoexo4d')
                np.save(
                    f'/local/home/bjbraun/Projects/egoPPG/predicted_{feature_name}/{config.MODEL.NAME}/egoexo4d/{participant}_{feature_name}.npy',
                    np.asarray(resample_signal(pred_part_save, prediction.shape[0], 'linear')))
                np.save(
                    f'/local/home/bjbraun/Projects/egoPPG/predicted_hrvs/{config.MODEL.NAME}/egoexo4d/{participant}_hrvs.npy',
                    np.asarray(resample_signal(hrv_part_save, prediction.shape[0], 'linear')))
                np.save(f'/local/home/bjbraun/Projects/egoPPG/predicted_{wave_name}/{config.MODEL.NAME}/egoexo4d/{participant}_{wave_name}.npy',
                        np.asarray(prediction))
            elif signal_name == 'rr':
                feature_name, wave_name = 'rrs', 'rrwaves'
                np.save(
                    f'/local/home/bjbraun/Projects/egoPPG/predicted_{feature_name}/{config.MODEL.NAME}/egoexo4d/{participant}_{feature_name}.npy',
                    np.asarray(resample_signal(pred_part_save, prediction.shape[0], 'linear')))
                np.save(f'/local/home/bjbraun/Projects/egoPPG/predicted_{wave_name}/{config.MODEL.NAME}/egoexo4d/{participant}_{wave_name}.npy',
                        np.asarray(prediction))
            else:
                raise ValueError("Unsupported signal name in testing!")

            """fig, ax = plt.subplots()
            ax.plot(pred_part_save, label=f'Predicted {feature_name}')
            ax.set_title(f'{participant}: Predicted {feature_name}')
            fig.show()"""

    # Calculate different metrics from predicted HR
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
            gt_hr_task = np.array(gt_hr_all[task])
            predict_hr_task = np.array(predict_hr_all[task])
            SNR_task = np.array(SNR_all[task])
            MSE_task = np.array(MSE_all[task])
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
                elif metric == "SNR":
                    result = np.nanmean(SNR_task)
                elif metric == "MSE":
                    result = np.nanmean(MSE_task)
                else:
                    raise ValueError("Wrong Test Metric Type")

                metrics_out[task][metric] = result
                metrics_out[task]['MAE_all'] = [np.abs(predict_hr_task[i] - gt_hr_task[i]) for i in
                                                range(len(predict_hr_task))]
                if metric == 'Pearson':
                    standard_error = np.sqrt((1 - result ** 2) / (n_samples_task - 2))
                else:
                    standard_error = np.std(np.abs(predict_hr_task - gt_hr_task)) / np.sqrt(n_samples_task)
                print(f"{metric} ({config.INFERENCE.EVALUATION_METHOD}): "
                      f"{round(result, 2)} +/- {round(standard_error, 2)}")
                f.write(f"\n{metric} ({config.INFERENCE.EVALUATION_METHOD}): "
                        f"{round(result, 2)} +/- {round(standard_error, 2)}")

    # metrics_out['MAE_all'] = [np.abs(predict_hr_all['overall'][i] - gt_hr_all['overall'][i]) for i in
    #                           range(len(predict_hr_all['overall']))]
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