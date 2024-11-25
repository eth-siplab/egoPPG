import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import pearsonr


def evaluation_single_part(participant, mae, hrs, task, task_categories, methods, method_names, tasks_evaluate):
    rounding = 2
    for method in methods:
        print(f'{participant}, {task}: Mean MAE+-STD with {method_names[method]} over all windows: '
              f'{round(np.mean(mae[method]), rounding)} +- {round(np.std(mae[method]), rounding)}')
        if method == 'pa':
            continue
        pearson_temp, _ = pearsonr(hrs[method], hrs['ecg'])
        print(f'{participant}, {task}: Pearson with {method_names[method]} over all windows: '
              f'{round(pearson_temp, rounding)}')
        if method[-3:] == 'fft':
            print('')
    print(f'Finished participant {participant}!')
    print('\n=======================================================================================================\n')

    plot_signals = True
    if plot_signals:
        # Plot MAE over time
        """fig, ax = plt.subplots()
        ax.plot(mae['rppg_peaks'], label='rPPG Peaks')
        ax.plot(mae['ppg_md_peaks'], label='PPG MD Peaks')
        ax.set_xlabel('Window')
        ax.set_ylabel('MAE [bpm]')
        ax.set_title(f'{participant}, {task}: MAE over time of participant {participant}')
        ax.legend()
        fig.show()

        # Plot HRs over time
        fig, ax = plt.subplots()
        ax.plot(hrs['ecg'], label='ECG')
        ax.plot(hrs['rppg_peaks'], label='rPPG Peaks')
        # ax.plot(hrs['rppg_fft'], label='rPPG FFT')
        ax.plot(hrs['ppg_md_peaks'], label='PPG MD Peaks')
        ax.set_xlabel('Window')
        ax.set_ylabel('HR [bpm]')
        ax.set_title(f'{participant}, {task}: HRs over time of participant {participant}')
        ax.legend()
        fig.show()"""

        # Plot scatter plot of predicted vs GT HR
        fig, ax = plt.subplots()
        colors = {'video': 'orange', 'office': 'purple', 'kitchen': 'yellow', 'dancing': 'blue',
                  'bike': 'red', 'walking': 'black', 'running': 'green'}
        color_list = [colors[group] for group in task_categories]
        ax.scatter(hrs['ecg'], hrs['rppg_peaks'], c=color_list, lw=0.5)
        ax.set_xlabel('Ground truth HR [bpm]')
        ax.set_ylabel('Predicted HR (rPPG) [bpm]')
        ax.set_title(f'{participant}, {task}: Scatter plot of participant {participant}')
        tasks_plot = tasks_evaluate.copy()
        tasks_plot.remove('all')
        legend_handles = [mpatches.Patch(color=colors[task], label=task) for task in tasks_plot]
        ax.legend(handles=legend_handles, loc='upper left')
        fig.show()
