import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
import yaml


def plot_eda_signal(fig, spec_all, i, data_plots, participant, configs_pre, signals):
    ax = fig.add_subplot(spec_all[i])
    x = np.linspace(0, len(data_plots[0]) / configs_pre['fs_all']['et'], len(data_plots[0]))
    t = pd.to_datetime(x, unit='s')
    for data in data_plots:
        ax.plot(t, data, label=f'Participant {participant}')
    ax.set_xlabel('Time [min]')
    start = int(configs_pre['task_times'][participant]['video'][0])
    for task in configs_pre['task_times'][participant]:
        task_lines = configs_pre['task_times'][participant][task]
        ax.axvline(x=pd.to_datetime((int(task_lines[0]) - start) / configs_pre['fs_all']['et'], unit='s'),
                   color='black',
                   linestyle='--')
        start_time = pd.to_datetime((int(task_lines[0]) - start) / configs_pre['fs_all']['et'], unit='s')
        end_time = pd.to_datetime((int(task_lines[1]) - start) / configs_pre['fs_all']['et'], unit='s')
        middle = start_time + (end_time - start_time) / 2
        ymin, ymax = ax.get_ylim()
        y_position = ymin + 0.05 * (ymax - ymin)
        ax.text(x=middle, y=y_position, s=f'{task}', rotation=0, verticalalignment='bottom',
                horizontalalignment='center')
    ax.axvline(x=pd.to_datetime((int(configs_pre['task_times'][participant]['running'][1]) - start) /
                                configs_pre['fs_all']['et'], unit='s'), color='black', linestyle='--')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%M'))
    ax.set_title(f'Participant {participant}')


def main():
    # Participants: Exclude 008 (low EDA)
    participants = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013',
                    '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025']

    # Load configuration files
    cfg_path = f'./configs/preprocessing/config_preprocessing_extended_egoppg.yml'
    with open(cfg_path, 'r') as yamlfile:
        configs = yaml.load(yamlfile, Loader=yaml.FullLoader)
    cfg_path = f'./configs/preprocessing/config_preprocessing_egoppg.yml'
    with open(cfg_path, 'r') as yamlfile:
        configs_pre = yaml.load(yamlfile, Loader=yaml.FullLoader)

    # Get folder where to load data from
    configuration_old = f'CL{configs["clip_length_old"]}_W{configs["w"]}_H{configs["h"]}_LabelRaw_VideoTypeRaw'
    data_path = configs['dir_preprocessed'] + f'/Data_ML/{configuration_old}/Data'

    # Loop through all participants, load data and create one big plot with the ..._label_eda.npy data signals
    # Create one big plot with an individual subplot for each participant and then save the file
    w, h = 150, 75
    ncols, nrows = 5, 4
    fontsize = 20
    fig_raw_clean = plt.figure(figsize=(w, h))
    fig_raw_clean.suptitle(f'Predicted vs GT signals', fontsize=fontsize)
    spec_raw_clean = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig_raw_clean)

    fig_phasic = plt.figure(figsize=(w, h))
    fig_phasic.suptitle(f'Predicted vs GT signals', fontsize=fontsize)
    spec_phasic = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig_phasic)

    fig_tonic = plt.figure(figsize=(w, h))
    fig_tonic.suptitle(f'Predicted vs GT signals', fontsize=fontsize)
    spec_tonic = gridspec.GridSpec(ncols=ncols, nrows=nrows, figure=fig_tonic)

    i = 0
    for participant in participants:
        if participant in configs_pre['exclusion_list_shimmer']:
            continue
        data = np.load(data_path + f'/{participant}_label_eda.npy')
        signals, info = nk.eda_process(data, sampling_rate=configs_pre['fs_all']['et'])

        plot_eda_signal(fig_raw_clean, spec_raw_clean, i, [signals['EDA_Raw'], signals['EDA_Clean']], participant,
                        configs_pre, signals)
        plot_eda_signal(fig_phasic, spec_phasic, i, [signals['EDA_Phasic']], participant, configs_pre, signals)
        plot_eda_signal(fig_tonic, spec_tonic, i, [signals['EDA_Tonic']], participant, configs_pre, signals)
        i += 1

    fig_raw_clean.savefig(f'/local/home/bjbraun/Projects/egoPPG/plots/EDA/eda_raw_clean.png')
    fig_phasic.savefig(f'/local/home/bjbraun/Projects/egoPPG/plots/EDA/eda_phasic.png')
    fig_tonic.savefig(f'/local/home/bjbraun/Projects/egoPPG/plots/EDA/eda_tonic.png')


if __name__ == "__main__":
    main()
