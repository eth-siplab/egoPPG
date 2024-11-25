import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from source.sp.signal_filtering import filter_rppg, filter_ppg
from source.utils import normalize, resample_signal


def plot_md_rppg_signals():
    # Variable parameters
    participant = '001'
    start = 100
    end = 107
    low_pass = 0.6
    high_pass = 3.0

    # Load signals
    data_path = f'/data/bjbraun/Datasets/PreprocessedData/egoPPG/Data_SP/{participant}'
    channel_values = np.load(f'{data_path}/et_channel_values.npy')
    channel_values_eyes = np.load(f'{data_path}/et_channel_values_eyes.npy')
    ecg = np.load(f'{data_path}/ms_ecg.npy')
    md_ppg = np.load(f'{data_path}/md.npy')
    fs_all = {'et': 30, 'rgb': 15, 'aria_imu_left': 800, 'aria_imu_right': 1000, 'ms_imu': 64, 'ms_ecg': 1024,
              'shimmer': 256, 'md': 128}

    # Invert MD and rPPG and get the defined window
    md_ppg = np.max(md_ppg) - md_ppg
    channel_values = np.max(channel_values) - channel_values
    channel_values_eyes = np.max(channel_values_eyes) - channel_values_eyes
    md_ppg_temp = md_ppg[start*fs_all['md']:end * fs_all['md']]
    channel_values_temp = channel_values[start * fs_all['et']:end * fs_all['et']]
    channel_values_eyes_temp = channel_values_eyes[start * fs_all['et']:end * fs_all['et']]

    # Filter signals
    md_ppg_filtered = filter_ppg(md_ppg_temp, fs_all['md'], 'MD')
    channel_values_filtered = filter_rppg(channel_values_temp, fs_all['et'], low_pass, high_pass, 'butter', ntaps=128)
    channel_values_eyes_filtered = filter_rppg(channel_values_eyes_temp, fs_all['et'], low_pass, high_pass, 'butter', ntaps=128)

    # Raw signals
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.linspace(0, len(channel_values_temp) / fs_all['et'], len(channel_values_temp))
    t = pd.to_datetime(x, unit='s')
    ax.plot(t, normalize(resample_signal(md_ppg_temp, len(channel_values_filtered), 'cubic'), 'zero_one'), label='MD', c='black')
    ax.plot(t, normalize(channel_values_temp, 'zero_one'), label='rPPG')
    ax.plot(t, normalize(channel_values_eyes_temp, 'zero_one'), label='rPPG eyes')
    # ax.set_title('Raw signals')
    # ax.legend()
    ax.xaxis.set_major_locator(mdates.SecondLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%S'))
    # ax.set_xlabel('Time [s]', fontsize=fontsize)
    # ax.axis('off')
    fig.show()
    fig.savefig(f'/local/home/bjbraun/plots/{participant}_raw_signals.svg')

    # Filtered signals
    """fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(normalize(channel_values_filtered, 'zero_one'), label='rPPG')
    ax.plot(normalize(resample_signal(md_ppg_filtered, len(channel_values_filtered), 'cubic'), 'zero_one'), label='MD')
    ax.axis('off')
    fig.show()
    fig.savefig(f'/local/home/bjbraun/plots/{participant}_filtered_signals.svg')"""


def plot_et_frames():
    participant = '005'
    batch = 1  # 4, 4, 5
    frame = 30   # 001: 4, 4, 5...82, 107, 23, 002: 0...99, 003:0 ... 112
    data_path = (f'/data/bjbraun/Datasets/PreprocessedData/egoPPG/Data_ML/CL128_W128_H48_LabelRaw_VideoTypeRaw/Data')
                 # f'CL128_Down1_W128_H48_LabelDiffStandardized_VideoTypeDiffStandardized')
    et_frames = np.load(f'{data_path}/{participant}_input_et{batch}.npy')

    fig, ax = plt.subplots()
    ax.imshow(et_frames[frame], cmap='gray')
    ax.axis('off')
    # ax.set_title(f'Frame {frame}')
    fig.show()
    fig.savefig(f'/local/home/bjbraun/plots/{participant}_et_frame_for_SPA_{(batch % 4) * 128 + frame}.svg')


# plot_et_frames()
plot_md_rppg_signals()
