"""The post processing files for caluclating heart rate using FFT or peak detection.
The file also  includes helper funcs such as detrend, mag2db etc.
"""

import matplotlib.pyplot as plt
import numpy as np

from scipy.signal import butter, detrend, find_peaks, filtfilt, periodogram
from sklearn.metrics import mean_squared_error

from utils import normalize, calculate_ibis


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def mag2db(mag):
    """Convert magnitude to db."""
    return 20. * np.log10(mag)


def _calculate_fft_hr_rr(ppg_signal, fs=30, low_pass=0.75, high_pass=2.5):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


def _calculate_peak_hr(ppg_signal, fs, show_plots=False):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)

    if show_plots:
        fig, ax = plt.subplots()
        ax.plot(ppg_signal)
        y_peaks = np.asarray([ppg_signal[i] for i in ppg_peaks])
        ax.scatter(ppg_peaks, y_peaks, color='green')
        fig.show()

    ibis = calculate_ibis(ppg_peaks, fs, 1.5, reject_outliers=True)
    hrv_sdnn = np.std(ibis)

    return hr_peak, hrv_sdnn


def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.75, high_pass=2.5):
    """Calculate SNR as the ratio of the area under the curve of the frequency spectrum around the first and second harmonics 
        of the ground truth HR frequency to the area under the curve of the remainder of the frequency spectrum, from 0.75 Hz
        to 2.5 Hz. 

        Args:
            pred_ppg_signal(np.array): predicted PPG signal 
            label_ppg_signal(np.array): ground truth, label PPG signal
            fs(int or float): sampling rate of the video
        Returns:
            SNR(float): Signal-to-Noise Ratio
    """
    # Get the first and second harmonics of the ground truth HR in Hz
    first_harmonic_freq = hr_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60  # 6 beats/min converted to Hz (1 Hz = 60 beats/min)

    # Calculate FFT
    pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(pred_ppg_signal.shape[1])
    f_ppg, pxx_ppg = periodogram(pred_ppg_signal, fs=fs, nfft=N, detrend=False)

    # Calculate the indices corresponding to the frequency ranges
    idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) \
     & ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) \
     & ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))

    # Select the corresponding values from the periodogram
    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    # Calculate the signal power
    signal_power_hm1 = np.sum(pxx_harmonic1)
    signal_power_hm2 = np.sum(pxx_harmonic2)
    signal_power_rem = np.sum(pxx_remainder)

    # Calculate the SNR as the ratio of the areas
    if not signal_power_rem == 0: # catches divide by 0 runtime warning 
        SNR = mag2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0
    return SNR


def calculate_metric_per_video_ppg(predictions, labels, hr_method, diff_flag, fs, use_bandpass=True):
    """Calculate video-level HR and SNR"""
    mse = mean_squared_error(normalize(detrend(np.cumsum(labels)), 'zero_one'),
                             normalize(detrend(np.cumsum(predictions)), 'zero_one'))
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        predictions = detrend(np.cumsum(predictions))
        labels = detrend(np.cumsum(labels))
    else:
        predictions = detrend(predictions)
        labels = detrend(labels)

    # Invert signal
    predictions = np.max(predictions) - predictions
    labels = np.max(labels) - labels

    show_plots = False
    low_pass, high_pass = 0.7, 2.8  # 0.6, 3.0; 0.5, 2.8
    if use_bandpass:
        if show_plots:
            fig, ax = plt.subplots()
            ax.plot(normalize(predictions, 'zero_one'), label='Prediction')
            ax.plot(normalize(labels, 'zero_one'), label='Ground Truth')
            fig.show()

        [b, a] = butter(4, [low_pass / fs * 2, high_pass / fs * 2], btype='bandpass')  # 0.7, 2.8
        predictions = filtfilt(b, a, np.double(predictions))
        labels = filtfilt(b, a, np.double(labels))

        if show_plots:
            fig, ax = plt.subplots()
            ax.plot(normalize(predictions, 'zero_one'), label='Prediction')
            ax.plot(normalize(labels, 'zero_one'), label='Ground Truth')
            fig.show()

    if hr_method == 'FFT':
        hr_pred = _calculate_fft_hr_rr(predictions, fs=fs, low_pass=low_pass, high_pass=high_pass)
        hr_label = _calculate_fft_hr_rr(labels, fs=fs, low_pass=low_pass, high_pass=high_pass)
        hrv_pred = None
    elif hr_method == 'Peak_Detection':
        hr_pred, hrv_pred = _calculate_peak_hr(predictions, fs=fs, show_plots=show_plots)
        hr_label, _ = _calculate_peak_hr(labels, fs=fs, show_plots=False)
    else:
        raise ValueError('Please use FFT or Peak to calculate your HR.')

    SNR = _calculate_SNR(predictions, hr_label, fs=fs)

    return hr_label, hr_pred, SNR, mse, hrv_pred