import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pywt
import scipy.io

from source.utils import quantile_artifact_removal, normalize

from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, hilbert, freqz, firwin, remez, kaiser_atten, kaiser_beta


def wavelet_denoise(data, wavelet='db4', level=3):
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(data, wavelet, level=level)

    # Thresholding
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    denoised_coeffs = coeffs[:]
    denoised_coeffs[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in denoised_coeffs[1:]]

    # Reconstruct the signal using the thresholded coefficients
    denoised_rppg_signal = pywt.waverec(denoised_coeffs, wavelet)
    return denoised_rppg_signal


def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    taps = firwin(ntaps, [lowcut, highcut], fs=fs, pass_zero=False,
                  window=window, scale=False)
    return taps


def bandpass_kaiser(ntaps, lowcut, highcut, fs, width):
    atten = kaiser_atten(ntaps, width/(0.5*fs))
    beta = kaiser_beta(atten)
    taps = firwin(ntaps, [lowcut, highcut], fs=fs, pass_zero=False,
                  window=('kaiser', beta), scale=False)
    return taps


def bandpass_remez(ntaps, lowcut, highcut, fs, width):
    delta = 0.5 * width
    edges = [0, lowcut - delta, lowcut + delta,
             highcut - delta, highcut + delta, 0.5*fs]
    taps = remez(ntaps, edges, [0, 1, 0], fs=fs)
    return taps


def filter_ppg(ppg, fs, signal_name):
    # Bandpass filter signal
    ppg = np.asarray(nk.ppg_clean(ppg, sampling_rate=fs, method='elgendi'))

    show_plots = False
    if show_plots:
        fig, ax = plt.subplots()
        ax.plot(ppg[:15*fs])
        ax.set_title(f'{signal_name}: Filtered signal')
        fig.show()

    # [b, a] = butter(2, [0.5 / (fs / 2), 3.0 / (fs / 2)], btype='bandpass')
    # ppg = scipy.signal.filtfilt(b, a, ppg)

    if show_plots:
        fig, ax = plt.subplots()
        ax.plot(ppg)
        ax.set_title(f'{signal_name}: Second filtered signal')
        fig.show()

    return ppg


def filter_imu(imu, fs, signal_name):
    show_plots = False
    if show_plots:
        fig, ax = plt.subplots()
        ax.plot(imu)
        ax.set_title(f'{signal_name}: Raw signal')
        fig.show()

    [b, a] = butter(4, [4.0 / (fs / 2)], btype='lowpass')
    if len(imu.shape) == 1:
        imu = scipy.signal.filtfilt(b, a, imu)
        imu = normalize(imu, 'zero_one')
    else:
        for i in range(imu.shape[1]):
            imu[:, i] = scipy.signal.filtfilt(b, a, imu[:, i])
            imu[:, i] = normalize(imu[:, i], 'zero_one')

    if show_plots:
        fig, ax = plt.subplots()
        ax.plot(imu)
        ax.set_title(f'{signal_name}: Lowpass filtered signal')
        fig.show()

     # Filter out sudden artifacts
    threshold_sudden_filtering = 0.01
    sig_change = np.diff(imu, axis=0)
    # big_diffs = np.where(np.abs(sig_change) > threshold_sudden_filtering)
    big_diffs = np.where(np.abs(sig_change) > 6 * np.std(sig_change))  # 6 * np.std(sig_change)
    for i in range(len(big_diffs[0])):
        imu[big_diffs[0][i] + 1:] = imu[big_diffs[0][i] + 1:] + (imu[big_diffs[0][i]] - imu[big_diffs[0][i] + 1])

    if show_plots:
        fig, ax = plt.subplots()
        ax.plot(imu)
        ax.set_title(f'{signal_name}: Artifact filtered signal 1')
        fig.show()

     # Filter out sudden artifacts
    big_diffs = np.where(abs((np.mean(imu) - imu)) > 1 * np.std(imu))
    for i in range(len(big_diffs[0])-1):
        imu[big_diffs[0][i] + 1:] = imu[big_diffs[0][i] + 1:] + (imu[big_diffs[0][i]] - imu[big_diffs[0][i] + 1])

    if show_plots:
        fig, ax = plt.subplots()
        ax.plot(imu)
        ax.set_title(f'{signal_name}: Artifact filtered signal 2')
        fig.show()

    # https://pmc.ncbi.nlm.nih.gov/articles/PMC3203837/
    low_pass, high_pass = 0.15, 0.6  # 0.15, 0.6
    [b, a] = butter(4, [low_pass / (fs / 2), high_pass / (fs / 2)], btype='bandpass')
    if len(imu.shape) == 1:
        imu = scipy.signal.filtfilt(b, a, imu)
    else:
        for i in range(imu.shape[1]):
            imu[:, i] = scipy.signal.filtfilt(b, a, imu[:, i])

    if show_plots:
        fig, ax = plt.subplots()
        ax.plot(imu)
        ax.set_title(f'{signal_name}: Butterworth-filtered signal')
        fig.show()

    signals_imu, info_imu = nk.rsp_process(imu, sampling_rate=fs, show=False)
    imu = signals_imu['RSP_Clean']

    if show_plots:
        fig, ax = plt.subplots()
        ax.plot(imu)
        ax.set_title(f'{signal_name}: NK-filtered signal')
        fig.show()

    # from source.sp.ICA import ICA
    # imu = ICA(imu, fs, low_pass, high_pass)
    # from source.sp.POS_WANG import CHROME_DEHAAN
    # imu = CHROME_DEHAAN(imu, fs, low_pass, high_pass)

    """if show_plots:
        fig, ax = plt.subplots()
        ax.plot(imu[:30*fs])
        ax.set_title(f'{signal_name}: Filtered signal')
        fig.show()"""

    return imu


def filter_rppg(rppg_signal, fs, low_pass, high_pass, filter_type, ntaps=128):
    plot_signals = True

    if plot_signals:
        fig, ax = plt.subplots()
        ax.plot(rppg_signal)
        ax.set_title('Original signal')
        fig.show()

    rppg_signal = quantile_artifact_removal(rppg_signal, 0.25, 0.75, 3)
    # rppg_signal = np.asarray(nk.ppg_clean(rppg_signal, sampling_rate=fs, method='elgendi'))

    if plot_signals:
        fig, ax = plt.subplots()
        ax.plot(rppg_signal)
        ax.set_title('Original signal')
        fig.show()

    # Filter signal
    if filter_type == 'butter':
        [b, a] = butter(4, [low_pass / (fs / 2), high_pass / (fs / 2)], btype='bandpass')
        rppg_signal = scipy.signal.filtfilt(b, a, rppg_signal)
    elif filter_type == 'firwin':  # Larger width = sharper cutoff, higher ntaps = sharper cutoff
        taps = bandpass_firwin(ntaps, low_pass, high_pass, fs=fs)
        # taps = bandpass_kaiser(ntaps, low_pass, high_pass, fs=fs, width=1.0)
        # taps = bandpass_remez(ntaps, low_pass, high_pass, fs=fs, width=1.0)
        rppg_signal = scipy.signal.filtfilt(taps, 1.0, rppg_signal)
    else:
        raise ValueError('Filter type not supported')

    # rppg_signal = wavelet_denoise(rppg_signal, wavelet='db4', level=2)

    if plot_signals:
        signals_sys, info_sys = nk.ppg_peaks(rppg_signal, sampling_rate=fs, method="elgendi", show=False)
        peaks = info_sys["PPG_Peaks"]
        fig, ax = plt.subplots()
        ax.plot(rppg_signal)
        y_peaks = np.asarray([rppg_signal[i] for i in peaks])
        ax.scatter(peaks, y_peaks, color='green')
        ax.set_title('Filtered signal')
        fig.show()

    return rppg_signal


def filter_rr(rr, fs, signal_name):
    # Bandpass filter signal
    rr = np.asarray(nk.rsp_clean(rr, sampling_rate=fs, method="khodadad2018"))

    show_plots = False
    if show_plots:
        fig, ax = plt.subplots()
        ax.plot(rr[:15*fs])
        ax.set_title(f'{signal_name}: Filtered signal')
        fig.show()

    return rr


def get_edr(ecg, fs_ecg):
    # RR from ECG
    rpeaks, info = nk.ecg_peaks(ecg, sampling_rate=fs_ecg)
    ecg_rate = nk.ecg_rate(rpeaks, sampling_rate=fs_ecg, desired_length=len(ecg))
    edr = nk.ecg_rsp(ecg_rate, sampling_rate=fs_ecg)

    """fig, ax = plt.subplots()
    ax.plot(signals_RSP['RSP_Clean'], label='Cleaned EDR')
    y_peaks = np.asarray([signals_RSP['RSP_Clean'][i] for i in info_RSP['RSP_Peaks']])
    ax.scatter(info_RSP['RSP_Peaks'], y_peaks, color='green')
    ax.set_title('EDR')
    ax.legend()
    fig.show()"""

    return edr


def plot_filter_frequency_response(b, a, fs):
    w, h = freqz(b, a, worN=2000, fs=fs)
    fig, ax1 = plt.subplots()
    ax1.set_title('Digital filter frequency response')
    ax1.plot(w, abs(h), label='Hamming')
    ax1.set_ylabel('Gain', color='b')
    ax1.set_xlabel('Frequency [Hz]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w, angles, 'g')
    ax2.set_ylabel('Angle (radians)', color='g')
    ax2.grid(True)
    ax2.axis('tight')
    fig.show()
