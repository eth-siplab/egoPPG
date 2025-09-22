import math
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import scipy.io
import unisens

from source.sp.curve_tracing import get_stft_peaks, track_curves, refine_curves, STFT_from_window
from source.sp.signal_filtering import wavelet_denoise

from scipy.signal import butter, find_peaks
from source.utils import normalize


def get_closest_task_category(start_time, win_size, task_times_part):
    closest_category = None
    for task in ['video', 'office', 'walking_1', 'kitchen', 'walking_2', 'dancing', 'bike', 'walking_3', 'running']:
        if task_times_part[task][0] <= start_time + task_times_part['video'][0] + win_size/2 <= task_times_part[task][1]:
            closest_category = task
            break
    if closest_category in ['walking_1', 'walking_2', 'walking_3']:
        closest_category = 'walking'
    if closest_category is None:
        closest_category = np.nan
    return closest_category


def load_timeseries_data(data_path):
    ms_ecg = np.load(f'{data_path}/ms_ecg.npy')
    ms_imu = np.load(f'{data_path}/ms_imu.npy')
    aria_imu_left = np.load(f'{data_path}/aria_imu_left.npy', allow_pickle=True).item()
    aria_imu_right = np.load(f'{data_path}/aria_imu_right.npy', allow_pickle=True).item()
    return ms_ecg, ms_imu, aria_imu_left, aria_imu_right


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _check_acc_magn(acc_magn, fft_hr, fs_acc, low_pass, high_pass, mask_f_ppg, mask_pxx_ppg):
    # Get filtered periodogram of accelerometer magnitude signal
    N_acc = _next_power_of_2(acc_magn.shape[0])
    f_acc, pxx_acc = scipy.signal.periodogram(acc_magn, fs=fs_acc, nfft=N_acc, detrend=False)
    mask_acc = np.argwhere((f_acc >= low_pass) & (f_acc <= high_pass))
    mask_f_acc = np.take(f_acc, mask_acc)
    mask_pxx_acc = np.take(pxx_acc, mask_acc)

    # Get dominant frequency of PPG and accelerometer magnitude signal
    dominant_acc_magn_hr = np.take(mask_f_acc, np.argmax(mask_pxx_acc, 0))[0] * 60
    # dominant_ppg_freq = np.take(mask_f_ppg, np.argmax(mask_pxx_ppg, 0))[0]

    # If both the dominant frequencies don't match, then PPG frequency is the estimated heart rate from this window
    # If both of them match, then check the next strongest PPG frequencies if there is another good candidate
    if dominant_acc_magn_hr != fft_hr:
        fft_hr_out = fft_hr
    else:
        k = 3
        top_k_dominant_ppg_freq = np.take(mask_f_ppg, np.argsort(mask_pxx_ppg, axis=0)[-3:])
        top_k_dominant_acc_magn_freq = np.take(mask_f_ppg, np.argsort(mask_pxx_ppg, axis=0)[-3:])
        idx = 0
        while idx != (k - 1):
            if top_k_dominant_acc_magn_freq[idx] != top_k_dominant_ppg_freq[idx]:
                fft_hr_out = top_k_dominant_ppg_freq[idx] * 60
                break
            idx += 1
    return fft_hr_out


def calculate_fft_hr(ppg_signal, fs, acc_magn=None, fs_acc=None, prev_hr=None, peak_hr=None, low_pass=0.70,
                     high_pass=3.0):
    ppg_signal = np.diff(ppg_signal, n=1, axis=0)

    plot_signals = False
    if plot_signals:
        fig, ax = plt.subplots()
        ax.plot(ppg_signal)
        ax.set_title('Original signal')
        fig.show()

    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)

    # Get only the frequencies between low_pass and high_pass
    mask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_f_ppg = np.take(f_ppg, mask_ppg)
    mask_pxx_ppg = np.take(pxx_ppg, mask_ppg)

    if plot_signals:
        fig, ax = plt.subplots()
        ax.plot(ppg_signal)
        ax.set_title('Original signal')
        fig.show()

        plt.semilogy(mask_f_ppg, mask_pxx_ppg)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.show()

        plt.semilogy(f_ppg, pxx_ppg[0, :])
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.show()

    # If peak_hr is given, choose the HR closest to the peak_hr
    if peak_hr is not None:
        possible_hrs = np.take(mask_f_ppg, np.argsort(mask_pxx_ppg, axis=0)[-3:]) * 60
        fft_hr = possible_hrs[np.argmin(np.abs(possible_hrs - peak_hr))][0]
    # If prev_hr is given, choose the HR closest to the prev_hr
    elif prev_hr is not None:
        n_peaks = 2
        possible_hrs_idx = np.argsort(mask_pxx_ppg, axis=0)[-n_peaks:]
        prev_hr_idx = np.argwhere(prev_hr/60 == mask_f_ppg)[0][0]
        possible_hrs_idx_select = []
        for idx in range(n_peaks):
            if (possible_hrs_idx[idx][0] % prev_hr_idx == 0) and (possible_hrs_idx[idx][0] != prev_hr_idx):
                continue
            else:
                possible_hrs_idx_select.append(possible_hrs_idx[idx])
        possible_hrs = np.take(mask_f_ppg, possible_hrs_idx_select) * 60
        if len(possible_hrs) == 0:
            fft_hr = np.take(mask_f_ppg, np.argmax(mask_pxx_ppg, 0))[0] * 60
        else:
            fft_hr = possible_hrs[np.argmin(np.abs(possible_hrs - prev_hr))][0]
    else:
        fft_hr = np.take(mask_f_ppg, np.argmax(mask_pxx_ppg, 0))[0] * 60

    # If accelerometer magnitude signal is given, check if the dominant frequencies match
    # if acc_magn is not None:
    #     fft_hr = _check_acc_magn(acc_magn, fft_hr, fs_acc, low_pass, high_pass, mask_f_ppg, mask_pxx_ppg)

    return fft_hr


def calculate_peak_hr(ppg_signal, fs, reject_outliers=True, prev_hr=None):
    ppg_signal = ppg_signal - min(ppg_signal)
    ppg_signal = ppg_signal / max(ppg_signal)

    # NK requires PPG signal to be inverted (have standard PPG form)
    signals_sys, info_sys = nk.ppg_peaks(ppg_signal, sampling_rate=fs, method="elgendi", show=False)
    peaks = info_sys["PPG_Peaks"]
    # peaks = get_simple_footpoints(ppg_signal, peaks, 'Min', fs)

    plot_signals = False
    if plot_signals:
        fig, ax = plt.subplots()
        ax.plot(ppg_signal)
        y_peaks = np.asarray([ppg_signal[i] for i in peaks])
        ax.scatter(peaks, y_peaks, color='green')
        fig.show()

    ibis = calculate_ibis(peaks, fs, 1.5, reject_outliers=reject_outliers)
    peaks_hr = 60 / (np.mean(ibis) / fs)

    return peaks_hr


def create_perfect_sine(ppg_signal, fs):
    """
    Create a perfect sine wave signal that has its peaks aligned with the detected PPG peaks.

    Parameters:
        ppg_signal (np.array): The preprocessed PPG signal (e.g., truncated segment).
        peaks (list or np.array): The indices of the detected peaks in ppg_signal.
        fs (int or float): The sampling frequency.

    Returns:
        sine_wave (np.array): A perfect sine wave whose peaks align with the detected peaks.
    """
    # Create a time vector for the given signal segment
    t = np.arange(len(ppg_signal)) / fs

    peaks = calculate_peaks(ppg_signal, fs)

    # Define the phase at each detected peak.
    # We set the phase at the first peak to pi/2 (sine maximum),
    # then each subsequent peak increases the phase by 2*pi.
    phases_at_peaks = np.pi / 2 + 2 * np.pi * np.arange(len(peaks))

    # Use linear interpolation to compute the phase at every time point in the segment.
    # For time points outside the range of the detected peaks, np.interp
    # will by default use the value at the nearest boundary.
    phase = np.interp(t, t[peaks], phases_at_peaks)

    # Construct the sine wave from the phase
    sine_wave = np.sin(phase)

    plot_signals = False
    if plot_signals:
        start = 10
        end = 30
        fig, ax = plt.subplots()
        ax.plot(normalize(sine_wave[start*fs:end*fs], 'zero_one'))
        ax.plot(normalize(ppg_signal[start*fs:end*fs], 'zero_one'))
        # y_peaks = np.asarray([ppg_signal[i] for i in peaks])
        # ax.scatter(peaks, y_peaks, color='green')
        fig.show()

    return sine_wave


def calculate_peaks(ppg_signal, fs):
    ppg_signal = ppg_signal - min(ppg_signal)
    ppg_signal = ppg_signal / max(ppg_signal)

    # NK requires PPG signal to be inverted (have standard PPG form)
    signals_sys, info_sys = nk.ppg_peaks(ppg_signal, sampling_rate=fs, method="elgendi", show=False)
    peaks = info_sys["PPG_Peaks"]
    # peaks = get_simple_footpoints(ppg_signal, peaks, 'Min', fs)
    # peaks = find_peaks(ppg_signal, distance=60/180*fs, prominence=0.1)[0]

    plot_signals = False
    if plot_signals:
        fig, ax = plt.subplots()
        ax.plot(ppg_signal)
        y_peaks = np.asarray([ppg_signal[i] for i in peaks])
        ax.scatter(peaks, y_peaks, color='green')
        fig.show()

    return peaks


def calculate_peak_rr(rr_signal, fs, signal_name, prev_rr, reject_outliers=True):
    # NK requires PPG signal to be inverted (have standard PPG form)
    from scipy.signal import find_peaks
    signals_rr, info_rr = nk.rsp_process(rr_signal, sampling_rate=fs, show=False)
    peaks = info_rr['RSP_Peaks']
    # peaks, _ = find_peaks(rr_signal, distance=60/35*fs)

    if len(peaks) in [0, 1]:
        print('AH')
        return prev_rr
    # peaks = get_simple_footpoints(ppg_signal, peaks, 'Min', fs)

    plot_signals = False
    if plot_signals:
        fig, ax = plt.subplots()
        ax.plot(signals_rr['RSP_Clean'])
        y_peaks = np.asarray([signals_rr['RSP_Clean'][i] for i in peaks])
        ax.scatter(peaks, y_peaks, color='green')
        ax.set_title(f'{signal_name}: Cleaned signal')
        fig.show()

    ibis = calculate_ibis(peaks, fs, 30, reject_outliers=reject_outliers)
    peaks_rr = 60 / (np.mean(ibis) / fs)

    return peaks_rr


def get_curves(magnitude: np.array, extent, threshold: float = 0.1, height: float = 0.1, width: int = 1,
               mid_point: bool = False, double_threshold: float = 0.05, search_radius_px: int = 5, maximum_gap: int = 5,
               min_length: int = 100):
    fig, ax = plt.subplots()
    ax.imshow(magnitude, extent=extent, origin='lower', aspect='auto')
    ax.set_title('Magnitude')
    fig.show()

    stft_peaks = get_stft_peaks(magnitude, threshold=threshold, height=height, width=width, mid_point=mid_point,
                                double_threshold=double_threshold)
    fig, ax = plt.subplots()
    ax.imshow(stft_peaks, extent=extent,  origin='lower', aspect='auto')
    ax.set_title('STFT Peaks')
    fig.show()

    curve_idxs, curve_lens, points_curves = track_curves(stft_peaks, search_radius_px, maximum_gap, min_length)
    fig, ax = plt.subplots()
    ax.imshow(curve_idxs, extent=extent, origin='lower', aspect='auto')
    ax.set_title('Curve Indices')
    fig.show()

    return curve_idxs, curve_lens, points_curves, stft_peaks


def calculate_curve_tracing_hr(rppg_signal, fs, low_pass, high_pass, win_size, plot_signals=False):
    nperseg = 2**9
    noverlap = nperseg-1
    padding_factor = 5
    # Gives back matrix for each window different numbers of frequencies according to overlap and padding factor
    magnitude, extent = STFT_from_window(rppg_signal, fs, nperseg, noverlap, padding_factor, high_pass)

    """fig, ax = plt.subplots()
    ax.plot(rppg_signal[:600])
    fig.show()"""

    # For each window, take FFT peak
    # Ignore peaks that are harmonic (e.g., peak bei 1, dann bei 2 und 3 ignorieren)
    f = np.linspace(extent[2], extent[3], magnitude.shape[0])
    f_index = (f >= low_pass) & (f <= high_pass)
    curve_idxs, curve_lens, points_curves, stft_peaks = get_curves(magnitude[f_index, :], extent, threshold=0.5,
                                                                   height=0.5, width=1, mid_point=False,
                                                                   double_threshold=0, search_radius_px=5,
                                                                   maximum_gap=5, min_length=200)

    # If curve below minimum length, kick out
    debug = False
    curve_idxs, curve_lens, points_curves = refine_curves(stft_peaks, curve_lens, points_curves, f, std_1=5, std_2=10,
                                                          debug=debug)

    fig, ax = plt.subplots()
    ax.imshow(curve_idxs, extent=extent, origin='lower', aspect='auto')
    ax.set_title('Refined Curve Indices')
    fig.show()

    start_sec = 0
    overlap = 0.5
    pred_hrs = []
    for i_win in range(rppg_signal.shape[0] // fs // win_size):
        end_sec = start_sec + win_size
        t = np.linspace(extent[0], extent[1], magnitude.shape[1])
        t_index = np.where((t >= start_sec) & (t < end_sec))[0]
        f_s = f[f_index][np.where(stft_peaks[:, t_index] > 0)[0]]
        if len(f_s) == 0:
            start_sec += int(win_size * (1 - overlap))
        pred_hrs.append(np.mean(f_s) * 60)
        start_sec += win_size

    return pred_hrs


def calculate_ibis(peaks, fs, threshold, reject_outliers=False):
    ibis = np.diff(peaks)

    # Only keep IBIs that are within the interquartile range
    if reject_outliers:
        ibis = ibis[np.where(ibis < (threshold * fs))]
        if len(ibis) == 0:
            return ibis
        quantile_q1 = np.quantile(ibis, 0.25)
        quantile_q3 = np.quantile(ibis, 0.75)
        ibis = ibis[np.where((ibis >= quantile_q1) & (ibis <= quantile_q3))]

    return ibis


def get_simple_footpoints(ppg_cleaned, peaks, trough_method, fs):
    footpoints = []
    for i_sys, sys_peak in enumerate(peaks):
        if i_sys < len(peaks) - 1:
            diff_next_peak = peaks[i_sys + 1] - sys_peak
            if 1.5 * fs > diff_next_peak > 0.33 * fs:
                end = peaks[i_sys + 1]
            else:
                continue

            footp = get_trough(ppg_cleaned, sys_peak, end, trough_method)
            if not math.isnan(footp):
                footpoints.append(int(footp))

    return np.asarray(footpoints)


def get_trough(data, s, e, trough_method):
    if s is not None and e is not None:
        data = data[s:e]

    if trough_method == 'Tangent':
        sig_1d = np.diff(data, n=1)
        slope, loc_x = max(sig_1d), np.argmax(sig_1d)
        loc_y = data[loc_x]
        min_y = min(data)
        b = loc_y - slope * loc_x
        # tang_footpt = np.floor((min_y - b) / slope)
        footpoint_out = (min_y - b) / slope
    elif trough_method == 'Min':
        footpoint_out = np.argmin(data)
    elif trough_method == '2ndDeriv':
        sig_1d = np.diff(data, n=1)
        sig_2d = np.diff(data, n=2)
        """sig_2d = sig_2d[np.argmin(data):np.argmax(sig_1d)]
        if len(sig_2d) == 0:
            return np.nan
        footpoint_out = np.argmax(sig_2d) + np.argmin(data)"""
        footpoint_out = np.argmax(sig_2d)
    else:
        raise ValueError(f'Trough detection method {trough_method} not supported.')

    if footpoint_out < 0 or footpoint_out >= len(data):
        footpoint_out = np.nan

    if s is None:
        return footpoint_out
    else:
        return footpoint_out+s

# For calculating hr_fft
# If there is a peak which is a harmonic of the previous HR, choose this HR
"""N_prev = np.argwhere(f_ppg == prev_hr / 60)[0][0]
delta_s = 2
R_0_start_idx = N_prev - delta_s
R_0_end_idx = N_prev + delta_s
R_1_start_idx = 2 * (N_prev - delta_s - 1) + 1
R_1_end_idx = 2 * (N_prev + delta_s - 1) + 1
R_0_idx = np.arange(R_0_start_idx, R_0_end_idx + 1)
R_1_idx = np.arange(R_1_start_idx, R_1_end_idx + 1)
R_0 = R_0_idx[(0 < R_0_idx) & (R_0_idx < len(mask_pxx))]
R_1 = R_1_idx[(0 <= R_1_idx) & (R_1_idx < len(mask_pxx))]

n_max = 1
eta = 0.3
N_0 = np.argpartition(mask_pxx[R_0, 0], -n_max)[-n_max:] + R_0_start_idx
N_1 = np.argpartition(mask_pxx[R_1, 0], -n_max)[-n_max:] + R_1_start_idx
threshold = eta * np.max(mask_pxx[R_0, 0])
N_0 = N_0[mask_pxx[N_0, 0] >= threshold]
N_1 = N_1[mask_pxx[N_1, 0] >= threshold]  # N_1 can be empty
if len(N_0) == 0:
    N_0 = np.array([np.argmax(mask_pxx[R_0, 0]) + R_0_start_idx])

N_hat = None
for n_0 in N_0:
    for n_1 in N_1:
        if n_0 % n_1 == 0 or n_1 % n_0 == 0:
            if N_hat is None or np.abs(N_hat - N_prev) > np.abs(n_0 - N_prev):
                N_hat = n_0
if N_hat is not None:
    fft_hr = mask_ppg[N_hat][0] * 60"""