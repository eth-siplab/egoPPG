from scipy.signal import find_peaks
from scipy.stats import iqr
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import ShortTimeFFT


def is_double(peak, peaks, threshold: float = 0.05):
    upper = peaks * 2 * (1 + threshold)
    lower = peaks * 2 * (1 - threshold)

    if np.any((peak > lower) & (peak < upper)):
        return True

    return False


def is_treble(peak, peaks, threshold: float = 0.05):
    upper = peaks * 3 * (1 + threshold)
    lower = peaks * 3 * (1 - threshold)

    if np.any((peak > lower) & (peak < upper)):
        return True

    return False


def is_multiple(peak, peaks, multiple, threshold: float = 0.05):
    upper = peaks * multiple * (1 + threshold)
    lower = peaks * multiple * (1 - threshold)

    if np.any((peak > lower) & (peak < upper)):
        return True

    return False


def STFT_from_window(x, fs, nperseg, noverlap, padding_factor, high_pass, fft_mode='centered', scale_to='magnitude',
                     phase_shift=None):
    win = ('kaiser', 1)
    SFT = ShortTimeFFT.from_window(win, fs, nperseg, noverlap, mfft=nperseg * padding_factor, fft_mode=fft_mode,
                                   scale_to=scale_to, phase_shift=phase_shift)

    Zxx = SFT.stft(x)
    # extent[0] minimum time value, extent[1] maximum time value,
    # extent[2] minimum frequency value, extent[3] maximum frequency value
    extent = SFT.extent(len(x), center_bins=True)  # minimum and maximum time-frequency values

    f = np.linspace(extent[2], extent[3], Zxx.shape[0])
    t = np.linspace(extent[0], extent[1], Zxx.shape[1])

    magnitude = np.abs(Zxx)[(f > 0) & (f < high_pass), :]
    f = f[(f > 0) & (f < high_pass)]
    extent = (t[0], t[-1], f[0], f[-1])

    return magnitude, extent


def get_fft_peaks(stft_slice, threshold, height, width=None):
    slice_max = np.max(stft_slice)
    peaks = find_peaks(stft_slice, prominence=threshold * slice_max, height=height * slice_max, width=width)[0]

    return peaks


def get_stft_peaks(stft_mat, threshold, height, width=None, mid_point=False, double_threshold=0.05):
    if mid_point:
        mid_point = stft_mat.shape[0] // 2
    else:
        mid_point = 0

    stft_peaks = np.zeros_like(stft_mat)
    for tmsp in range(stft_mat.shape[1]):
        peaks = get_fft_peaks(stft_mat[:, tmsp], threshold, height, width)

        if len(peaks) > 0:
            peaks = peaks[peaks > mid_point]

        if len(peaks) > 0:
            # select the three peaks with the highest prominence
            peaks = peaks[np.argsort(stft_mat[peaks, tmsp])[max(-1, -len(peaks)):]]
            for peak in peaks:
                if (not is_double(peak, peaks, double_threshold)) and (not is_treble(peak, peaks, double_threshold)):
                    stft_peaks[peak, tmsp] = 1

    # plt.imshow(stft_peaks, origin='lower', aspect='auto')
    # plt.show()

    return stft_peaks


def get_long_curves(curve_lens, threshold, idx_offset=1):
    if isinstance(curve_lens, list):
        curve_lens = np.array(curve_lens)

    curve_idxs = np.where(np.array(curve_lens) >= threshold)[0]
    curve_lens = curve_lens[curve_idxs]
    curve_idxs += idx_offset

    return curve_idxs, curve_lens


def convert_curves_to_points(curve_idxs, selected_curves, individually=True):
    plot_curve_mat = np.zeros_like(curve_idxs)
    points_curves = []

    for c in selected_curves:
        y_idxs, x_idxs = np.where(curve_idxs == c)

        # x_idxs = np.where(np.any(curve_idxs==c, axis = 1))[0]
        # y_idxs = np.array([np.where(curve_idxs[:, x] == c)[0] for x in x_idxs])
        # sort x_idxs and y_idxs by x_idxs
        sort_idxs = np.argsort(x_idxs)

        y_idxs = y_idxs[sort_idxs]
        x_idxs = x_idxs[sort_idxs]

        points_curves.append(list(zip(x_idxs, y_idxs)))

    return points_curves


def track_curves(magnitude_peaks, search_radius_px: int = 5, maximum_gap: int = 5, min_length: int = 100):
    curve_idx = 0
    curve_idxs = np.zeros_like(magnitude_peaks).astype(int)

    start_col = 0
    starting_points = []

    while len(starting_points) == 0 and start_col < magnitude_peaks.shape[1]:
        starting_points = np.where(magnitude_peaks[:, start_col] == 1)[0]
        start_col += 1

    if len(starting_points) == 0:
        return curve_idxs, [], []

    for sp in starting_points:
        curve_idxs[sp, start_col-1] = curve_idx
        curve_idx += 1

    for tmp in range(start_col-1, curve_idxs.shape[1]):
        curve_points = np.where(magnitude_peaks[:, tmp] == 1)[0]

        for cp in curve_points:
            gap = 0
            prev_col = curve_idxs[:, max(0, tmp - 1 - gap)]
            previous_curves = np.where(prev_col > 0)[0]
            possible_curves = previous_curves[np.abs(previous_curves - cp) < search_radius_px]

            while len(possible_curves) == 0 and gap < maximum_gap:
                gap += 1
                prev_col = curve_idxs[:, max(0, tmp - 1 - gap)]
                previous_curves = np.where(prev_col > 0)[0]
                possible_curves = previous_curves[np.abs(previous_curves - cp) < search_radius_px]

            if len(possible_curves) == 0:

                curve_idxs[cp, tmp] = curve_idx
                curve_idx += 1

            elif len(possible_curves) == 1:
                curve_idxs[cp, tmp] = int(prev_col[possible_curves][0])

            else:
                # if there are multiple, take the closest which might be random here given argmin behavior
                curve_idxs[cp, tmp] = int(np.array(prev_col)[possible_curves][np.argmin(np.abs(possible_curves - cp))])

    curve_lens = []
    for i in range(1, int(np.max(curve_idxs))):
        curve_lens.append(len(np.where(curve_idxs == i)[0]))

    selected_curves, curve_lens = get_long_curves(curve_lens, min_length)
    points_curves = convert_curves_to_points(curve_idxs, selected_curves)

    return curve_idxs, curve_lens, points_curves


def get_curves(magnitude: np.array, threshold: float = 0.1, height: float = 0.1, width: int = 1,
               mid_point: bool = False, double_threshold: float = 0.05, search_radius_px: int = 5,
               maximum_gap: int = 5, min_length: int = 100):
    # plt.imshow(magnitude, origin='lower', aspect='auto')
    # plt.show()

    stft_peaks = get_stft_peaks(magnitude, threshold=threshold, height=height, width=width, mid_point=mid_point,
                                double_threshold=double_threshold)
    # plt.imshow(stft_peaks, origin='lower', aspect='auto')
    # plt.show()

    # Go through peaks and assigns curve
    curve_idxs, curve_lens, points_curves = track_curves(stft_peaks, search_radius_px, maximum_gap, min_length)
    # plt.imshow(curve_idxs, origin='lower', aspect='auto')
    # plt.show()

    return curve_idxs, curve_lens, points_curves, stft_peaks


def refine_curves(stft_peaks, curve_lens, points_curves, f, std_1:int = 3, std_2:int=5, debug=False):
    weights = []
    means = []
    y_s_l = []

    if len(points_curves) == 0:
        return np.zeros_like(stft_peaks), [], []

    for i, c in enumerate(points_curves):
        x_s = [p[0] / 5 - 5.15 for p in c]
        y_s = [p[1] for p in c]

        y_s_l.extend(y_s)

        weights.append(len(x_s))
        means.append(np.mean(y_s))

    median = np.median(y_s_l)
    std = iqr(y_s_l) / 1.349

    if debug:
        for i, c in enumerate(points_curves):
            x_s = [p[0] / 5 - 5.15 for p in c]
            y_s = [p[1] for p in c]

            if np.abs(median - np.mean(y_s)) < std_1 * std:
                plt.plot(x_s, f[(f > 0) & (f < 3.5)][y_s], 'y-')

    # put them in an array again of the shape as magnitude_peaks
    curve_idxs = np.zeros_like(stft_peaks)

    for i, c in enumerate(points_curves):
        x_s = [p[0] for p in c]
        y_s = [p[1] for p in c]
        if np.abs(median - np.mean(y_s)) < std_1 * std:
            curve_idxs[y_s, x_s] = 1

    # anything below 0.5 Hz goes to zero
    # curve_idxs[f < 0.5, :] = 0

    # if there are two possible curves at some point in time, take the points that are closest to the median
    for tmsp in range(curve_idxs.shape[1]):

        curve_points = np.where(curve_idxs[:, tmsp] == 1)[0]

        if len(curve_points) > 1:
            closest = np.argmin(np.abs(curve_points - median))
            curve_idxs[:, tmsp] = 0
            curve_idxs[curve_points[closest], tmsp] = 1

    # get new meidan and std
    median = np.median(np.where(curve_idxs == 1)[0])
    std = iqr(np.where(curve_idxs == 1)[0]) / 1.349

    # set everything beyond std_2 std of the median to 0
    curve_idxs[np.arange(curve_idxs.shape[0])[np.abs(np.arange(curve_idxs.shape[0]) - median) > std_2 * std], :] = 0

    return curve_idxs, curve_lens, points_curves