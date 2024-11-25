import numpy as np
import scipy
import matplotlib.pyplot as plt

from scipy.linalg import hankel
from scipy.signal import butter
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from sp_helper import _next_power_of_2
from signal_filtering import bandpass_firwin, filter_rppg


# See: https://github.com/hpi-dhc/TROIKA/tree/main

def ssa(ts: np.ndarray, L: int, perform_grouping: bool = True, wcorr_threshold: float = 0.3, ret_Wcorr: bool = False):
    """
    Performs SSA on ts
    https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition

    Parameters
    ----------
        ts : ndarray of shape (n_timestamps, )
            The time series to decompose
        L : int
            first dimension of the L-trajectory-matrix
        grouping : bool, default=True
            If True, perform grouping based on the w-correlations of the deconstructed time series
            using agglomerative hierarchical clustering with single linkage.
            If this parameter is True, the parameter distance_threshold must be set.
        wcorr_threshold : float, default=0.3
            The w-correlation threshold used with the agglomerative hierarchical clustering.
            Time series with at least this w-correlation will be grouped together.
            This parameter will be ignored if grouping is set to False.
        ret_Wcorr : bool, default=False
            Whether the resulting w-correlation matrix should be returned.
            If grouping is enabled, return the w-correlation matrix of the grouped time series.
            If grouping is disabled, return the w-correlation matrix of the ungrouped time series.


    Returns
    ----------
        Y : ndarray of shape (n_groups, n_timestamps) if grouping is enabled and (L, n_timestamps) if it is disabled.
        Wcorr : ndarray
            The Wcorrelation matrix.
            Wcorr will only be returned if ret_Wcorr is True
    """
    N = len(ts)
    K = N - L + 1

    L_trajectory_matrix = hankel(ts[:L], ts[L - 1:])  # (L, K)
    U, Sigma, V = np.linalg.svd(L_trajectory_matrix)  # (L, L); (d, ); (K, K)
    V = V.T  # (K, K)
    d = len(Sigma)

    deconstructed_ts = []
    for i in range(d):
        X_elem = np.array(Sigma[i] * np.outer(U[:, i], V[:, i]))  # (L, K)
        X_elem_rev = X_elem[::-1]  # (L, K)
        ts_i = np.array([X_elem_rev.diagonal(i).mean() for i in range(-L + 1, K)])
        deconstructed_ts.append(ts_i)
    deconstructed_ts = np.array(deconstructed_ts)  # (d, L, K)

    if not perform_grouping and not ret_Wcorr:
        return deconstructed_ts

    w = np.concatenate((np.arange(1, L + 1), np.full((K - L,), L), np.arange(L - 1, 0, -1)))

    def wcorr(ts1: np.ndarray, ts2: np.ndarray) -> float:
        """
        weighted correlation of ts1 and ts2.
        w is precomputed for reuse.
        """
        w_covar = (w * ts1 * ts2).sum()
        ts1_w_norm = np.sqrt((w * ts1 * ts1).sum())
        ts2_w_norm = np.sqrt((w * ts2 * ts2).sum())

        return w_covar / (ts1_w_norm * ts2_w_norm)

    Wcorr_mat = pairwise_distances(deconstructed_ts, metric=wcorr)

    if not perform_grouping:
        return deconstructed_ts, Wcorr_mat

    Wcorr_mat_dist = 1 - Wcorr_mat
    distance_threshold = 1 - wcorr_threshold
    agg_clust = AgglomerativeClustering(linkage='single',
                                        distance_threshold=distance_threshold, n_clusters=None)
    clust_labels = agg_clust.fit_predict(Wcorr_mat_dist)
    n_clusters = clust_labels.max() + 1
    grouped_ts = [np.sum(deconstructed_ts[clust_labels == cluster_id], axis=0)
                  for cluster_id in range(n_clusters)]
    grouped_ts = np.array(grouped_ts)

    if not ret_Wcorr:
        return grouped_ts

    Wcorr_mat = pairwise_distances(grouped_ts, metric=wcorr)

    return grouped_ts, Wcorr_mat


class SpectralPeakTracker:
    def __init__(self, n_freq_bins=1024, ppg_sampling_freq=30, eta=.3):
        self.n_freq_bins = n_freq_bins
        self.ppg_sampling_freq = ppg_sampling_freq
        bin_difference = ppg_sampling_freq / n_freq_bins / 2  # in Hz

        # parameters for stage 2 (peak selection)
        delta_s_bpm = 20
        self.delta_s = int(delta_s_bpm / 60 / bin_difference)  # Check below and above 16 bins for the old peak for the maximum peak
        self.eta = eta

        # parameters for stage 3.1 (verification)
        tau_bpm = 2
        theta_bpm = 20
        self.tau = int(tau_bpm / 60 / bin_difference)  # Add 2 bpm difference if old frequency bigger than old + theta
        self.theta = int(theta_bpm / 60 / bin_difference)  # To not have more than 10 bpm difference

        self.history = []  # frequency indices

    def _get_N0_N1(self, spectrum):
        N_prev = self.history[-1]
        R_0_start_idx = N_prev - self.delta_s
        R_0_end_idx = N_prev + self.delta_s
        R_1_start_idx = 2 * (N_prev - self.delta_s - 1) + 1
        R_1_end_idx = 2 * (N_prev + self.delta_s - 1) + 1
        R_0_idx = np.arange(R_0_start_idx, R_0_end_idx + 1)
        R_1_idx = np.arange(R_1_start_idx, R_1_end_idx + 1)
        R_0 = R_0_idx[(0 < R_0_idx) & (R_0_idx < len(spectrum))]
        R_1 = R_1_idx[(0 <= R_1_idx) & (R_1_idx < len(spectrum))]

        n_max = 3
        N_0 = np.argpartition(spectrum[R_0], -n_max)[-n_max:] + R_0_start_idx
        N_1 = np.argpartition(spectrum[R_1], -n_max)[-n_max:] + R_1_start_idx
        threshold = self.eta * np.max(spectrum[R_0])
        N_0 = N_0[spectrum[N_0] >= threshold]
        N_1 = N_1[spectrum[N_1] >= threshold]  # N_1 can be empty
        if len(N_0) == 0:
            N_0 = np.array([np.argmax(spectrum[R_0]) + R_0_start_idx])

        return N_0, N_1

    def _get_N_hat(self, N_0, N_1):
        N_prev = self.history[-1]

        # Case 1
        N_hat = None
        for n_0 in N_0:
            for n_1 in N_1:
                if n_0 % n_1 == 0 or n_1 % n_0 == 0:
                    if N_hat is None or np.abs(N_hat - N_prev) > np.abs(n_0 - N_prev):
                        N_hat = n_0

        # Case 2
        if N_hat is None:
            Nf_set = np.concatenate((N_0, (N_1 - 1) / 2))
            N_hat_idx = np.argmin(np.abs(Nf_set - N_prev))
            N_hat = Nf_set[N_hat_idx]
            # N_hat = N_0[0]

        return int(N_hat)

    def _verification_stage_1(self, N_hat):
        N_prev = self.history[-1]
        if N_hat - N_prev >= self.theta:
            N_cur = N_prev + self.tau
        elif N_hat - N_prev <= -self.theta:
            N_cur = N_prev - self.tau
        else:
            N_cur = N_hat

        return int(N_cur)

    def transform_first(self, spectrum: np.ndarray):
        N_cur = np.argmax(spectrum)
        self.history.append(N_cur)

        return N_cur

    def transform(self, spectrum: np.ndarray):
        N_0, N_1 = self._get_N0_N1(spectrum)
        N_hat = self._get_N_hat(N_0, N_1)
        N_cur = self._verification_stage_1(N_hat)
        self.history.append(N_cur)

        return N_cur


class Troika:
    def __init__(self, signal_len, win_duration=8, step_duration=2, ppg_sampling_freq=30, acc_sampling_freq=1000,
                 cutoff_freqs=(0.66, 3.0)):
        self.window_duration = win_duration
        self.step_duration = step_duration
        self.sampling_freq = acc_sampling_freq
        self.cutoff_freqs = cutoff_freqs
        self.ppg_sampling_freq = ppg_sampling_freq
        self.acc_sampling_freq = acc_sampling_freq
        self.ppg_window_len = win_duration * ppg_sampling_freq
        self.acc_window_len = win_duration * acc_sampling_freq
        self.ppg_step_len = int(step_duration * ppg_sampling_freq)
        self.acc_step_len = int(step_duration * acc_sampling_freq)
        self.n_freq_bins = _next_power_of_2(signal_len)

    def _get_current_window_bounds(self, cur_window: int, n_ppg_samples: int, n_acc_samples: int):
        ppg_low_bound = (cur_window - 1) * self.ppg_step_len
        ppg_high_bound = min(ppg_low_bound + self.ppg_window_len, n_ppg_samples)
        acc_low_bound = (cur_window - 1) * self.acc_step_len
        acc_high_bound = min(acc_low_bound + self.acc_window_len, n_acc_samples)

        return (ppg_low_bound, ppg_high_bound), (acc_low_bound, acc_high_bound)

    def _get_dominant_frequencies(self, spectrum: np.ndarray, axis=-1, threshold=.5):
        """
        Given the frequency spectra of one or multiple signals, compute the dominant frequencies along a specified axis.
        """

        max_amplitudes = np.max(spectrum, axis=axis, keepdims=True)
        dom_freqs = spectrum > threshold * max_amplitudes

        return dom_freqs

    def _temporal_difference(self, ts, k):
        """
        Perform the kth-order temporal-difference operation on the time series ts.
        """
        diff = np.diff(ts, k)
        diff_pad = np.pad(diff, (0, k), mode='constant', constant_values=0)
        return diff_pad

    # f_acc is the set of location indexes of selected dominant frequencies in the spectra of the acceleration signal
    def _get_f_acc(self, acc_window: np.ndarray, prev_window_hr_idx: int, delta: int = 10):
        _, acc_freqs = scipy.signal.periodogram(acc_window, nfft=self.n_freq_bins * 2 - 1, fs=self.acc_sampling_freq)
        acc_dom_freqs = self._get_dominant_frequencies(acc_freqs)
        F_acc = np.logical_or.reduce(acc_dom_freqs)

        # Filter out harmonic frequencies +- delta of the previous window's HR
        N_p = np.arange(prev_window_hr_idx, self.n_freq_bins + 1, prev_window_hr_idx)
        for idx in N_p:
            F_acc[idx - delta: idx + delta] = False

        return F_acc

    def _filter_ssa_groups(self, ssa_groups: np.ndarray, F_acc: np.ndarray):
        _, ssa_groups_spectra = scipy.signal.periodogram(ssa_groups, nfft=self.n_freq_bins * 2 - 1,
                                                         fs=self.ppg_sampling_freq)
        ssa_dom_freqs = self._get_dominant_frequencies(ssa_groups_spectra)
        group_filter = np.logical_or.reduce(np.logical_and(ssa_dom_freqs, F_acc), axis=1)
        group_filter = np.logical_not(group_filter)

        return ssa_groups[group_filter]

    def transform(self, ppg: np.ndarray, acc: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
            ppg : ndarray of shape (n_channels, n_timestamps)
                The PPG signal.
            acc : ndarray of shape (n_dimensions, n_timestamps)
                The accelaration data used for denoising.
        """
        n_ppg_samples = ppg.shape[-1]
        n_acc_samples = acc.shape[-1]
        n_ppg_windows = int(np.ceil((n_ppg_samples - self.ppg_window_len) / self.ppg_step_len))
        n_acc_windows = int(np.ceil((n_acc_samples - self.acc_window_len) / self.acc_step_len))
        assert n_acc_windows == n_ppg_windows, "The given ppg data and ppg sampling frequency do not have the same number of windows as the given acc signal and acc sampling frequency"
        n_windows = n_acc_windows
        current_window = 1
        progress_bar = tqdm(total=n_windows, initial=current_window)

        spt = SpectralPeakTracker(n_freq_bins=self.n_freq_bins, ppg_sampling_freq=self.ppg_sampling_freq)
        prev_window_hr_idx = None

        [b, a] = butter(4, [self.cutoff_freqs[0] / (self.acc_sampling_freq / 2),
                            self.cutoff_freqs[1] / (self.acc_sampling_freq / 2)], btype='bandpass')

        while current_window <= n_windows:
            (ppg_l, ppg_h), (acc_l, acc_h) = (
                self._get_current_window_bounds(current_window, n_ppg_samples, n_acc_samples))
            progress_bar.set_description(f"Calculating window {current_window}/{n_windows}")

            ppg_window = ppg[ppg_l:ppg_h]
            acc_window = acc[:, acc_l:acc_h]

            acc_window = scipy.signal.filtfilt(b, a, acc_window)
            ppg_window = filter_rppg(ppg_window, self.ppg_sampling_freq, self.cutoff_freqs[0], self.cutoff_freqs[1],
                                     0.1, 128, 'firwin', plot_signals=False)

            if current_window == 1:
                ppg_freq, ppg_spectrum = scipy.signal.periodogram(ppg_window, nfft=self.n_freq_bins * 2 - 1,
                                                                  fs=self.ppg_sampling_freq)
                prev_window_hr_idx = spt.transform_first(ppg_spectrum)
                yield ppg_freq[prev_window_hr_idx] * 60
            else:
                F_acc = self._get_f_acc(acc_window, prev_window_hr_idx, delta=3)
                # ssa_groups = ssa(ppg_window, 100, perform_grouping=True)
                # filtered_ssa_groups = self._filter_ssa_groups(ssa_groups, F_acc)
                # filtered_ppg = np.sum(filtered_ssa_groups, axis=0)
                # temporal_difference = self._temporal_difference(filtered_ppg, 2)
                ppg_freq, ppg_spectrum = scipy.signal.periodogram(ppg_window, nfft=self.n_freq_bins * 2 - 1,
                                                                  fs=self.ppg_sampling_freq)

                """fig, ax = plt.subplots()
                ax.plot(ppg_window)
                ax.set_title('Original signal')
                fig.show()

                fig, ax = plt.subplots()
                ax.semilogy(ppg_freq, ppg_spectrum)
                fig.show()"""

                prev_window_hr_idx = spt.transform(ppg_spectrum)
                yield ppg_freq[prev_window_hr_idx] * 60

            current_window += 1
            progress_bar.update()

        progress_bar.close()
