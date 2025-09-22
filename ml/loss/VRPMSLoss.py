import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.fft

tr = torch
import math
from scipy.interpolate import Akima1DInterpolator


def mse_loss_VRMPS(input, negative_arr):
    l_mse = 0
    for i in range(len(negative_arr)):
        l_mse_negative = torch.nn.MSELoss()(input, negative_arr[i])
        l_mse += l_mse_negative
    return l_mse/len(negative_arr)


class CalculateNormPSD(nn.Module):
    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0):
        x = x - torch.mean(x)
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant', 0)

        # Get PSD
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = torch.add(x[:, :, 0] ** 2, x[:, :, 1] ** 2)

        # Normalize PSD
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return x


"""class CalculateNormPSD(nn.Module):
    # we reuse the code in Gideon2021 to get the normalized power spectral density
    # Gideon, John, and Simon Stent. "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video." Proceedings of the IEEE/CVF international conference on computer vision. 2021.

    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad / 2 * L), int(zero_pad / 2 * L)), 'constant', 0)

        # Get PSD
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = tr.add(x[:, 0] ** 2, x[:, 1] ** 2)

        # Filter PSD for relevant parts
        Fn = self.Fs / 2
        # freqs = torch.linspace(0, Fn, x.shape[0])
        freqs = torch.linspace(0, Fn, x.shape[0], device=x.device)
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        x = x[use_freqs]

        # Normalize PSD
        # x = x / torch.sum(x, dim=-1, keepdim=True)
        x = x / torch.sum(x)
        return x"""


def compute_power_spectrum(signal, Fs, zero_pad=None):
    if zero_pad is not None:
        L = len(signal)
        signal = np.pad(signal, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant')
    freqs = np.fft.fftfreq(len(signal), 1 / Fs) * 60  # in bpm
    ps = np.abs(np.fft.fft(signal))**2
    cutoff = len(freqs)//2
    freqs = freqs[:cutoff]
    ps = ps[:cutoff]
    return freqs, ps


def predict_heart_rate(signal, Fs, min_hr=40., max_hr=180., method='fast_ideal'):

    if method == 'ideal':
        """ Zero-pad in time domain for ideal interp in freq domain
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, Fs, zero_pad=100)
        cs = Akima1DInterpolator(freqs, ps)
        max_val = -np.Inf
        interval = 0.1
        min_bound = max(min(freqs), min_hr)
        max_bound = min(max(freqs), max_hr) + interval
        for bpm in np.arange(min_bound, max_bound, interval):
            cur_val = cs(bpm)
            if cur_val > max_val:
                max_val = cur_val
                max_bpm = bpm
        return max_bpm

    elif method == 'fast_ideal':
        """ Zero-pad in time domain for ideal interp in freq domain
        """
        signal = signal - np.mean(signal)
        freqs, ps = compute_power_spectrum(signal, Fs, zero_pad=100)
        freqs_valid = np.logical_and(freqs >= min_hr, freqs <= max_hr)
        freqs = freqs[freqs_valid]
        ps = ps[freqs_valid]
        max_ind = np.argmax(ps)
        if 0 < max_ind < len(ps)-1:
            inds = [-1, 0, 1] + max_ind
            x = ps[inds]
            f = freqs[inds]
            d1 = x[1]-x[0]
            d2 = x[1]-x[2]
            offset = (1 - min(d1,d2)/max(d1,d2)) * (f[1]-f[0])
            if d2 > d1:
                offset *= -1
            max_bpm = f[1] + offset
        elif max_ind == 0:
            x0, x1 = ps[0], ps[1]
            f0, f1 = freqs[0], freqs[1]
            max_bpm = f0 + (x1 / (x0 + x1)) * (f1 - f0)
        elif max_ind == len(ps) - 1:
            x0, x1 = ps[-2], ps[-1]
            f0, f1 = freqs[-2], freqs[-1]
            max_bpm = f0 + (x1 / (x0 + x1)) * (f1 - f0)
        return max_bpm

class FRL(nn.Module):
    def __init__(self,Fs,min_hr ,max_hr):
        super(FRL, self).__init__()
        self.Fs=Fs
        self.min_hr=min_hr
        self.max_hr=max_hr

    def forward(self, neg_rppgarr,pos_rppg1,pos_rppg2,ratio_array):
        loss=0
        count=0
        # rppg B,T
        # ratio_array B,K,T
        for j in range(pos_rppg1.shape[0]):
            poshr1= predict_heart_rate(pos_rppg1[j].detach().cpu().numpy(),self.Fs,self.min_hr,self.max_hr)

            poshr2= predict_heart_rate(pos_rppg2[j].detach().cpu().numpy(),self.Fs,self.min_hr,self.max_hr)
            for i in range(len(neg_rppgarr)):
                # neg_rppgarr K,B,T
                neghr=predict_heart_rate(neg_rppgarr[i][j].detach().cpu().numpy(),self.Fs,self.min_hr,self.max_hr)
                loss+=np.abs(neghr/poshr1-ratio_array[j][i][0].detach().cpu().numpy())+\
                      np.abs(neghr/poshr2-ratio_array[j][i][0].detach().cpu().numpy())
                count+=2
        loss=loss/count
        return loss/pos_rppg1.shape[0]

class FAL(nn.Module):
    def __init__(self, Fs, high_pass=2.5, low_pass=0.4):
        super(FAL, self).__init__()
        #PSD_MSE
        self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)
        self.distance_func = nn.MSELoss()

    def forward(self, pos_rppg1,pos_rppg2,neighbor_rppg1,neighbor_rppg2,neighbor_rppg3):
        posfre1= self.norm_psd(pos_rppg1)
        posfre2= self.norm_psd(pos_rppg2)
        neifre1=self.norm_psd(neighbor_rppg1)
        neifre2=self.norm_psd(neighbor_rppg2)
        neifre3=self.norm_psd(neighbor_rppg3)

        loss = self.distance_func(posfre1, neifre1)+self.distance_func(posfre1, neifre2)+self.distance_func(posfre1, neifre3)\
               +self.distance_func(posfre2, neifre1)+self.distance_func(posfre2, neifre2)+self.distance_func(posfre2, neifre3)
        loss=loss/6
        return loss

class FCL(nn.Module):
    def __init__(self, Fs, high_pass=2.5, low_pass=0.4,tau=0.08):
        super(FCL, self).__init__()
        #PSD_MSE
        self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)
        self.distance_func = nn.MSELoss()
        self.tau=tau

    def forward(self, neg_rppgarr,pos_rppg1,pos_rppg2):

        posfre1= self.norm_psd(pos_rppg1)
        posfre2= self.norm_psd(pos_rppg2)
        pos_dis=torch.exp(self.distance_func(posfre1, posfre2)/self.tau)
        neg_dis_total=0
        # neg_rppgarr K,B,T
        for i in range(len(neg_rppgarr)):
            negfre=self.norm_psd(neg_rppgarr[i])
            neg_dis = torch.exp(self.distance_func(posfre1, negfre) / self.tau)+torch.exp(self.distance_func(posfre2, negfre) / self.tau)
            neg_dis_total+=neg_dis

        loss = torch.log10(pos_dis/neg_dis_total+1)
        return loss
