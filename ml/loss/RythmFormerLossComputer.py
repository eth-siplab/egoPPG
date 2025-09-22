'''
  Adapted from here: https://github.com/ZitongYu/PhysFormer/TorchLossComputer.py
  Modifed based on the HR-CNN here: https://github.com/radimspetlik/hr-cnn
'''
import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from source.evaluation.post_process import calculate_metric_per_video_ppg, calculate_metric_per_video_rr


def normal_sampling(mean, label_k, std):
    return math.exp(-(label_k - mean) ** 2 / (2 * std ** 2)) / (math.sqrt(2 * math.pi) * std)


def kl_loss(inputs, labels):
    criterion = nn.KLDivLoss(reduction='batchmean')
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    # loss = loss.sum()/loss.shape[0]
    loss = loss.sum()
    return loss


class Neg_Pearson(nn.Module):  # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson, self).__init__()

    def forward(self, preds, labels):  # all variable operation
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])  # x
            sum_y = torch.sum(labels[i])  # y
            sum_xy = torch.sum(preds[i] * labels[i])  # xy
            sum_x2 = torch.sum(torch.pow(preds[i], 2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i], 2))  # y^2
            N = preds.shape[1]
            pearson = (N * sum_xy - sum_x * sum_y) / (
                torch.sqrt((N * sum_x2 - torch.pow(sum_x, 2)) * (N * sum_y2 - torch.pow(sum_y, 2))))
            loss += 1 - pearson

        loss = loss / preds.shape[0]
        return loss


class RhythmFormer_Loss(nn.Module):
    def __init__(self, signal_name):
        super(RhythmFormer_Loss, self).__init__()
        self.criterion_Pearson = Neg_Pearson()
        self.signal_name = signal_name

    def forward(self, pred_ppg, labels, epoch, FS, diff_flag, device, min_hr_rr, max_hr_rr):
        loss_time = self.criterion_Pearson(pred_ppg.view(1, -1), labels.view(1, -1))
        loss_CE, loss_distribution_kl = TorchLossComputer.Frequency_loss(
            pred_ppg.squeeze(-1), labels.squeeze(-1), device, diff_flag, FS, 3.0, min_hr_rr, max_hr_rr,
            self.signal_name)
        loss_hr = TorchLossComputer.HR_loss(pred_ppg.squeeze(-1), labels.squeeze(-1), device, diff_flag, FS, 3.0,
                                            self.signal_name)
        if torch.isnan(loss_time):
            loss_time = 0

        loss = 0.2 * loss_time + 1.0 * loss_CE + 1.0 * loss_hr
        return loss


class TorchLossComputer(object):
    @staticmethod
    def compute_complex_absolute_given_k(output, k, N, device):
        two_pi_n_over_N = Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True) / N
        hanning = Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        k = k.type(torch.FloatTensor).to(device)
        two_pi_n_over_N = two_pi_n_over_N.to(device)
        hanning = hanning.to(device)

        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

        return complex_absolute

    @staticmethod
    def complex_absolute(output, Fs, device, bpm_range=None):
        output = output.view(1, -1)

        N = output.size()[1]

        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz

        # only calculate feasible PSD range [0.7,4]Hz
        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k(output, k, N, device)

        return (1.0 / complex_absolute.sum()) * complex_absolute  # Analogous Softmax operator

    @staticmethod
    def cross_entropy_power_spectrum_loss(inputs, target, Fs, min_hr_rr, max_hr_rr):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(min_hr_rr, max_hr_rr, dtype=torch.float).cuda()
        # bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        # pdb.set_trace()

        # return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
        return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)), torch.abs(
            target[0] - whole_max_idx)

    @staticmethod
    def cross_entropy_power_spectrum_focal_loss(inputs, target, Fs, gamma, min_hr_rr, max_hr_rr):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(min_hr_rr, max_hr_rr, dtype=torch.float).cuda()
        # bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        # pdb.set_trace()
        criterion = FocalLoss(gamma=gamma)

        # return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
        return criterion(complex_absolute, target.view((1)).type(torch.long)), torch.abs(target[0] - whole_max_idx)

    @staticmethod
    def cross_entropy_power_spectrum_forward_pred(inputs, Fs, min_hr_rr, max_hr_rr):
        inputs = inputs.view(1, -1)
        bpm_range = torch.arange(min_hr_rr, max_hr_rr, dtype=torch.float).cuda()
        # bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        # bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        return whole_max_idx

    @staticmethod
    def Frequency_loss(inputs, target, device, diff_flag, Fs, std, min_hr_rr, max_hr_rr, signal_name):
        if signal_name == 'ppg':
            hr_gt, pred_hr_peak, SNR, macc = calculate_metric_per_video_ppg(
                inputs.detach().cpu(), target.detach().cpu(), diff_flag=diff_flag, fs=Fs, hr_method='Peak_Detection')
        else:
            hr_gt, pred_hr_peak, SNR, macc = calculate_metric_per_video_rr(
                inputs.detach().cpu(), target.detach().cpu(), diff_flag=diff_flag, fs=Fs, hr_method='Peak_Detection')
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(min_hr_rr, max_hr_rr, dtype=torch.float).to(device)
        ca = TorchLossComputer.complex_absolute(inputs, Fs, device, bpm_range)
        sa = ca / torch.sum(ca)

        target_distribution = [normal_sampling(int(hr_gt), i, std) for i in range(min_hr_rr, max_hr_rr)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(device)

        hr_gt = torch.tensor(hr_gt - min_hr_rr).view(1).type(torch.long).to(device)
        return F.cross_entropy(ca, hr_gt), kl_loss(sa, target_distribution)

    @staticmethod
    def HR_loss(inputs, target, device, diff_flag, Fs, std, signal_name):
        if signal_name == 'ppg':
            psd_gt, psd_pred, SNR, macc = calculate_metric_per_video_ppg(
                inputs.detach().cpu(), target.detach().cpu(), diff_flag=diff_flag, fs=Fs, hr_method='Peak_Detection')
        else:
            psd_gt, psd_pred, SNR, macc = calculate_metric_per_video_rr(
                inputs.detach().cpu(), target.detach().cpu(), diff_flag=diff_flag, fs=Fs, hr_method='Peak_Detection')
        pred_distribution = [normal_sampling(np.argmax(psd_pred), i, std) for i in range(psd_pred.size)]
        pred_distribution = [i if i > 1e-15 else 1e-15 for i in pred_distribution]
        pred_distribution = torch.Tensor(pred_distribution).to(device)
        target_distribution = [normal_sampling(np.argmax(psd_gt), i, std) for i in range(psd_gt.size)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(device)
        return kl_loss(pred_distribution, target_distribution)
