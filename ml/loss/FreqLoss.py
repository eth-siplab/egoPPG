from __future__ import print_function, division
import math
import torch
import torch.linalg as linalg
from torch import nn


def calculate_2_wasserstein_dist(X, Y):
    '''
    Calulates the two components of the 2-Wasserstein metric:
    The general formula is given by: d(P_X, P_Y) = min_{X, Y} E[|X-Y|^2]
    For multivariate gaussian distributed inputs z_X ~ MN(mu_X, cov_X) and z_Y ~ MN(mu_Y, cov_Y),
    this reduces to: d = |mu_X - mu_Y|^2 - Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    Fast method implemented according to following paper: https://arxiv.org/pdf/2009.14075.pdf
    Input shape: [b, n] (e.g. batch_size x num_features)
    Output shape: scalar
    '''

    if X.shape != Y.shape:
        raise ValueError("Expecting equal shapes for X and Y!")

    # the linear algebra ops will need some extra precision -> convert to double
    X, Y = X.transpose(0, 1).double(), Y.transpose(0, 1).double()  # [n, b]
    mu_X, mu_Y = torch.mean(X, dim=1, keepdim=True), torch.mean(Y, dim=1, keepdim=True)  # [n, 1]
    n, b = X.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)

    # Cov. Matrix
    E_X = X - mu_X
    E_Y = Y - mu_Y
    cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
    cov_Y = torch.matmul(E_Y, E_Y.t()) * fact

    # calculate Tr((cov_X * cov_Y)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
    # The eigenvalues for M are real-valued.
    C_X = E_X * math.sqrt(fact)  # [n, n], "root" of covariance
    C_Y = E_Y * math.sqrt(fact)
    M_l = torch.matmul(C_X.t(), C_Y)
    M_r = torch.matmul(C_Y.t(), C_X)
    M = torch.matmul(M_l, M_r)
    S = linalg.eigvals(M) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

    # |mu_X - mu_Y|^2
    diff = mu_X - mu_Y  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together
    return (trace_term + mean_term).float()


class FreqLoss(nn.Module):
    def __init__(self, signal_type, fs, device):
        super(FreqLoss, self).__init__()
        if signal_type == 'ppg':
            L = 512
        elif signal_type == 'rr':
            L = 512
        else:
            raise ValueError('Invalid signal type')
        self.freq = fs * (torch.fft.rfftfreq(n=L).cuda(device))
        if signal_type == 'ppg':
            min_hr, max_hr = 42, 180
            loc_0, loc_1 = (torch.where((self.freq > min_hr / 60))[0][0].item(),
                            torch.where((self.freq > max_hr / 60))[0][0].item())
        elif signal_type == 'rr':
            min_rr, max_rr = 9, 36
            loc_0, loc_1 = (torch.where((self.freq > min_rr / 60))[0][0].item(),
                            torch.where((self.freq > max_rr / 60))[0][0].item())
        else:
            raise ValueError('Invalid signal type')
        self.L = L
        self.loc_0 = loc_0
        self.loc_1 = loc_1
        self.fs = fs
        self.device = device
        return

    def forward(self, pred, label):
        x_fft = torch.abs(torch.fft.rfft(pred[:, None, :], n=self.L, norm='forward'))
        org_fft = torch.abs(torch.fft.rfft(label[:, None, :], n=self.L, norm='forward'))
        # freq = self.fs * (torch.fft.rfftfreq(n=self.L).cuda(self.device))

        """import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(x_fft[0, 0, :].cpu().detach().numpy())
        fig.show()

        fig, ax = plt.subplots()
        ax.plot(pred[0, :].cpu().detach().numpy())
        fig.show()"""

        # Loss term l1: sum over frequencies outside the band of interest.
        l1 = torch.sum((torch.sum(x_fft[:, :, 0:self.loc_0], dim=2).squeeze() +
                        torch.sum(x_fft[:, :, self.loc_1:], dim=2).squeeze()), dim=0)

        # Loss term l2: Entropy-based loss for the predicted distribution.
        freq_interest = x_fft[:, :, self.loc_0:self.loc_1] / torch.sum(x_fft[:, :, self.loc_0:self.loc_1], axis=2, keepdim=True)
        freq_interest_org = org_fft[:, :, self.loc_0:self.loc_1] / torch.sum(org_fft[:, :, self.loc_0:self.loc_1], axis=2, keepdim=True)
        l2 = torch.sum(-torch.sum(freq_interest * torch.log(freq_interest), dim=2), dim=0)

        # --- NEW: Additional spectral MSE term with frequency-dependent weighting ---
        # This term enforces that errors in higher frequencies (which correspond to higher HR)
        # are penalized more. We create a weighting factor that increases with frequency.
        # freq_band = self.freq[self.loc_0:self.loc_1]  # frequencies in band of interest.
        # Normalize and square to emphasize higher frequencies.
        # f_weight = (freq_band / freq_band[-1]) ** 2
        # f_weight = f_weight.unsqueeze(0).unsqueeze(0)  # reshape to [1, 1, num_bins]

        # Loss term l3: KL divergence between the predicted and target spectral distributions.
        kl_loss = nn.KLDivLoss(reduction='sum')
        l3 = kl_loss(torch.log(freq_interest), freq_interest_org)
        # l3 = kl_loss(torch.log(f_weight*freq_interest), f_weight*freq_interest_org)

        # l4 is the weighted mean squared error between predicted and target FFT magnitudes
        # l4 = torch.mean(f_weight * (x_fft[:, :, self.loc_0:self.loc_1] - org_fft[:, :, self.loc_0:self.loc_1]) ** 2)

        # peak_locs = torch.argmax(x_fft[:, 0, self.loc_0:self.loc_1], dim=1)

        mse = torch.nn.functional.mse_loss(pred, label)
        w = calculate_2_wasserstein_dist(x_fft[:, 0, :], org_fft[:, 0, :])
        # w = calculate_2_wasserstein_dist(x_fft[:, 0, :] / torch.sum(x_fft[:, 0, :], axis=1, keepdim=True), org_fft[:, 0, :] / torch.sum(org_fft[:, 0, :], axis=1, keepdim=True))

        # return mse + w
        return mse + 0.5 * (l2 + l3)
        # return mse
