import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from source.ml.models.PhysNetNewIMUCA import PhysNetNewIMUCA

from source.utils import quantile_artifact_removal_multi


#############################################
#         Time Embedding Module             #
#############################################
class TimeEmbedding(nn.Module):
    """
    Creates a sinusoidal time embedding and projects it.
    """
    def __init__(self, d_model):
        super(TimeEmbedding, self).__init__()
        self.d_model = d_model
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, t):
        # t: [batch]
        half_dim = self.d_model // 2
        emb_factor = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * - emb_factor)
        # t is [B], so t.unsqueeze(1) becomes [B,1], and emb.unsqueeze(0) becomes [1, half_dim]
        emb = t.unsqueeze(1) * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [B, d_model]
        return self.linear(emb)  # [B, d_model]


#############################################
#         Denoiser Architectures            #
#############################################
# 2. U-Net Denoiser with Time Conditioning
# -- U-Net building blocks for 1D signals
class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv1D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down1D, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up1D(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up1D, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv1D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff = x2.size(2) - x1.size(2)
        if diff > 0:
            x1 = F.pad(x1, (0, diff))
        elif diff < 0:
            x2 = F.pad(x2, (0, -diff))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet1D(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, base_features=64):
        super(UNet1D, self).__init__()
        self.inc = DoubleConv1D(n_channels, base_features)
        self.down1 = Down1D(base_features, base_features * 2)
        self.down2 = Down1D(base_features * 2, base_features * 4)
        self.down3 = Down1D(base_features * 4, base_features * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down1D(base_features * 8, base_features * 16 // factor)
        self.up1 = Up1D(base_features * 16, base_features * 8 // factor, bilinear)
        self.up2 = Up1D(base_features * 8, base_features * 4 // factor, bilinear)
        self.up3 = Up1D(base_features * 4, base_features * 2 // factor, bilinear)
        self.up4 = Up1D(base_features * 2, base_features, bilinear)
        self.outc = nn.Conv1d(base_features, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# U-Net denoiser with time conditioning:
class UNetDenoiser(nn.Module):
    def __init__(self):
        super(UNetDenoiser, self).__init__()
        self.unet = UNet1D(n_channels=1, n_classes=1, bilinear=True, base_features=64)
        self.time_embed = TimeEmbedding(d_model=64)
        self.t_proj = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x, t):
        # x: [B, 1, L], t: [B]
        t_emb = self.time_embed(t)                          # [B, 64]
        t_emb = t_emb.unsqueeze(2).expand(-1, -1, x.shape[2]) # [B, 64, L]
        t_emb_proj = self.t_proj(t_emb)                       # [B, 1, L]
        x_cond = x + t_emb_proj
        return self.unet(x_cond)


# 3. Transformer Denoiser with Time Conditioning
class TransformerDenoiser(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=8, num_layers=4, dropout=0.1, frames=128):
        super(TransformerDenoiser, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        # Learned positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, frames, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x: [B, 1, L]
        batch, _, L = x.shape
        x = x.permute(0, 2, 1)            # [B, L, 1]
        x = self.input_proj(x)            # [B, L, d_model]
        x = x + self.pos_emb              # add positional encoding [B, L, d_model]
        x = x.transpose(0, 1)             # [L, B, d_model] for transformer
        x = self.transformer_encoder(x)   # [L, B, d_model]
        x = x.transpose(0, 1)             # [B, L, d_model]
        x = self.output_proj(x)           # [B, L, 1]
        x = x.permute(0, 2, 1)            # [B, 1, L]
        return x


class TransformerDenoiserWrapper(nn.Module):
    def __init__(self, frames=128):
        super(TransformerDenoiserWrapper, self).__init__()
        self.transformer = TransformerDenoiser(input_dim=1, d_model=64, nhead=8,
                                                num_layers=4, dropout=0.1, frames=frames)
        self.time_embed = TimeEmbedding(d_model=64)
        self.t_proj = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, x, t):
        # x: [B, 1, L], t: [B]
        t_emb = self.time_embed(t)                          # [B, 64]
        t_emb = t_emb.unsqueeze(2).expand(-1, -1, x.shape[2]) # [B, 64, L]
        t_emb_proj = self.t_proj(t_emb)                       # [B, 1, L]
        x_cond = x + t_emb_proj
        return self.transformer(x_cond)


# 1. Simple CNN Denoiser with Time Conditioning
class CNNDenoiser(nn.Module):
    def __init__(self):
        super(CNNDenoiser, self).__init__()
        self.time_embed = TimeEmbedding(d_model=32)
        self.t_proj = nn.Conv1d(32, 1, kernel_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        # x: [B, 1, L], t: [B]
        # t_emb = self.time_embed(t)                    # [B, 32]
        # t_emb = t_emb.unsqueeze(2).expand(-1, -1, x.shape[2])  # [B, 32, L]
        # t_emb_proj = self.t_proj(t_emb)               # [B, 1, L]
        # x_cond = x + t_emb_proj
        x_cond = x
        return self.conv(x_cond)


def linear_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    """Original DDPM linear schedule."""
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T, s=0.008):
    """
    Nichol & Dhariwal 2021.
    The cosine schedule is defined on ᾱₜ, then converted to βₜ.
    """
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_bar = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]          # normalise to 1 at t=0
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])   # βₜ = 1 - ᾱₜ / ᾱ_{t-1}
    return betas.clamp(1e-8, 0.999)


#############################################
#      End-to-End Joint Diffusion Models    #
#############################################
class DiffusionPhys(nn.Module):
    """
    Two-stage model:
      1) PhysNetNewIMUCA → raw rPPG x₀                (shape [B, L])
      2) DDPM reverse process (CNNDenoiser) → refined  (shape [B, L])
    """

    def __init__(self, diffusion_steps=50, device="cpu", beta_schedule="cosine"):
        super().__init__()
        self.T       = diffusion_steps
        self.denoiser = CNNDenoiser()
        # self.denoiser = UNetDenoiser()

        # β schedule (linear); you can swap for cosine later
        beta_start, beta_end = 1e-4, 2e-2
        betas = torch.linspace(beta_start, beta_end, self.T)           # [T]
        alphas = 1.0 - betas                                           # [T]

        if beta_schedule == "linear":
            betas = linear_beta_schedule(self.T)
        else:  # default = cosine
            betas = cosine_beta_schedule(self.T)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1 - betas, 0))
        # exact posterior variance 𝛃̃ₜ = (1-ᾱ_{t-1})/(1-ᾱₜ)·βₜ
        posterior_var = betas * (1.0 - torch.cat([torch.ones(1), self.alphas_cumprod[:-1]])) \
                               / (1.0 - self.alphas_cumprod)
        self.register_buffer("posterior_var", posterior_var.clamp(min=1e-20))

        self.device = torch.device(device)

    def _q_sample(self, x0, t, noise):
        """
        Forward-diffuse x₀ → xₜ at timestep t  (vectorised for a batch)
        """
        a_bar = self.alphas_cumprod[t].view(-1, 1, 1)          # [B,1,1]
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

    def forward(self, vid, compute_diff_loss=False, signal_norm=True, diff_loss_weight=1.0):
        """
        Returns:
            refined_signal               – always
            diffusion_loss (optional)    – only if compute_diff_loss=True
        """
        # Mean video
        raw = torch.mean(vid, dim=(1, 3, 4))
        # raw = quantile_artifact_removal_multi(raw.cpu(), 0.25, 0.75, 3).to(self.device)

        if signal_norm:
            mu  = raw.mean(dim=1, keepdim=True)
            std = raw.std (dim=1, keepdim=True).clamp(min=1e-6)
            raw_n = (raw - mu) / std           # roughly in N(0,1)
        else:
            raw_n = raw
        x0 = raw_n.unsqueeze(1)                # [B,1,L]


        x_sig = self.denoiser(x0, None)
        diff_loss = 0

        # -------------------------------------------------------------- #
        # 1) Sample a timestep & add noise
        # -------------------------------------------------------------- #
        """B = x0.size(0)
        t_rand = torch.randint(0, self.T, (B,), device=self.device, dtype=torch.long)
        eps    = torch.randn_like(x0)
        x_t    = self._q_sample(x0, t_rand, eps)

        # -------------------------------------------------------------- #
        # 2) Compute ε-prediction loss (only if requested)
        # -------------------------------------------------------------- #
        if compute_diff_loss:
            eps_pred  = self.denoiser(x_t, t_rand)        # CNNDenoiser embeds t internally
            diff_loss = F.mse_loss(eps_pred, eps) * diff_loss_weight
        else:
            diff_loss = None

        # -------------------------------------------------------------- #
        # 3) Reverse loop (inference only)
        # -------------------------------------------------------------- #
        x_sig = x_t.detach()                         # avoid grads in reverse
        for t in reversed(range(self.T)):
            beta      = self.betas[t]
            alpha_t   = self.alphas[t]
            alpha_bar = self.alphas_cumprod[t]

            # predict noise ε̂
            t_vec   = torch.full((B,), t, device=self.device, dtype=torch.long)
            eps_hat = self.denoiser(x_sig, t_vec)

            # DDPM posterior mean μₜ(xₜ, ε̂)
            coef1 = 1.0 / torch.sqrt(alpha_t)
            coef2 = (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar)
            mean = coef1 * (x_sig - coef2 * eps_hat)

            if t > 0:
                noise = torch.randn_like(x_sig)
                sigma = torch.sqrt(self.posterior_var[t])
                x_sig = mean + sigma * noise
            else:
                x_sig = mean                       # final, deterministic"""

        refined = x_sig.squeeze(1)                 # [B,L]

        # de-normalise if we normalised earlier
        if signal_norm:
            refined = refined * std + mu

        return (refined, diff_loss) if compute_diff_loss else refined
