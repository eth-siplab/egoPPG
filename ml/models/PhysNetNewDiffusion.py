import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from source.ml.models.PhysNetNew import PhysNetNew


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
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb_factor)
        # t is [B], so t.unsqueeze(1) becomes [B,1], and emb.unsqueeze(0) becomes [1, half_dim]
        emb = t.unsqueeze(1) * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [B, d_model]
        return self.linear(emb)  # [B, d_model]


#############################################
#         Denoiser Architectures            #
#############################################
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
        t_emb = self.time_embed(t)                    # [B, 32]
        t_emb = t_emb.unsqueeze(2).expand(-1, -1, x.shape[2])  # [B, 32, L]
        t_emb_proj = self.t_proj(t_emb)               # [B, 1, L]
        x_cond = x + t_emb_proj
        return self.conv(x_cond)


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


#############################################
#      End-to-End Joint Diffusion Models    #
#############################################
class PhysNetDiffusionBase(nn.Module):
    """
    Base joint model that uses PhysNetNew to generate a raw rPPG signal,
    then applies a reverse diffusion loop using a provided denoiser.
    """
    def __init__(self, diffusion_steps=50, frames=128, device='cpu'):
        super(PhysNetDiffusionBase, self).__init__()
        self.physnet = PhysNetNew(frames=frames)  # Adjust frames if needed
        self.diffusion_steps = diffusion_steps
        beta_start, beta_end = 1e-4, 0.02
        self.register_buffer("betas", torch.linspace(beta_start, beta_end, diffusion_steps))
        self.device = device

    def forward(self, x, denoiser, compute_diff_loss=False):
        """
        x: Input images for PhysNetNew.
        denoiser: A diffusion denoiser module (expects: noisy signal [B,1,L] and t [B]).
        compute_diff_loss: If True, the function returns a tuple (refined_signal, diffusion_loss).
        """
        # 1. Obtain clean signal from PhysNetNew.
        raw_signal = self.physnet(x)  # [B, L]
        x0 = raw_signal.unsqueeze(1)  # [B, 1, L]

        # 2. Simulate forward diffusion (adding noise).
        batch_size = x0.size(0)
        # For direct diffusion supervision, sample a random timestep for each sample.
        t_rand = torch.randint(0, self.diffusion_steps, (batch_size,), dtype=torch.float32, device=self.device)
        alpha_bars = torch.cumprod(1 - self.betas, dim=0)  # [diffusion_steps]
        alpha_bar_t = alpha_bars[t_rand.long()].view(-1, 1, 1)
        noise = torch.randn_like(x0)
        x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

        # 3. Reverse diffusion loop: here, we run the loop for all diffusion steps.
        x_signal = x_t
        # For diffusion loss, we can (for example) compute loss at the first reverse step.
        diff_loss = 0.0
        for t in reversed(range(self.diffusion_steps)):
            t_tensor = torch.full((x_signal.size(0),), t, dtype=torch.float32,
                                  device=self.device) / self.diffusion_steps
            predicted_noise = denoiser(x_signal, t_tensor)
            beta = self.betas[t]
            # Option: Compute diffusion loss only at a particular step (say, the first reverse step)
            if compute_diff_loss and t == (self.diffusion_steps - 1):
                diff_loss = nn.functional.mse_loss(predicted_noise, noise)
            x_signal = (x_signal - beta * predicted_noise) / torch.sqrt(1 - beta)
        refined_signal = x_signal.squeeze(1)  # [B, L]

        if compute_diff_loss:
            return refined_signal, diff_loss
        else:
            return refined_signal


# Joint model variant with CNN denoiser
class PhysNetDiffusionCNN(nn.Module):
    def __init__(self, diffusion_steps=50, frames=128, device='cpu'):
        super(PhysNetDiffusionCNN, self).__init__()
        self.base = PhysNetDiffusionBase(diffusion_steps=diffusion_steps, device=device, frames=frames)
        self.cnn_denoiser = CNNDenoiser()

    def forward(self, x, compute_diff_loss=False):
        return self.base(x, self.cnn_denoiser, compute_diff_loss=compute_diff_loss)


# Joint model variant with U-Net denoiser
class PhysNetDiffusionUNet(nn.Module):
    def __init__(self, diffusion_steps=50, frames=128, device='cpu'):
        super(PhysNetDiffusionUNet, self).__init__()
        self.base = PhysNetDiffusionBase(diffusion_steps=diffusion_steps, device=device, frames=frames)
        self.unet_denoiser = UNetDenoiser()

    def forward(self, x, compute_diff_loss=False):
        return self.base(x, self.unet_denoiser, compute_diff_loss=compute_diff_loss)


# Joint model variant with Transformer denoiser
class PhysNetDiffusionTransformer(nn.Module):
    def __init__(self, diffusion_steps=50, frames=128, device='cpu'):
        super(PhysNetDiffusionTransformer, self).__init__()
        self.base = PhysNetDiffusionBase(diffusion_steps=diffusion_steps, device=device, frames=frames)
        self.transformer_denoiser = TransformerDenoiserWrapper(frames=frames)

    def forward(self, x, compute_diff_loss=False):
        return self.base(x, self.transformer_denoiser, compute_diff_loss=compute_diff_loss)
