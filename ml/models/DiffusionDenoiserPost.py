import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        # Create sinusoidal embeddings (like in Transformers)
        half_dim = self.d_model // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb_factor)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)  # [batch, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [batch, d_model]
        return self.linear(emb)  # [batch, d_model]

#############################################
#              U-NET Modules                #
#############################################

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
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            DoubleConv1D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

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
        x1 = self.inc(x)         # [B, base_features, L]
        x2 = self.down1(x1)      # [B, base_features*2, L/2]
        x3 = self.down2(x2)      # [B, base_features*4, L/4]
        x4 = self.down3(x3)      # [B, base_features*8, L/8]
        x5 = self.down4(x4)      # [B, base_features*16, L/16]
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

#############################################
#       Post-Processing Diffusion Models    #
#############################################

class PostProcessingDiffusionUNet(nn.Module):
    """
    Post-processing diffusion denoiser using a U-Net with time conditioning.
    """
    def __init__(self, frames=128, diffusion_steps=50, device='cpu'):
        super(PostProcessingDiffusionUNet, self).__init__()
        self.diffusion_steps = diffusion_steps
        self.unet = UNet1D(n_channels=1, n_classes=1, bilinear=True, base_features=64)
        self.time_embed = TimeEmbedding(d_model=64)
        self.t_proj = nn.Conv1d(64, 1, kernel_size=1)
        self.device = device

    def forward(self, x, t):
        """
        x: [batch, 1, signal_length]
        t: [batch] (normalized time values between 0 and 1)
        """
        # Compute time embedding and project it to one channel
        t_emb = self.time_embed(t)                   # [batch, 64]
        t_emb = t_emb.unsqueeze(2).expand(-1, -1, x.shape[2])  # [batch, 64, signal_length]
        t_emb_proj = self.t_proj(t_emb)                # [batch, 1, signal_length]
        # Add the time conditioning to the input signal
        conditioned_input = x + t_emb_proj
        # Pass through U-Net
        out = self.unet(conditioned_input)
        return out


#############################################
#       Transformer Denoiser Modules        #
#############################################

class TransformerDenoiser(nn.Module):
    """
    A simple Transformer-based denoiser for 1D signals.
    """
    def __init__(self, input_dim=1, d_model=64, nhead=8, num_layers=4, dropout=0.1, signal_length=128):
        super(TransformerDenoiser, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        # Learned positional embeddings (shape: [1, signal_length, d_model])
        self.pos_emb = nn.Parameter(torch.zeros(1, signal_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        """
        x: [batch, 1, signal_length]
        """
        batch, _, L = x.shape
        # Permute to [batch, signal_length, 1]
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)              # [batch, signal_length, d_model]
        x = x + self.pos_emb                # Add positional encoding
        # Transformer expects input shape: [seq_len, batch, d_model]
        x = x.transpose(0, 1)               # [signal_length, batch, d_model]
        x = self.transformer_encoder(x)     # [signal_length, batch, d_model]
        x = x.transpose(0, 1)               # [batch, signal_length, d_model]
        x = self.output_proj(x)             # [batch, signal_length, input_dim]
        x = x.permute(0, 2, 1)              # [batch, input_dim, signal_length]
        return x


class PostProcessingDiffusionTransformer(nn.Module):
    """
    Post-processing diffusion denoiser using a Transformer with time conditioning.
    """
    def __init__(self, frames=128, diffusion_steps=50, transformer_params=None, device='cpu'):
        super(PostProcessingDiffusionTransformer, self).__init__()
        if transformer_params is None:
            transformer_params = {
                'input_dim': 1,
                'd_model': 64,
                'nhead': 8,
                'num_layers': 4,
                'dropout': 0.1,
                'signal_length': frames
            }
        self.transformer = TransformerDenoiser(**transformer_params)
        self.diffusion_steps = diffusion_steps
        self.time_embed = TimeEmbedding(d_model=64)
        self.t_proj = nn.Conv1d(64, 1, kernel_size=1)
        self.device = device

    def forward(self, x, t):
        """
        x: [batch, 1, signal_length]
        t: [batch] (normalized time values between 0 and 1)
        """
        # Compute time embedding and project it to one channel
        t_emb = self.time_embed(t)                   # [batch, 64]
        t_emb = t_emb.unsqueeze(2).expand(-1, -1, x.shape[2])  # [batch, 64, signal_length]
        t_emb_proj = self.t_proj(t_emb)                # [batch, 1, signal_length]
        # Add time conditioning to input signal
        conditioned_input = x + t_emb_proj
        # Pass through Transformer-based denoiser
        out = self.transformer(conditioned_input)
        return out


#############################################
#       CNN Denoiser Modules        #
#############################################

class DiffusionDenoiserPost(nn.Module):
    """
    A simple 1D convolutional network to predict noise at a given diffusion timestep.
    """

    def __init__(self, num_steps=50):
        super(DiffusionDenoiserPost, self).__init__()
        self.num_steps = num_steps
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x, t):
        """
        x: input signal of shape [batch, 1, signal_length]
        t: tensor of shape [batch] with normalized timestep (between 0 and 1)
        """
        t_scaled = t.view(-1, 1, 1)  # shape: [batch, 1, 1]
        conditioned_input = x + t_scaled  # very basic conditioning on time
        predicted_noise = self.net(conditioned_input)
        return predicted_noise


def diffusion_denoising(diffusion_model, noisy_signal, num_steps=50, device='cpu'):
    """
    Reverse diffusion process to denoise a 1D signal.

    Args:
        diffusion_model: The DiffusionDenoiser model.
        noisy_signal: Tensor of shape [batch, 1, signal_length].
        num_steps: Number of diffusion steps.
        device: 'cpu' or 'cuda'.

    Returns:
        Denoised signal tensor.
    """
    # Define a simple linear beta schedule
    beta_start = 1e-4
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, num_steps).to(device)

    x = noisy_signal.to(device)

    # Reverse diffusion loop
    for t in reversed(range(num_steps)):
        # Create a time tensor normalized between 0 and 1.
        t_tensor = torch.full((x.size(0),), t, dtype=torch.float32, device=device) / num_steps
        # Predict the noise in the current state.
        predicted_noise = diffusion_model(x, t_tensor)
        beta = betas[t]
        # A simplified reverse step.
        x = (x - beta * predicted_noise) / torch.sqrt(1 - beta)
        if t > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(beta) * noise
    return x
