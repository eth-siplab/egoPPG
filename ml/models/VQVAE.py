import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# ---------------------------
# Helper functions and modules (same as in PhysNet)
# ---------------------------
def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1),
                          torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SpatialAttentionGate(nn.Module):
    def __init__(self):
        super(SpatialAttentionGate, self).__init__()
        self.compress = ChannelPool()
        kernel_size = 7
        self.spatial = BasicConv(2, 1, kernel_size, stride=1,
                                 padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return scale * x


# ---------------------------
# PhysNet Encoder (modified from your PhysNet)
# We use the convolutional backbone (up to pooling) as the encoder.
# ---------------------------
class PhysNetEncoder(nn.Module):
    def __init__(self, frames=128):
        super(PhysNetEncoder, self).__init__()
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(1, 16, (1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, (3, 3, 3), stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(64, 64, (4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, (4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))
        self.SpatialAttentionGate1 = SpatialAttentionGate()
        self.SpatialAttentionGate2 = SpatialAttentionGate()
        self.SpatialAttentionGate3 = SpatialAttentionGate()
        self.SpatialAttentionGate4 = SpatialAttentionGate()

    def forward(self, x):
        # x: [B, 1, T, H, W]
        [batch, channel, length, height, width] = x.shape
        x = self.ConvBlock1(x)
        x = self.SpatialAttentionGate1(x.view(batch * length, 16, height, width)).view(batch, 16, length, height, width)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.SpatialAttentionGate2(x.view(batch * length, 64, height // 2, width // 2)).view(batch, 64, length,
                                                                                                 height // 2,
                                                                                                 width // 2)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.SpatialAttentionGate3(x.view(batch * (length // 2), 64, height // 4, width // 4)).view(batch, 64,
                                                                                                        (length // 2),
                                                                                                        height // 4,
                                                                                                        width // 4)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        x = self.SpatialAttentionGate4(x.view(batch * (length // 4), 64, height // 8, width // 8)).view(batch, 64,
                                                                                                        (length // 4),
                                                                                                        height // 8,
                                                                                                        width // 8)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock8(x)
        x = self.ConvBlock9(x)
        x = self.upsample(x)
        x = self.upsample2(x)  # x [B, 64, T, 8, 8]
        x = self.poolspa(x)  # [B, 64, T, 1, 1]
        # Reshape to [B, T, 64]
        x = x.view(batch, 64, length).transpose(1, 2)  # [B, T, 64]
        return x


class AdaptiveVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.6):
        super(AdaptiveVectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Standard embedding lookup table.
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

        # Register the HR values associated with each code (uniformly spaced from 40 to 180)
        self.register_buffer('hr_values', torch.linspace(40, 180, steps=num_embeddings))
        # Learnable scaling factors to adjust the sensitivity for each embedding.
        self.scaling = nn.Parameter(torch.ones(num_embeddings))

    def forward(self, inputs):
        # inputs: [B, T, embedding_dim]
        flat_input = inputs.contiguous().view(-1, self.embedding_dim)  # [B*T, D]

        # Standard Euclidean distance calculation.
        distances = (flat_input.pow(2).sum(1, keepdim=True)
                     + self.embeddings.weight.pow(2).sum(1)
                     - 2 * torch.matmul(flat_input, self.embeddings.weight.t()))

        # Weight distances so that embeddings representing higher HR get a higher weight.
        mid_hr = 100.0  # Mid-point in the HR range.
        hr_range = 70.0  # A scaling factor (approximately half the overall range).
        hr_weights = 1.0 + self.scaling * ((self.hr_values - mid_hr) / hr_range)  # [num_embeddings]
        distances = distances * hr_weights.unsqueeze(0)  # Apply weights.

        # Find nearest embeddings.
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embeddings.weight).view_as(inputs)

        # Compute losses to update the encoder and the codebook.
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator.
        quantized = inputs + (quantized - inputs).detach()
        encoding_indices = encoding_indices.view(inputs.size(0), inputs.size(1))
        return quantized, loss, encoding_indices


# ---------------------------
# Vector Quantizer module (VQVAE)
# ---------------------------
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        """
        Args:
            num_embeddings: Number of latent codes (e.g., 141 for HR classes 40-180 BPM,
                            or more if you want finer resolution).
            embedding_dim: Dimensionality of latent vectors.
            commitment_cost: Weight for the commitment loss (training encoder).
        """
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim  # n channels of output of encoder
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)  # [num_embeddings, embedding_dim]
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, inputs):
        # inputs: [B, T, embedding_dim]
        flat_input = inputs.contiguous().view(-1, self.embedding_dim)  # [B*T, D]

        # Compute distances to embeddings
        distances = (flat_input.pow(2).sum(1, keepdim=True) + self.embeddings.weight.pow(2).sum(1) -
                     2 * torch.matmul(flat_input, self.embeddings.weight.t()))  # [B*T, num_embeddings]

        # Encoding indices: [B*T, 1]
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [B*T, 1]  class for each time from min dist.
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)  # [B*T, num_embeddings]
        encodings.scatter_(1, encoding_indices, 1)  # Just write 1 in each row at the index of the class.

        # Quantized at this stage means that each vector from the input is replaced with the weights of the closest
        # class in the codebook (closest embedding)
        # GOAL: ENCODER OUTPUT SHOULD BE QUANTIZED TO THE NEAREST VECTOR IN THE LEARNED CODEBOOK
        # Because if the encoder’s output is close to a code, the quantization step won’t change it very much,
        # and the network can reconstruct the input better.
        # This means, the encoder outputs a continuous latent vector that is then replaced by its nearest neighbor
        # from the discrete codebook (but argmin not differentiable)
        quantized = torch.matmul(encodings, self.embeddings.weight).view_as(inputs)

        # Compute losses
        # This loss only updates the encoder as the quantized outputs are not used for backpropagation
        # Updates the encoder to output vectors that are close to the codebook
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)

        # This loss only updates the codebook and not the encoder
        # Updates the codebook embeddings to that its vectors get closer to the encoder's outputs
        q_latent_loss = F.mse_loss(quantized, inputs.detach())

        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        # Basically, quantized=quantized but the gradient is passed directly back to the encoder during backprop
        # This means, whatever happens in here does not affect the gradient during training
        # *** This means, even though the encoder’s output is replaced with a discrete code in the forward pass,
        # the gradient flows back as if the encoder’s output had not been quantized at all ***
        # In this way, the gradient is not flowing back to the argmin operation
        # This also means that the encoder is updated based on the reconstruction loss
        quantized = inputs + (quantized - inputs).detach()
        encoding_indices = encoding_indices.view(inputs.size(0), inputs.size(1))
        return quantized, loss, encoding_indices



class UNetDecoder(nn.Module):
    def __init__(self, embedding_dim=64, out_length=256):
        """
        A U-Net style decoder upsamples in multiple stages with convolutional blocks.
        Note: If you have skip connections from the encoder, incorporate them accordingly.
        """
        super(UNetDecoder, self).__init__()
        # Up Block 1: Upsample from latent sequence resolution to 2x
        self.up1 = nn.ConvTranspose1d(embedding_dim, 128, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Up Block 2: Upsample to 4x the latent resolution
        self.up2 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Up Block 3: Upsample to 8x the latent resolution
        self.up3 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Final layer: Map to 1 channel and adjust length.
        self.final_conv = nn.Conv1d(32, 1, kernel_size=1)
        self.out_length = out_length

    def forward(self, x):
        # x: [B, T', embedding_dim]
        x = x.transpose(1, 2)  # -> [B, embedding_dim, T']
        x = self.up1(x)        # -> [B, 128, T1]
        x = self.conv1(x)
        x = self.up2(x)        # -> [B, 64, T2]
        x = self.conv2(x)
        x = self.up3(x)        # -> [B, 32, T3]
        x = self.conv3(x)
        out = self.final_conv(x)  # -> [B, 1, T3]
        # Interpolate to the desired output temporal length
        out = F.interpolate(out, size=self.out_length, mode='linear', align_corners=False)
        out = out.squeeze(1)  # -> [B, out_length]
        return out


class ResidualDilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super(ResidualDilatedBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ResidualDilatedDecoder(nn.Module):
    def __init__(self, embedding_dim=64, out_length=256):
        """
        A decoder that upsamples and refines features using residual dilated blocks.
        """
        super(ResidualDilatedDecoder, self).__init__()
        self.up1 = nn.ConvTranspose1d(embedding_dim, 64, kernel_size=4, stride=2, padding=1)
        self.resblock1 = ResidualDilatedBlock(64, dilation=1)
        self.up2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.resblock2 = ResidualDilatedBlock(32, dilation=2)
        self.conv_final = nn.Conv1d(32, 1, kernel_size=3, stride=1, padding=1)
        self.out_length = out_length

    def forward(self, x):
        # x: [B, T', embedding_dim]
        x = x.transpose(1, 2)  # -> [B, embedding_dim, T']
        x = self.up1(x)        # Upsample temporal dimension (e.g., T' -> 2T')
        x = self.resblock1(x)  # Refinement with local context
        x = self.up2(x)        # Further upsample (e.g., 2T' -> 4T')
        x = self.resblock2(x)  # Refinement with a larger receptive field (dilation=2)
        x = self.conv_final(x) # Map to 1 channel
        x = F.interpolate(x, size=self.out_length, mode='linear', align_corners=False)
        x = x.squeeze(1)       # Final shape: [B, out_length]
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim=64, num_layers=2, num_heads=4, out_length=256):
        """
        A Transformer decoder that creates a target sequence and decodes using self-attention.
        """
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=0.1, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, 1)
        self.out_length = out_length

    def forward(self, x):
        # x: [B, T', embedding_dim]
        B, _, _ = x.shape
        # Create a target sequence (could be zeros or learned embeddings)
        tgt = torch.zeros(B, self.out_length, x.size(-1), device=x.device)
        decoded = self.transformer_decoder(tgt, x)  # -> [B, out_length, embedding_dim]
        out = self.fc(decoded).squeeze(-1)          # -> [B, out_length]
        return out


# ---------------------------
# Decoder: Reconstruct continuous rPPG signal from quantized latent codes.
# Here we use a simple decoder with 1D transposed convolutions.
# ---------------------------
class Decoder(nn.Module):
    def __init__(self, embedding_dim=64, out_length=256):
        """
        Args:
            embedding_dim: Should match the encoder's latent dimension.
            out_length: The desired output signal length.
        """
        super(Decoder, self).__init__()
        # A simple design: use a few 1D transposed conv layers.
        self.decoder = nn.Sequential(
            # Input shape: [B, embedding_dim, T'] where T' is smaller than out_length.
            nn.ConvTranspose1d(embedding_dim, 32, kernel_size=4, stride=2, padding=1),  # doubles time dimension
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),  # doubles time dimension
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Assume the rPPG signal is normalized between -1 and 1.
        )
        self.out_length = out_length

    def forward(self, x):
        # x: [B, T', embedding_dim]
        # Rearrange to [B, embedding_dim, T']
        x = x.transpose(1, 2)
        x = self.decoder(x)
        # Optionally, if T' does not equal out_length, you could interpolate:
        x = F.interpolate(x, size=self.out_length, mode='linear', align_corners=False)
        # Output shape: [B, 1, out_length] -> squeeze to [B, out_length]
        x = x.squeeze(1)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, num_layers=2, num_heads=4):
        super(TransformerClassifier, self).__init__()
        decoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # x: [B, T, embedding_dim] (quantized latent sequence)
        x_transformed = self.transformer_encoder(x)  # process the entire sequence
        pooled = x_transformed.mean(dim=1)  # global pooling over time
        logits = self.fc(pooled)
        return logits


class HierarchicalVQVAE(nn.Module):
    def __init__(self, frames=128, out_length=256,
                 num_embeddings_low=141, num_embeddings_high=50,
                 embedding_dim_low=64, embedding_dim_high=32,
                 commitment_cost=0.6):
        super(HierarchicalVQVAE, self).__init__()
        # Low-level encoder: outputs latent with dimension embedding_dim_low
        self.encoder_low = PhysNetEncoder(frames=frames)  # outputs shape: [B, T, embedding_dim_low]

        # Global encoder: downsamples the channel dimension from embedding_dim_low to embedding_dim_high
        self.encoder_high = nn.Sequential(
            nn.Conv1d(embedding_dim_low, embedding_dim_high, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Vector quantizers
        self.vq_low = VectorQuantizer(num_embeddings_low, embedding_dim_low, commitment_cost)
        # Note: vq_high expects inputs of dimension embedding_dim_high
        self.vq_high = VectorQuantizer(num_embeddings_high, embedding_dim_high, commitment_cost)

        self.decoder = Decoder(embedding_dim=embedding_dim_low, out_length=out_length)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim_low, 128),
            nn.ReLU(),
            nn.Linear(128, num_embeddings_low)
        )
        # (Optional) If you want to fuse the global latent with the low-level latent, you might need an additional
        # projection layer. For simplicity, we currently use only the low-level quantized output.

    def forward(self, x):
        # Get low-level latent: shape [B, T, embedding_dim_low]
        latent = self.encoder_low(x)
        quantized_low, loss_low, encoding_indices_low = self.vq_low(latent)

        # Compute a global latent by temporal averaging: shape [B, embedding_dim_low]
        global_feature = latent.mean(dim=1)
        # Prepare for encoder_high: unsqueeze to [B, embedding_dim_low, 1]
        global_feature_unsq = global_feature.unsqueeze(-1)
        # Process global_feature to obtain a global latent of dimension embedding_dim_high: shape [B, embedding_dim_high, 1]
        global_latent = self.encoder_high(global_feature_unsq)
        # Squeeze the time dimension: shape becomes [B, embedding_dim_high]
        global_latent = global_latent.squeeze(-1)

        # Quantize the global latent.
        # Unsqueeze to get a "sequence" of length 1: shape [B, 1, embedding_dim_high]
        quantized_high, loss_high, encoding_indices_high = self.vq_high(global_latent.unsqueeze(1))

        # For fusion, you might choose to combine low- and high-level quantized outputs.
        # Here, for simplicity, we use just the low-level quantized latent.
        recon = self.decoder(quantized_low)
        class_logits = self.classifier(quantized_low.mean(dim=1))
        total_vq_loss = loss_low + loss_high

        return recon, total_vq_loss, class_logits, encoding_indices_low


# ---------------------------
# VQVAE Model using PhysNet encoder
# ---------------------------
class VQVAE(nn.Module):
    def __init__(self, frames=128, out_length=128, num_embeddings=141, embedding_dim=64, commitment_cost=0.5):
        super(VQVAE, self).__init__()
        self.encoder = PhysNetEncoder(frames=frames)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        # self.vector_quantizer = AdaptiveVectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim=embedding_dim, out_length=out_length)
        # self.decoder = ResidualDilatedDecoder(embedding_dim=embedding_dim, out_length=out_length)
        # self.decoder = UNetDecoder(embedding_dim=embedding_dim, out_length=out_length)
        # self.decoder = TransformerDecoder(embedding_dim=embedding_dim, out_length=out_length)
        # Classification head: Predict HR class from the averaged quantized latent.
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_embeddings)  # num_embeddings classes for HR (e.g., 40 to 180 BPM)
        )
        # self.classifier = TransformerClassifier(embedding_dim, num_embeddings, num_layers=2, num_heads=4)


    def forward(self, x):
        """
        Args:
            x: Video input of shape [B, 1, T, H, W]
        Returns:
            recon: Reconstructed rPPG signal [B, out_length]
            vq_loss: VQVAE loss (scalar)
            class_logits: Predicted HR class logits [B, num_embeddings]
            encoding_indices: Discrete latent codes [B, T']
        """
        # Encoder: Extract latent representation from video.
        latent = self.encoder(x)  # [B, T', embedding_dim]

        # Vector quantization:
        # Quantized = nearest codebook vector to the output of the encoder.
        # vq_loss = loss quantifying how big difference between encoder output and nearest codebook vector is
        quantized, vq_loss, encoding_indices = self.vector_quantizer(latent)

        # Decoder: Reconstruct continuous rPPG signal from quantized latent.
        recon = self.decoder(quantized)  # [B, out_length]

        # Classification: Average quantized latent over time, then classify.
        # So for each e.g. 128 frames, take the mean latent vector and classify it
        latent_avg = quantized.mean(dim=1)  # [B, embedding_dim]
        class_logits = self.classifier(latent_avg)  # [B, num_embeddings]

        # class_logits = self.classifier(quantized)  # [B, num_embeddings]

        return recon, vq_loss, class_logits, encoding_indices
