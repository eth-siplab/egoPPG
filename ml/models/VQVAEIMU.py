import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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


class AdvancedFeatureExtractor(nn.Module):
    def __init__(self, output_dim=128):
        super(AdvancedFeatureExtractor, self).__init__()
        # Load a pretrained ResNet18 model.
        resnet = models.resnet18(pretrained=True)
        # Modify the first convolution layer to accept 1 channel instead of 3.
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Remove the fully connected layer.
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Outputs [B, 512, 1, 1]
        # Add a projection layer to map the 512-dimensional feature to output_dim.
        self.proj = nn.Linear(512, output_dim)

    def forward(self, x):
        # x: [B, 1, H, W]
        feat = self.features(x)        # [B, 512, 1, 1]
        feat = feat.view(x.size(0), -1)  # [B, 512]
        out = self.proj(feat)          # [B, output_dim]
        return out


class IMUEmbeddingCNN(nn.Module):
    def __init__(self, embedding_dim=64):
        super(IMUEmbeddingCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=embedding_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        # Using LayerNorm on the final features:
        self.layernorm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # x: [B, T, 1] -> transpose to [B, 1, T] for conv1d
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # Transpose back: [B, T, embedding_dim]
        x = x.transpose(1, 2)
        x = self.layernorm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Here we add a new dimension so that pe becomes [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, T, d_model]
        Returns:
            Tensor of shape [B, T, d_model] with positional encodings added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CrossAttentionIMUImage(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4, feature_extractor_dim_in=1, feature_extractor_dim_out=32):
        super(CrossAttentionIMUImage, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.feature_extractor_dim_in = feature_extractor_dim_in

        # Linear layer to transform IMU data into embedding dimension
        # self.imu_embedding = nn.Linear(1, embedding_dim)
        self.imu_embedding = IMUEmbeddingCNN(embedding_dim=embedding_dim)

        self.imu_positional_encoding = PositionalEncoding(d_model=embedding_dim, dropout=0.1)
        self.img_positional_encoding = PositionalEncoding(d_model=embedding_dim, dropout=0.1)

        # CNN feature extractor for image frames
        """self.feature_extractor = nn.Sequential(
            nn.Conv2d(feature_extractor_dim_in, feature_extractor_dim_out, kernel_size=3, stride=1, padding=1),  # 16 for emb 64, 32 for emb 128
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),  # Downsample to 8x8 for efficiency
            nn.Flatten(start_dim=1)
        )"""
        self.feature_extractor = AdvancedFeatureExtractor(output_dim=embedding_dim)

        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)

        # Output layer to combine attended features
        self.output_layer = nn.Linear(embedding_dim, 1)

    def forward(self, imu_data, rppg_frames):
        batch_size, channel_len, seq_len, h, w = rppg_frames.shape

        # 1. Embed IMU data
        if imu_data.dim() == 2:
            imu_data = imu_data.unsqueeze(-1)
        imu_embedded = self.imu_embedding(imu_data)  # Shape: (batch_size, sequence_length, embedding_dim)
        imu_embedded = self.imu_positional_encoding(imu_embedded)  # [B, seq_len, embedding_dim]
        imu_embedded = imu_embedded.permute(1, 0, 2)

        # --- Process image frames ---
        # Rearrange rppg_frames from [B, C, T, h, w] to [B, T, C, h, w]
        rppg_frames = rppg_frames.permute(0, 2, 1, 3, 4)
        rppg_frames = rppg_frames.reshape(batch_size * seq_len, channel_len, h, w)
        img_features = self.feature_extractor(rppg_frames)  # [B*T, feature_extractor_dim_out * 2 * 2]
        img_features = img_features.view(batch_size, seq_len, self.embedding_dim)
        img_features = self.img_positional_encoding(img_features)  # [B, T, embedding_dim]

        # Reshape img_features for attention: (sequence_length, batch_size, feature_dim)
        img_features = img_features.permute(1, 0, 2)

        # 3. Cross-attention: IMU as query, Image features as key/value
        attn_output, attn_weights = self.cross_attention(imu_embedded, img_features, img_features)

        # 4. Reduce dimensions and apply final output layer
        attended_features = attn_output.permute(1, 0, 2)  # Aggregate over time
        output = self.output_layer(attended_features)  # Shape: (batch_size, 1)

        return output


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

        self.cross_attention = CrossAttentionIMUImage(embedding_dim=128, num_heads=4, feature_extractor_dim_in=1,
                                                      feature_extractor_dim_out=32)

    def forward(self, x, imu):
        # x: [B, 1, T, H, W]
        [batch, channel, length, height, width] = x.shape

        imu = imu.unsqueeze(-1)
        ta_weights = self.cross_attention(imu, x)
        ta_weights = ta_weights.view(batch, 1, length, 1, 1)
        x = x * ta_weights

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


# ---------------------------
# VQVAE Model using PhysNet encoder
# ---------------------------
class VQVAEIMU(nn.Module):
    def __init__(self, frames=128, out_length=128, num_embeddings=141, embedding_dim=64, commitment_cost=0.6):
        super(VQVAEIMU, self).__init__()
        self.encoder = PhysNetEncoder(frames=frames)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        # self.vector_quantizer = AdaptiveVectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim=embedding_dim, out_length=out_length)
        # self.decoder = UNetDecoder(embedding_dim=embedding_dim, out_length=out_length)
        # self.decoder = TransformerDecoder(embedding_dim=embedding_dim, out_length=out_length)
        # Classification head: Predict HR class from the averaged quantized latent.
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_embeddings)  # num_embeddings classes for HR (e.g., 40 to 180 BPM)
        )

    def forward(self, x, imu):
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
        latent = self.encoder(x, imu)  # [B, T', embedding_dim]

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

        return recon, vq_loss, class_logits, encoding_indices
