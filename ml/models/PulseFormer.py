""" PhysNet
We repulicate the net pipeline of the orginal paper, but set the input as diffnormalized data.
orginal source:
Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks
British Machine Vision Conference (BMVC)} 2019,
By Zitong Yu, 2019/05/05
Only for research purpose, and commercial use is not allowed.
MIT License
Copyright (c) 2019
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from collections import OrderedDict
from torchvision.models import ResNet18_Weights


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
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
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
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
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return scale * x


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


class ImageFeatureExtractor(nn.Module):
    def __init__(self, output_dim=128):
        super(ImageFeatureExtractor, self).__init__()
        # Load a pretrained ResNet18 model.
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
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


class CrossAttentionIMUImage(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4, feature_extractor_dim_in=1, feature_extractor_dim_out=32):
        super(CrossAttentionIMUImage, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.feature_extractor_dim_in = feature_extractor_dim_in

        # Linear layer to transform IMU data into embedding dimension
        self.imu_embedding = IMUEmbeddingCNN(embedding_dim=embedding_dim)

        self.imu_positional_encoding = PositionalEncoding(d_model=embedding_dim, dropout=0.1)
        self.img_positional_encoding = PositionalEncoding(d_model=embedding_dim, dropout=0.1)

        # CNN feature extractor for image frames
        self.image_feature_extractor = ImageFeatureExtractor(output_dim=embedding_dim)

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
        img_features = self.image_feature_extractor(rppg_frames)  # [B*T, feature_extractor_dim_out * 2 * 2]
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


class PulseFormer(nn.Module):
    def __init__(self, frames=128, h=48, w=128, num_embeddings=141):
        super(PulseFormer, self).__init__()

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
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(4, 1, 1), stride=(2, 1, 1),
                               padding=(1, 0, 0)),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(4, 1, 1), stride=(2, 1, 1),
                               padding=(1, 0, 0)),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(64, 1, (1, 1, 1), stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

        self.SpatialAttentionGate1 = SpatialAttentionGate()
        self.SpatialAttentionGate2 = SpatialAttentionGate()
        self.SpatialAttentionGate3 = SpatialAttentionGate()
        self.SpatialAttentionGate4 = SpatialAttentionGate()

        self.cross_attention = CrossAttentionIMUImage(embedding_dim=128, num_heads=4, feature_extractor_dim_in=1,
                                                      feature_extractor_dim_out=32)

    def forward(self, x, imu):  # Batch_size*[3, T, 128,128]
        [batch, channel, length, height, width] = x.shape

        # IMU
        imu = imu.unsqueeze(-1)
        ta_weights = self.cross_attention(imu, x)
        ta_weights = ta_weights.view(batch, 1, length, 1, 1)
        x = x * ta_weights

        x = self.ConvBlock1(x)  # x [16, T, 128,128]
        x = self.SpatialAttentionGate1(x.view(batch * length, 16, height, width)).view(batch, 16, length, height, width)
        x = self.MaxpoolSpa(x)  # x [16, T, h/2, w/2]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x = self.ConvBlock3(x)  # x [64, T, 64,64]
        x = self.SpatialAttentionGate2(x.view(batch * length, 64, height//2, width//2)).view(batch, 64, length, height//2, width//2)
        x = self.MaxpoolSpaTem(x)  # x [64, T/2, 64,64]

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = self.SpatialAttentionGate3(x.view(batch * (length//2), 64, height//4, width//4)).view(batch, 64, (length//2), height//4, width//4)
        x = self.MaxpoolSpaTem(x)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        x = self.SpatialAttentionGate4(x.view(batch * (length//4), 64, height//8, width//8)).view(batch, 64, (length//4), height//8, width//8)
        x = self.MaxpoolSpa(x)  # x [64, T/4, 8,8]

        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]

        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]

        x = self.poolspa(x)
        x = self.ConvBlock10(x)  # x [1, T, 1,1]

        out = x.view(-1, length)

        return out
