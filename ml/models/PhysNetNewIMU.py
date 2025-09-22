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

from collections import OrderedDict
from einops import rearrange

from source.ml.models.vit_utils import DropPath, to_2tuple, trunc_normal_


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


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)  # [batch_size, seq_length, input_dim]
        self.key = nn.Linear(input_dim, input_dim)  # [batch_size, seq_length, input_dim]
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):  # x.shape (batch_size, seq_length, input_dim)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class IMUSelfAttention(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4):
        super(IMUSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

        self.linear_embedding = nn.Linear(1, embedding_dim)
        self.attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=num_heads)

        self.linear = nn.Linear(self.embedding_dim, 1)
        # self.positional_encoder = PositionalEncoding(embedding_dim)

    def forward(self, imu_data):
        imu_embedded = self.linear_embedding(imu_data)  # Shape: (batch_size, sequence_length, embedding_dim)

        # Reshape for MultiheadAttention: (sequence_length, batch_size, embedding_dim)
        imu_embedded = imu_embedded.permute(1, 0, 2)

        # imu_embedded = self.positional_encoder(imu_embedded)

        # Compute self-attention
        attn_output, attn_weights = self.attention(imu_embedded, imu_embedded, imu_embedded)

        # Pass through linear layer to get one weight per time step
        frame_weights = self.linear(attn_output).squeeze(-1)  # Shape: (128, batch_size)

        frame_weights = frame_weights.permute(1, 0)  # Shape: (batch_size, sequence_length)
        frame_weights = torch.softmax(frame_weights, dim=-1)  # Normalize weights over the sequence dimension

        return frame_weights


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=(48, 128), patch_size=16, in_chans=1, embed_dim=64):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.pool = nn.Linear(24, 1)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = self.pool(x.flatten(2)).transpose(1, 2)
        # x = x.flatten(2).transpose(1, 2)
        # return x, T, W
        return x


class CrossAttentionIMUImage(nn.Module):
    def __init__(self, embedding_dim=128, num_heads=4, num_frames=128, in_channels=1, img_size=(48, 128)):
        super(CrossAttentionIMUImage, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.img_size = img_size

        # Linear layer to transform IMU data into embedding dimension
        self.imu_embedding = nn.Linear(1, embedding_dim)

        # CNN feature extractor for image frames
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, self.embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Downsample to 8x8 for efficiency
            nn.Flatten(start_dim=2)
        )

        # self.imu_pos_embed = nn.Parameter(torch.zeros(1, num_frames, self.embedding_dim))
        # self.img_pos_embed = nn.Parameter(torch.zeros(1, num_frames, self.embedding_dim))

        # self.feature_extractor = PatchEmbed(img_size=img_size, patch_size=16, in_chans=in_channels,
        #                                     embed_dim=self.embedding_dim)

        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads)

        # Output layer to combine attended features
        self.output_layer = nn.Linear(embedding_dim, 1)
        self.norm = nn.LayerNorm([num_frames, 1])  # ([frames, 3])
        # self.output_layer2 = nn.Linear(129, 128)

    def forward(self, imu_data, rppg_frames):
        batch_size, channel_len, seq_len, h, w = rppg_frames.shape

        # 1. Embed IMU data
        imu_data = imu_data.unsqueeze(-1)
        imu_embedded = self.imu_embedding(imu_data)  # Shape: (batch_size, sequence_length, embedding_dim)
        # imu_embedded += self.imu_pos_embed

        # Reshape IMU data for attention: (sequence_length, batch_size, embedding_dim)
        imu_embedded = imu_embedded.permute(1, 0, 2)

        # 2. Extract features from each frame
        rppg_frames = rppg_frames.view(batch_size * self.num_frames, self.in_channels, h, w)  # Reshape for CNN processing (4 * 128, 1, 48, 128)
        # rppg_frames = rppg_frames.flatten(2)

        img_features = self.feature_extractor(rppg_frames)  # Shape: (batch_size*sequence_length, 32, 8*8)
        img_features = img_features.view(batch_size, self.num_frames, -1)  # Shape: (batch_size, sequence_length, feature_dim)
        # img_features += self.img_pos_embed

        # Reshape img_features for attention: (sequence_length, batch_size, feature_dim)
        img_features = img_features.permute(1, 0, 2)

        # 3. Cross-attention: IMU as query, Image features as key/value
        attn_output, attn_weights = self.cross_attention(imu_embedded, img_features, img_features)

        # 4. Reduce dimensions and apply final output layer
        attended_features = attn_output.permute(1, 0, 2)  # Aggregate over time

        output = self.output_layer(attended_features)  # Shape: (batch_size, 1)

        output = self.norm(output)
        output = torch.sigmoid(output)
        output = output.view(batch_size, 1, self.num_frames, 1, 1)

        return output


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialAttentionGate(nn.Module):
    def __init__(self, in_channels_imu):
        super(SpatialAttentionGate, self).__init__()
        self.compress = ChannelPool()
        kernel_size = 7
        self.spatial = BasicConv(4, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        # self.spatial_imu = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        # self.imu = BasicConv(in_channels_imu, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x, imu):
        imu = imu.unsqueeze(2).unsqueeze(3).expand_as(x)
        imu_compress = self.compress(imu)
        # imu_out = self.spatial_imu(imu_compress)
        # imu = self.imu(imu)
        x_compress = self.compress(x)
        # x_compress = x_compress + imu
        x_compress_new = torch.cat([x_compress, imu_compress], 1)
        x_out = self.spatial(x_compress_new)
        # x_out = x_out + imu_out
        # x_out = x_out + imu
        scale = F.sigmoid(x_out)
        return scale * x


class SpatialAttentionGateold(nn.Module):
    def __init__(self):
        super(SpatialAttentionGateold, self).__init__()
        self.compress = ChannelPool()
        kernel_size = 7
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        """self.spatial = sequential(nn.Conv3d(2, 1, (1, 1, 1), bias=True, stride=1, padding=0),
                                  nn.BatchNorm3d(1, momentum=0.9, eps=1e-04, affine=True),
                                  nn.ReLU(inplace=True))"""

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return scale * x


class PhysNetNewIMU(nn.Module):
    def __init__(self, frames=128):
        super(PhysNetNewIMU, self).__init__()

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
        self.MaxpoolTem = nn.MaxPool1d(2, stride=2)

        # self.poolspa = nn.AdaptiveMaxPool3d((frames, 1, 1))    # pool only spatial space
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))   # better for EDA!

        """self.SpatialAttentionGate1 = SpatialAttentionGate()
        self.SpatialAttentionGate2 = SpatialAttentionGate()
        self.SpatialAttentionGate3 = SpatialAttentionGate()
        self.SpatialAttentionGate4 = SpatialAttentionGate()"""

        self.SpatialAttentionGate1 = SpatialAttentionGate(in_channels_imu=16)
        self.SpatialAttentionGate2 = SpatialAttentionGate(in_channels_imu=64)
        self.SpatialAttentionGate3 = SpatialAttentionGate(in_channels_imu=64)
        self.SpatialAttentionGate4 = SpatialAttentionGate(in_channels_imu=64)

        """self.SpatialAttentionGate1 = SpatialAttentionGateold()
        self.SpatialAttentionGate2 = SpatialAttentionGateold()
        self.SpatialAttentionGate3 = SpatialAttentionGateold()
        self.SpatialAttentionGate4 = SpatialAttentionGateold()"""

        self.IMUBlock1 = nn.Sequential(
            nn.Conv1d(1, 16, 1, stride=1, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.IMUBlock2 = nn.Sequential(
            nn.Conv1d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.IMUBlock3 = nn.Sequential(
            nn.Conv1d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.IMUBlock4 = nn.Sequential(
            nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.IMUBlock5 = nn.Sequential(
            nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.IMUBlock6 = nn.Sequential(
            nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.IMUBlock7 = nn.Sequential(
            nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # self.norm = nn.LayerNorm(frames)  # ([frames, 3])

        embedding_dim = 64
        heads = 4
        self.cross_attention1 = CrossAttentionIMUImage(embedding_dim=embedding_dim, num_heads=heads, num_frames=frames, in_channels=1)
        # self.cross_attention2 = CrossAttentionIMUImage(embedding_dim=embedding_dim, num_heads=heads, num_frames=frames, in_channels=64)
        # self.cross_attention3 = CrossAttentionIMUImage(embedding_dim=embedding_dim, num_heads=heads, num_frames=frames//2, in_channels=64)
        # self.cross_attention4 = CrossAttentionIMUImage(embedding_dim=embedding_dim, num_heads=heads, num_frames=frames//4, in_channels=64)
        # self.cross_attention5 = CrossAttentionIMUImage(embedding_dim=embedding_dim, num_heads=heads, num_frames=frames//4, in_channels=64)
        # self.cross_attention6 = CrossAttentionIMUImage(embedding_dim=embedding_dim, num_heads=heads, num_frames=frames//2, in_channels=64)

        # self.imu_down3 = nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1)
        # self.imu_down4 = nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1)
        """self.imu_down4 = sequential(
                nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1),
                nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1)
        )"""
        """self.imu_up6 = nn.ConvTranspose1d(1, 1, kernel_size=4, stride=2, padding=1)
        self.imu_down5 = sequential(
                nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1),
                nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1)
        )"""
        # self.imu_down6 = nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1)


    def forward(self, x, imu):  # Batch_size*[3, T, 128,128]
        [batch, channel, length, height, width] = x.shape

        # IMU
        # print(imu.shape)
        # imu = self.norm(imu)
        # imu = self.smoothing_conv(imu.unsqueeze(1)).squeeze(1)

        # imu = self.imu_conv1(imu.unsqueeze(1))
        # imu = self.imu_conv2(imu)
        # imu = self.imu_conv3(imu).squeeze(1)
        # imu = self.norm(imu)

        # Image
        ta_weights = self.cross_attention1(imu, x)
        x = x * ta_weights
        # x = x + x * ta_weights

        x = self.ConvBlock1(x)  # x [16, T, 128,128]
        imu = self.IMUBlock1(imu.unsqueeze(1))
        # x = x + imu.unsqueeze(3).unsqueeze(4).expand_as(x)  # TODO
        # ta_weights = self.cross_attention1(imu, x)
        # x = x * ta_weights
        # x = self.SpatialAttentionGate1(x.view(batch * length, 16, height, width), imu.view(batch * length, 16)).view(batch, 16, length, height, width)
        x = self.SpatialAttentionGate1(x.view(batch * length, 16, height, width)).view(batch, 16, length, height, width)
        x = self.MaxpoolSpa(x)  # x [16, T, h/2, w/2]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x = self.ConvBlock3(x)  # x [64, T, 64,64]
        imu = self.IMUBlock2(imu)
        imu = self.IMUBlock3(imu)
        # x = x + imu.unsqueeze(3).unsqueeze(4).expand_as(x)  # TODO

        # ta_weights = self.cross_attention2(imu, x)
        # x = x * ta_weights
        # x = self.SpatialAttentionGate2(x.view(batch * length, 64, height//2, width//2), imu.view(batch * length, 64)).view(batch, 64, length, height//2, width//2)
        x = self.SpatialAttentionGate2(x.view(batch * length, 64, height // 2, width // 2)).view(batch, 64, length, height // 2, width // 2)
        x = self.MaxpoolSpaTem(x)  # x [64, T/2, 64,64]
        imu = self.MaxpoolTem(imu)  # x [16, T/2]

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        imu = self.IMUBlock4(imu)
        imu = self.IMUBlock5(imu)
        # x = x + imu.unsqueeze(3).unsqueeze(4).expand_as(x)  # TODO
        # imu3 = self.imu_down3(imu.unsqueeze(1)).squeeze(1)
        # ta_weights = self.cross_attention3(imu3, x)
        # x = x * ta_weights
        # x = self.SpatialAttentionGate3(x.view(batch * (length//2), 64, height//4, width//4), imu.view(batch * length//2, 64)).view(batch, 64, (length//2), height//4, width//4)
        x = self.SpatialAttentionGate3(x.view(batch * (length // 2), 64, height // 4, width // 4)).view(batch, 64, (length // 2), height // 4, width // 4)
        x = self.MaxpoolSpaTem(x)  # x [64, T/4, 16,16]
        imu = self.MaxpoolTem(imu)  # x [16, T/2]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        imu = self.IMUBlock6(imu)
        imu = self.IMUBlock7(imu)
        # x = x + imu.unsqueeze(3).unsqueeze(4).expand_as(x)  # TODO
        # imu4 = self.imu_down4(imu3.unsqueeze(1)).squeeze(1)
        # ta_weights = self.cross_attention4(imu4, x)
        # x = x * ta_weights
        # x = self.SpatialAttentionGate4(x.view(batch * (length//4), 64, height//8, width//8), imu.view(batch * length//4, 64)).view(batch, 64, (length//4), height//8, width//8)
        x = self.SpatialAttentionGate4(x.view(batch * (length // 4), 64, height // 8, width // 8)).view(batch, 64, (length // 4), height // 8, width // 8)
        x = self.MaxpoolSpa(x)  # x [64, T/4, 8,8]

        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]

        # imu5 = self.imu_down5(imu.unsqueeze(1)).squeeze(1)
        # imu5 = imu4
        # ta_weights = self.cross_attention5(imu5, x)
        # x = x * ta_weights
        # x = x + x * ta_weights

        x = self.upsample(x)  # x [64, T/2, 8, 8]

        # imu6 = self.imu_down6(imu.unsqueeze(1)).squeeze(1)
        # imu6 = self.imu_up6(imu5.unsqueeze(1)).squeeze(1)
        # ta_weights = self.cross_attention6(imu6, x)
        # x = x * ta_weights
        # x  = x + x * ta_weights

        x = self.upsample2(x)  # x [64, T, 8, 8]
        x = self.poolspa(x)
        x = self.ConvBlock10(x)  # x [1, T, 1,1]

        out = x.view(-1, length)

        # out = out.mean(dim=1).unsqueeze(-1).repeat(1, length)

        return out