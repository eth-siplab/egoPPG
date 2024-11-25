import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from source.ml.models.vit_utils import DropPath, to_2tuple, trunc_normal_


class PatchEmbed(nn.Module):
    def __init__(self, h=48, w=128, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple((h, w))
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)  # [B*T, embed_dim, H', W']
        x = rearrange(x, '(b t) d h w -> b t (h w) d', b=B, t=T)  # shape [B, T, num_patches, embed_dim]
        return x


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, imu_query, video_key_value):
        B, T, C = imu_query.shape
        q = self.q_proj(imu_query).reshape(B, T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_proj(video_key_value).reshape(B, T, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x


class TemporalSpatialAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.spatial_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop)
        self.temporal_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop)

    def forward(self, x):
        B, T, embed_dim = x.shape

        # Spatial attention
        x = rearrange(x, 'b t d -> t b d')
        x = self.spatial_attn(x, x, x)[0]
        x = rearrange(x, 't b d -> b t d')

        # Temporal attention
        x = rearrange(x, 'b t d -> t b d')
        x = self.temporal_attn(x, x, x)[0]
        x = rearrange(x, 't b d -> b t d')

        return x


class IMUPatchEmbed(nn.Module):
    """ Embeds IMU data into a patch-wise embedding format """

    def __init__(self, imu_dim, embed_dim):
        super().__init__()
        self.fc = nn.Linear(imu_dim, embed_dim)

    def forward(self, imu):
        # imu: [B, T, imu_dim]
        x = self.fc(imu)
        return x  # [B, T, embed_dim]


class PhysNetNewIMUCA(nn.Module):
    def __init__(self, h=48, w=128, patch_size=16, in_chans=1, embed_dim=64, num_heads=4, num_frames=8):
        super().__init__()
        self.video_patch_embed = PatchEmbed(h, w, patch_size, in_chans, embed_dim)
        self.imu_patch_embed = IMUPatchEmbed(1, embed_dim)

        self.cross_attn = CrossAttention(embed_dim, num_heads=num_heads)
        self.temporal_spatial_block = TemporalSpatialAttentionBlock(embed_dim, num_heads)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames + 1, embed_dim))  # +1 for CLS token
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 1)
        self.head2 = nn.Linear(129, 128)

    def forward(self, video, imu):
        B = video.shape[0]

        # Embed video and IMU patches
        video_embeds = self.video_patch_embed(video).mean(dim=2)  # Reduce across spatial dimension
        imu_embeds = self.imu_patch_embed(imu.unsqueeze(-1))

        # Cross-attention between IMU and video features
        imu_attended_video = self.cross_attn(imu_embeds, video_embeds)

        # Concatenate CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        imu_attended_video = torch.cat((cls_tokens, imu_attended_video), dim=1) + self.pos_embed

        # Temporal-Spatial Attention
        attended_output = self.temporal_spatial_block(imu_attended_video)

        # Classification head
        x = attended_output
        return self.head2(self.head(x).squeeze(2))  # CLS token output

