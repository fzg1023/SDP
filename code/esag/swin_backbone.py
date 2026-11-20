"""
改进的 Swin Transformer 骨干网络 + CBAM 注意力机制
用于旋转机械故障诊断的多尺度特征提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from typing import Optional, Tuple, List
import numpy as np
import math


# ==============================================================================
# CBAM: Convolutional Block Attention Module
# ==============================================================================

class ChannelAttention(nn.Module):
    """通道注意力模块：关注"什么"特征重要"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """空间注意力模块：关注"哪里"特征重要"""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(combined))


class CBAM(nn.Module):
    """
    CBAM: 通道注意力 + 空间注意力
    让模型自动聚焦关键故障特征（冲击峰值、边频带）
    """

    def __init__(self, channels: int, reduction: int = 16,
                 spatial_kernel: int = 7):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x


# ==============================================================================
# 窗口多头自注意力 (Window Multi-Head Self-Attention)
# ==============================================================================

def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    将特征图分割为不重叠的窗口
    Args:
        x: (B, H, W, C)
        window_size: 窗口大小
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int,
                   H: int, W: int) -> torch.Tensor:
    """
    将窗口合并回特征图
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    窗口多头自注意力 (W-MSA / SW-MSA)
    支持相对位置偏置
    """

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # 计算相对位置索引
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (num_windows*B, N, C), N = actual_window_size * actual_window_size
            mask: (num_windows, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 相对位置偏置 - 支持动态窗口大小
        actual_ws = int(N ** 0.5)
        if actual_ws == self.window_size:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(N, N, -1)
        else:
            # 动态计算相对位置偏置
            coords_h = torch.arange(actual_ws, device=x.device)
            coords_w = torch.arange(actual_ws, device=x.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += actual_ws - 1
            relative_coords[:, :, 1] += actual_ws - 1
            relative_coords[:, :, 0] *= 2 * actual_ws - 1
            rel_pos_index = relative_coords.sum(-1)
            # 创建临时偏置表
            table_size = (2 * actual_ws - 1) * (2 * actual_ws - 1)
            if table_size <= self.relative_position_bias_table.shape[0]:
                bias_table = self.relative_position_bias_table[:table_size]
            else:
                bias_table = F.interpolate(
                    self.relative_position_bias_table.T.unsqueeze(0),
                    size=table_size, mode='linear', align_corners=False
                ).squeeze(0).T
            relative_position_bias = bias_table[
                rel_pos_index.view(-1)
            ].view(N, N, -1)

        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + \
                   mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ==============================================================================
# Swin Transformer Block
# ==============================================================================

class Mlp(nn.Module):
    """MLP 前馈网络"""

    def __init__(self, in_features: int, hidden_features: int = None,
                 out_features: int = None, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block：
    W-MSA / SW-MSA + MLP + 残差连接
    """

    def __init__(self, dim: int, num_heads: int, window_size: int = 7,
                 shift_size: int = 0, mlp_ratio: float = 4.,
                 qkv_bias: bool = True, drop: float = 0.,
                 attn_drop: float = 0., drop_path: float = 0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, L, C = x.shape
        assert L == H * W, f"Input feature size ({L}) != H*W ({H}*{W})"

        # 动态调整窗口大小：确保不超过特征图尺寸
        window_size = min(self.window_size, H, W)
        shift_size = 0 if window_size >= min(H, W) else min(self.shift_size, window_size // 2)

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 如果特征图不能被 window_size 整除，做 padding
        pad_r = (window_size - W % window_size) % window_size
        pad_b = (window_size - H % window_size) % window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # 循环移位
        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size),
                                   dims=(1, 2))
            # 计算注意力掩码
            attn_mask = self._compute_mask(Hp, Wp, x.device, window_size, shift_size)
        else:
            shifted_x = x
            attn_mask = None

        # 分窗口
        x_windows = window_partition(shifted_x, window_size)
        x_windows = x_windows.view(-1, window_size * window_size, C)

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # 合并窗口
        attn_windows = attn_windows.view(-1, window_size, window_size, C)
        shifted_x = window_reverse(attn_windows, window_size, Hp, Wp)

        # 逆循环移位
        if shift_size > 0:
            x = torch.roll(shifted_x,
                           shifts=(shift_size, shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x

        # 移除 padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def _compute_mask(self, H: int, W: int, device,
                      window_size: int = None, shift_size: int = None) -> torch.Tensor:
        """计算 SW-MSA 的注意力掩码"""
        if window_size is None:
            window_size = self.window_size
        if shift_size is None:
            shift_size = self.shift_size

        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size)
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask


class DropPath(nn.Module):
    """随机深度 (Stochastic Depth)"""

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


# ==============================================================================
# Patch Embedding & Merging
# ==============================================================================

class PatchEmbed(nn.Module):
    """将输入图像分割为不重叠的 patch 并嵌入"""

    def __init__(self, img_size: int = 64, patch_size: int = 4,
                 in_chans: int = 2, embed_dim: int = 96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = img_size // patch_size
        self.num_patches = self.patches_resolution ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, num_patches, embed_dim)
            H, W: patch grid 的高宽
        """
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        H_out, W_out = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        return x, H_out, W_out


class PatchMerging(nn.Module):
    """Patch 合并层：下采样 2x，通道数翻倍"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)
        # 取 2x2 邻域拼接
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x, H // 2, W // 2


# ==============================================================================
# Swin Transformer Stage
# ==============================================================================

class SwinStage(nn.Module):
    """Swin Transformer 的一个阶段（包含多个 Block + 可选下采样）"""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: list = None,
        downsample: bool = True,
        enable_cbam: bool = True,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if drop_path else 0.
            ))

        # CBAM 增强：每个 stage 末尾添加 CBAM
        self.cbam = CBAM(dim)
        # Runtime flag (not part of state_dict)
        self.enable_cbam = bool(enable_cbam)

        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        for blk in self.blocks:
            x = blk(x, H, W)

        # CBAM 增强
        B, L, C = x.shape
        x_2d = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        if bool(self.enable_cbam):
            x_2d = self.cbam(x_2d)
        x = x_2d.permute(0, 2, 3, 1).view(B, L, C)

        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)

        return x, H, W


# ==============================================================================
# 改进的 Swin Transformer 骨干网络
# ==============================================================================

class SwinBackbone(nn.Module):
    """
    改进的 Swin Transformer 骨干网络
    - 分层窗口注意力捕捉多尺度故障模式
    - 每阶段嵌入 CBAM 注意力增强
    - 输出多尺度特征用于后续零样本推理
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 4,
        in_chans: int = 2,
        embed_dim: int = 96,
        depths: list = None,
        num_heads: list = None,
        window_size: int = 8,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        enable_cbam: bool = True,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 2, 6, 2]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]

        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.num_features = embed_dim * (2 ** (self.num_stages - 1))

        # Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in
               torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建各阶段
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = SwinStage(
                dim=embed_dim * (2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=(i < self.num_stages - 1),
                enable_cbam=bool(enable_cbam),
            )
            self.stages.append(stage)

            self.enable_cbam = bool(enable_cbam)

        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.apply(self._init_weights)

    def set_cbam_enabled(self, enabled: bool) -> None:
        """Enable/disable CBAM at runtime (checkpoint-compatible)."""
        self.enable_cbam = bool(enabled)
        for s in self.stages:
            if hasattr(s, 'enable_cbam'):
                s.enable_cbam = bool(enabled)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) 输入双通道图像
        Returns:
            features: (B, num_features) 全局特征向量
        """
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for stage in self.stages:
            x, H, W = stage(x, H, W)

        x = self.norm(x)  # (B, L, C)
        x = self.avgpool(x.transpose(1, 2))  # (B, C, 1)
        x = torch.flatten(x, 1)  # (B, num_features)
        return x

    def forward_multiscale(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        提取多尺度特征（用于语义属性构建）
        Returns:
            features_list: 各阶段的特征
        """
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        features_list = []
        for stage in self.stages:
            x, H, W = stage(x, H, W)
            features_list.append(x.mean(dim=1))  # 全局平均池化

        return features_list
