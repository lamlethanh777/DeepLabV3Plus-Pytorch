"""
Attention modules for DeepLabV3+ variants.

- ShuffleAttention: Channel shuffle + group-wise channel-spatial attention
- ECAAttention: Efficient Channel Attention using 1D adaptive convolution
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class ShuffleAttention(nn.Module):
    """
    Shuffle Attention Module.
    
    Splits channels into groups, applies channel and spatial attention 
    within each group, then shuffles channels across groups.
    
    Reference: "SA-Net: Shuffle Attention for Deep Convolutional Neural Networks"
    https://arxiv.org/abs/2102.00240
    
    Args:
        channels (int): Number of input channels.
        groups (int): Number of groups for channel shuffle. Default: 64.
    """
    
    def __init__(self, channels, groups=64):
        super(ShuffleAttention, self).__init__()
        self.channels = channels
        self.groups = groups
        self.channels_per_group = channels // groups
        
        # Channel attention parameters (per group)
        self.channel_weight = nn.Parameter(torch.zeros(1, channels // 2, 1, 1))
        self.channel_bias = nn.Parameter(torch.ones(1, channels // 2, 1, 1))
        
        # Spatial attention parameters (per group)
        self.spatial_weight = nn.Parameter(torch.zeros(1, channels // 2, 1, 1))
        self.spatial_bias = nn.Parameter(torch.ones(1, channels // 2, 1, 1))
        
        self.gn = nn.GroupNorm(channels // 2, channels // 2)
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Reshape: (B, C, H, W) -> (B, G, C//G, H, W)
        x = x.view(b, self.groups, self.channels_per_group, h, w)
        
        # Split into two branches along channel dimension
        # x1: channel attention branch, x2: spatial attention branch
        x1, x2 = x.chunk(2, dim=2)  # Each: (B, G, C//(2*G), H, W)
        
        # Reshape for processing
        x1 = x1.reshape(b, c // 2, h, w)
        x2 = x2.reshape(b, c // 2, h, w)
        
        # Channel attention: global avg pool -> scale
        x1_gap = F.adaptive_avg_pool2d(x1, 1)
        x1_attention = torch.sigmoid(self.channel_weight * x1_gap + self.channel_bias)
        x1 = x1 * x1_attention
        
        # Spatial attention: group norm -> scale
        x2_gn = self.gn(x2)
        x2_attention = torch.sigmoid(self.spatial_weight * x2_gn + self.spatial_bias)
        x2 = x2 * x2_attention
        
        # Concatenate along channel dimension
        out = torch.cat([x1, x2], dim=1)  # (B, C, H, W)
        
        # Channel shuffle
        out = out.view(b, 2, c // 2, h, w)
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        out = out.view(b, c, h, w)
        
        return out


class ECAAttention(nn.Module):
    """
    Efficient Channel Attention Module.
    
    Uses 1D adaptive convolution on channel descriptors - lightweight with no FC reduction.
    Kernel size is adaptively determined based on channel count.
    
    Reference: "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"
    https://arxiv.org/abs/1910.03151
    
    Args:
        channels (int): Number of input channels.
        gamma (float): Gamma parameter for adaptive kernel size. Default: 2.
        beta (float): Beta parameter for adaptive kernel size. Default: 1.
        kernel_size (int, optional): Fixed kernel size. If None, computed adaptively.
    """
    
    def __init__(self, channels, gamma=2, beta=1, kernel_size=None):
        super(ECAAttention, self).__init__()
        self.channels = channels
        
        # Compute adaptive kernel size: k = |log2(C)/gamma + beta|_odd
        if kernel_size is None:
            t = int(abs(math.log2(channels) / gamma + beta))
            kernel_size = t if t % 2 == 1 else t + 1  # Ensure odd
        
        self.kernel_size = kernel_size
        
        # 1D convolution for channel attention
        self.conv = nn.Conv1d(
            1, 1, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2, 
            bias=False
        )
        
    def forward(self, x):
        # Global average pooling: (B, C, H, W) -> (B, C, 1, 1)
        y = F.adaptive_avg_pool2d(x, 1)
        
        # Reshape for 1D conv: (B, C, 1, 1) -> (B, 1, C)
        y = y.squeeze(-1).permute(0, 2, 1)
        
        # 1D convolution across channels
        y = self.conv(y)
        
        # Reshape back: (B, 1, C) -> (B, C, 1, 1)
        y = y.permute(0, 2, 1).unsqueeze(-1)
        
        # Sigmoid activation and scale
        y = torch.sigmoid(y)
        
        return x * y
