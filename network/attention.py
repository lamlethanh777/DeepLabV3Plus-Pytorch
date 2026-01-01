"""
Attention modules for DeepLabV3+ variants.

- ShuffleAttention: Channel shuffle + group-wise channel-spatial attention
- ECAAttention: Efficient Channel Attention using 1D adaptive convolution
- EPSAttention: Efficient Pyramid Split Attention
- StripPooling: Strip pooling module for capturing long-range dependencies
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
        # Adaptive groups: avoid too many groups for small channel counts
        self.groups = min(groups, channels // 4)  # At least 4 channels per group
        self.channels_per_group = channels // self.groups
        
        # Channel attention parameters (per group)
        self.channel_weight = nn.Parameter(torch.zeros(1, channels // 2, 1, 1))
        self.channel_bias = nn.Parameter(torch.ones(1, channels // 2, 1, 1))
        
        # Spatial attention parameters (per group)
        self.spatial_weight = nn.Parameter(torch.zeros(1, channels // 2, 1, 1))
        self.spatial_bias = nn.Parameter(torch.ones(1, channels // 2, 1, 1))
        
        self.gn = nn.GroupNorm(channels // 2, channels // 2)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize parameters with small values for stable training."""
        nn.init.constant_(self.channel_weight, 0.01)
        nn.init.constant_(self.channel_bias, 1.0)
        nn.init.constant_(self.spatial_weight, 0.01)
        nn.init.constant_(self.spatial_bias, 1.0)
        
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


class EPSAttention(nn.Module):
    """
    Efficient Pyramid Split Attention Module.
    
    Splits channels into multiple groups, applies multi-scale convolutions,
    and uses softmax attention to weight different scales.
    
    Reference: "EPSANet: An Efficient Pyramid Split Attention Block"
    https://arxiv.org/abs/2105.14447
    
    Args:
        channels (int): Number of input channels.
        num_splits (int): Number of channel splits/scales. Default: 4.
        reduction (int): Channel reduction ratio for attention. Default: 4.
    """
    
    def __init__(self, channels, num_splits=4, reduction=4):
        super(EPSAttention, self).__init__()
        self.channels = channels
        self.num_splits = num_splits
        self.split_channels = channels // num_splits
        
        # Multi-scale convolutions for each split
        self.convs = nn.ModuleList()
        for i in range(num_splits):
            # Increasing kernel sizes: 3, 5, 7, 9...
            kernel_size = 3 + i * 2
            padding = kernel_size // 2
            self.convs.append(nn.Sequential(
                nn.Conv2d(self.split_channels, self.split_channels, 
                         kernel_size=kernel_size, padding=padding, 
                         groups=self.split_channels, bias=False),
                nn.BatchNorm2d(self.split_channels),
                nn.ReLU(inplace=True)
            ))
        
        # Squeeze-and-Excitation style attention
        reduced_channels = max(channels // reduction, 8)
        self.se_conv1 = nn.Conv2d(channels, reduced_channels, 1, bias=False)
        self.se_bn = nn.BatchNorm2d(reduced_channels)
        self.se_relu = nn.ReLU(inplace=True)
        self.se_conv2 = nn.Conv2d(reduced_channels, num_splits, 1, bias=False)
        
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Split along channel dimension
        splits = torch.chunk(x, self.num_splits, dim=1)
        
        # Apply multi-scale convolutions
        feats = []
        for i, (split, conv) in enumerate(zip(splits, self.convs)):
            feats.append(conv(split))
        
        # Stack for attention: (B, num_splits, split_channels, H, W)
        feats_stack = torch.stack(feats, dim=1)
        
        # Global pooling for attention weights
        feats_sum = sum(feats)  # (B, split_channels, H, W)
        # Expand to full channels for SE
        feats_cat = torch.cat(feats, dim=1)  # (B, C, H, W)
        gap = F.adaptive_avg_pool2d(feats_cat, 1)  # (B, C, 1, 1)
        
        # SE attention to get split weights
        attn = self.se_conv1(gap)
        attn = self.se_bn(attn)
        attn = self.se_relu(attn)
        attn = self.se_conv2(attn)  # (B, num_splits, 1, 1)
        attn = F.softmax(attn, dim=1)  # Softmax over splits
        
        # Apply attention weights
        attn = attn.unsqueeze(2)  # (B, num_splits, 1, 1, 1)
        out = (feats_stack * attn).sum(dim=1)  # (B, split_channels, H, W)
        
        # Expand back to original channels
        out = out.repeat(1, self.num_splits, 1, 1)
        
        return out


class StripPooling(nn.Module):
    """
    Strip Pooling Module for capturing long-range dependencies.
    
    Uses horizontal and vertical strip average pooling to capture
    global context in both directions, then fuses with local features.
    
    Reference: "Strip Pooling: Rethinking Spatial Pooling for Scene Parsing"
    https://arxiv.org/abs/2003.13328
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        pool_size (tuple): Pool sizes for (horizontal_height, vertical_width). Default: (20, 12).
    """
    
    def __init__(self, in_channels, out_channels, pool_size=(20, 12)):
        super(StripPooling, self).__init__()
        self.pool_h, self.pool_w = pool_size
        
        # Horizontal strip pooling branch (pool across width)
        self.pool_h_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((self.pool_h, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Vertical strip pooling branch (pool across height)  
        self.pool_w_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, self.pool_w)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion convolution
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        _, _, h, w = x.shape
        
        # Horizontal strip pooling -> expand back
        x_h = self.pool_h_branch(x)
        x_h = F.interpolate(x_h, size=(h, w), mode='bilinear', align_corners=False)
        
        # Vertical strip pooling -> expand back
        x_w = self.pool_w_branch(x)
        x_w = F.interpolate(x_w, size=(h, w), mode='bilinear', align_corners=False)
        
        # Fuse horizontal and vertical
        out = x_h + x_w
        out = self.fusion(out)
        
        return out
