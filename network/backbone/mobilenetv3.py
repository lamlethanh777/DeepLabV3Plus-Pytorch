from functools import partial
from typing import Any, Callable, Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

try:  # for torchvision<0.4
    from torchvision.models.utils import load_state_dict_from_url
except:  # for torchvision>=0.4
    from torch.hub import load_state_dict_from_url


__all__ = ["MobileNetV3", "mobilenet_v3_large"]


model_urls = {
    "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth",
}


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        scale_activation: Callable[..., nn.Module] = nn.Hardsigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.activation = nn.ReLU()
        self.scale_activation = scale_activation()

    def forward(self, x: Tensor) -> Tensor:
        scale = self.avgpool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        scale = self.scale_activation(scale)
        return x * scale


class ConvBNActivation(nn.Sequential):
    """Convolution-BatchNorm-Activation block."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        if activation_layer is None:
            activation_layer = nn.ReLU

        # Calculate padding for same output size (considering dilation)
        # padding = (kernel_size - 1) // 2 * dilation
        # We'll use explicit padding in forward pass for dilated convolutions
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding=0,  # We'll pad explicitly
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            norm_layer(out_planes),
        ]
        if activation_layer is not None:
            layers.append(activation_layer(inplace=True))
        super().__init__(*layers)
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x: Tensor) -> Tensor:
        # Apply fixed padding to handle dilation properly
        padding = fixed_padding(self.kernel_size, self.dilation)
        x = F.pad(x, padding)
        return super().forward(x)


def fixed_padding(kernel_size: int, dilation: int) -> tuple:
    """Calculate fixed padding for a convolution with dilation."""
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return (pad_beg, pad_end, pad_beg, pad_end)


class InvertedResidualConfig:
    """Stores information for MobileNetV3 inverted residual blocks."""

    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float) -> int:
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    """MobileNetV3 Inverted Residual block."""

    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                ConvBNActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise - use stride=1 if dilation > 1
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            ConvBNActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(SqueezeExcitation(cnf.expanded_channels, squeeze_channels))

        # project
        layers.append(
            ConvBNActivation(
                cnf.expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x
        return result


class MobileNetV3(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            dropout (float): The dropout probability
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvBNActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        layers.append(
            ConvBNActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        self.features = nn.Sequential(*layers)
        self.last_channel = last_channel
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _mobilenet_v3_conf(
    arch: str,
    output_stride: int = 8,
    width_mult: float = 1.0,
    reduced_tail: bool = False,
    **kwargs: Any,
):
    """Build the inverted residual configuration for MobileNetV3.

    Args:
        arch: Architecture type ('mobilenet_v3_large')
        output_stride: The output stride of the backbone (8 or 16)
        width_mult: Width multiplier
        reduced_tail: Whether to reduce the tail
    """
    reduce_divider = 2 if reduced_tail else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(
        InvertedResidualConfig.adjust_channels, width_mult=width_mult
    )

    if arch == "mobilenet_v3_large":
        # MobileNetV3-Large configuration
        # The network has 5 stages with strides [2, 2, 2, 2, 2]
        # For output_stride=8: apply dilation starting from stage with cumulative stride > 8
        # For output_stride=16: apply dilation starting from stage with cumulative stride > 16

        # Current stride tracking:
        # Initial conv: stride 2 -> cumulative stride = 2
        # Layer 0 (stride 1): cumulative = 2
        # Layer 1 (stride 2): cumulative = 4 (C1)
        # Layer 2 (stride 1): cumulative = 4
        # Layer 3 (stride 2): cumulative = 8 (C2)
        # Layer 4-5 (stride 1): cumulative = 8
        # Layer 6 (stride 2): cumulative = 16 (C3)
        # Layer 7-11 (stride 1): cumulative = 16
        # Layer 12 (stride 2): cumulative = 32 (C4)
        # Layer 13-14 (stride 1): cumulative = 32

        if output_stride == 8:
            # Apply dilation to layers that would have cumulative stride > 8
            # Layer 6 (stride 2) would bring us to 16 - use dilation instead
            # Layer 12 (stride 2) would bring us to 32 - use dilation instead
            inverted_residual_setting = [
                bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),  # 0
                bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # 1 - C1 (stride 4)
                bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),  # 2
                bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # 3 - C2 (stride 8)
                bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),  # 4
                bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),  # 5
                bneck_conf(
                    40, 3, 240, 80, False, "HS", 2, 2
                ),  # 6 - C3 (dilated, stride=1)
                bneck_conf(80, 3, 200, 80, False, "HS", 1, 2),  # 7
                bneck_conf(80, 3, 184, 80, False, "HS", 1, 2),  # 8
                bneck_conf(80, 3, 184, 80, False, "HS", 1, 2),  # 9
                bneck_conf(80, 3, 480, 112, True, "HS", 1, 2),  # 10
                bneck_conf(112, 3, 672, 112, True, "HS", 1, 2),  # 11
                bneck_conf(
                    112, 5, 672, 160 // reduce_divider, True, "HS", 2, 4
                ),  # 12 - C4 (dilated)
                bneck_conf(
                    160 // reduce_divider,
                    5,
                    960 // reduce_divider,
                    160 // reduce_divider,
                    True,
                    "HS",
                    1,
                    4,
                ),  # 13
                bneck_conf(
                    160 // reduce_divider,
                    5,
                    960 // reduce_divider,
                    160 // reduce_divider,
                    True,
                    "HS",
                    1,
                    4,
                ),  # 14
            ]
        else:  # output_stride == 16
            # Apply dilation only to the last stage (layer 12+)
            inverted_residual_setting = [
                bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),  # 0
                bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # 1 - C1
                bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),  # 2
                bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # 3 - C2
                bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),  # 4
                bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),  # 5
                bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # 6 - C3
                bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),  # 7
                bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),  # 8
                bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),  # 9
                bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),  # 10
                bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),  # 11
                bneck_conf(
                    112, 5, 672, 160 // reduce_divider, True, "HS", 2, 2
                ),  # 12 - C4 (dilated)
                bneck_conf(
                    160 // reduce_divider,
                    5,
                    960 // reduce_divider,
                    160 // reduce_divider,
                    True,
                    "HS",
                    1,
                    2,
                ),  # 13
                bneck_conf(
                    160 // reduce_divider,
                    5,
                    960 // reduce_divider,
                    160 // reduce_divider,
                    True,
                    "HS",
                    1,
                    2,
                ),  # 14
            ]
        last_channel = adjust_channels(1280 // reduce_divider)
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def mobilenet_v3_large(
    pretrained: bool = False,
    progress: bool = True,
    output_stride: int = 8,
    **kwargs: Any,
) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        output_stride (int): The output stride of the backbone (8 or 16)
    """
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(
        "mobilenet_v3_large", output_stride=output_stride, **kwargs
    )
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls["mobilenet_v3_large"], progress=progress
        )
        # Load pretrained weights, allowing for mismatches due to dilation changes
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model
