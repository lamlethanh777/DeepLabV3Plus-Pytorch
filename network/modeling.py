from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabHeadV3PlusWithECA, DeepLabV3
from .backbone import resnet, mobilenetv2, mobilenetv3, hrnetv2, xception


def _segm_hrnet(name, backbone_name, num_classes, pretrained_backbone):
    backbone = hrnetv2.__dict__[backbone_name](pretrained_backbone)
    # HRNetV2 config:
    # the final output channels is dependent on highest resolution channel config (c).
    # output of backbone will be the inplanes to assp:
    hrnet_channels = int(backbone_name.split("_")[-1])
    inplanes = sum(hrnet_channels * 2**i for i in range(4))
    aspp_dilate = [12, 24, 36]  # If follow paper trend, can put [24, 48, 72].

    if name == "deeplabv3plus":
        return_layers = {"stage4": "out", "layer1": "low_level"}
        low_level_planes = (
            256  # all hrnet version channel output from bottleneck is the same
        )
        classifier = DeepLabHeadV3Plus(
            inplanes, low_level_planes, num_classes, aspp_dilate
        )
    elif name == "deeplabv3":
        return_layers = {"stage4": "out"}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)

    backbone = IntermediateLayerGetter(
        backbone, return_layers=return_layers, hrnet_flag=True
    )
    return DeepLabV3(backbone, classifier)


def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation,
    )

    inplanes = 2048
    if name == "deeplabv3plus":
        return_layers = {"layer4": "out", "layer1": "low_level"}
        low_level_planes = 256

        classifier = DeepLabHeadV3Plus(
            inplanes, low_level_planes, num_classes, aspp_dilate
        )
    elif name == "deeplabv3":
        return_layers = {"layer4": "out"}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    return DeepLabV3(backbone, classifier)


def _segm_xception(
    name, backbone_name, num_classes, output_stride, pretrained_backbone
):
    if output_stride == 8:
        replace_stride_with_dilation = [False, False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = xception.xception(
        pretrained="imagenet" if pretrained_backbone else False,
        replace_stride_with_dilation=replace_stride_with_dilation,
    )

    inplanes = 2048
    low_level_planes = 128

    if name == "deeplabv3plus":
        return_layers = {"conv4": "out", "block1": "low_level"}
        classifier = DeepLabHeadV3Plus(
            inplanes, low_level_planes, num_classes, aspp_dilate
        )
    elif name == "deeplabv3":
        return_layers = {"conv4": "out"}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    model = DeepLabV3(backbone, classifier)
    return model


def _segm_mobilenet(
    name, backbone_name, num_classes, output_stride, pretrained_backbone
):
    aspp_dilate = [12, 24, 36] if output_stride == 8 else [6, 12, 18]
    backbone = mobilenetv2.mobilenet_v2(
        pretrained=pretrained_backbone, output_stride=output_stride
    )

    # rename layers
    backbone.low_level_features = backbone.features[:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    if name == "deeplabv3plus":
        return_layers = {
            "high_level_features": "out",
            "low_level_features": "low_level",
        }
        low_level_planes = 24

        classifier = DeepLabHeadV3Plus(
            inplanes, low_level_planes, num_classes, aspp_dilate
        )
    elif name == "deeplabv3":
        return_layers = {"high_level_features": "out"}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    return DeepLabV3(backbone, classifier)


def _segm_mobilenet_v3(
    name, backbone_name, num_classes, output_stride, pretrained_backbone
):
    aspp_dilate = [12, 24, 36] if output_stride == 8 else [6, 12, 18]
    backbone = mobilenetv3.mobilenet_v3_large(
        pretrained=pretrained_backbone, output_stride=output_stride
    )

    # MobileNetV3-Large feature structure:
    # features[0]: first conv layer (16 channels, stride 2)
    # features[1]: 16 channels
    # features[2-3]: 24 channels (C1, stride 4)
    # features[4-6]: 40 channels (C2, stride 8)
    # features[7-10]: 80 channels
    # features[11-12]: 112 channels (C3)
    # features[13-15]: 160 channels (C4)
    # features[16]: last conv (960 channels)

    # For low_level_features: use layers 0-3 (outputs 24 channels at stride 4)
    # For high_level_features: layers 4-15 (outputs 160 channels)
    backbone.low_level_features = backbone.features[:4]  # Outputs 24 channels
    backbone.high_level_features = backbone.features[4:-1]  # Outputs 160 channels
    backbone.features = None
    backbone.classifier = None
    backbone.avgpool = None

    inplanes = 160
    if name == "deeplabv3plus":
        return_layers = {
            "high_level_features": "out",
            "low_level_features": "low_level",
        }
        low_level_planes = 24  # MobileNetV3-Large has 24 channels at layer 3

        classifier = DeepLabHeadV3Plus(
            inplanes, low_level_planes, num_classes, aspp_dilate
        )
    elif name == "deeplabv3":
        return_layers = {"high_level_features": "out"}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    return DeepLabV3(backbone, classifier)


def _segm_mobilenet_attention(
    name, backbone_name, num_classes, output_stride, pretrained_backbone,
    use_shuffle_attention=True, shuffle_attention_groups=64
):
    """MobileNetV2 backbone with attention-augmented DeepLabV3+ head."""
    aspp_dilate = [12, 24, 36] if output_stride == 8 else [6, 12, 18]
    backbone = mobilenetv2.mobilenet_v2(
        pretrained=pretrained_backbone, output_stride=output_stride
    )

    # rename layers
    backbone.low_level_features = backbone.features[:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    if name == "deeplabv3plus_attention":
        return_layers = {
            "high_level_features": "out",
            "low_level_features": "low_level",
        }
        low_level_planes = 24

        classifier = DeepLabHeadV3PlusWithECA(
            inplanes, low_level_planes, num_classes, aspp_dilate,
            use_shuffle_attention=use_shuffle_attention,
            shuffle_attention_groups=shuffle_attention_groups
        )
    else:
        raise ValueError(f"Unknown architecture: {name}")
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    return DeepLabV3(backbone, classifier)


def _segm_mobilenet_v3_attention(
    name, backbone_name, num_classes, output_stride, pretrained_backbone,
    use_shuffle_attention=True, shuffle_attention_groups=64
):
    """MobileNetV3-Large backbone with attention-augmented DeepLabV3+ head."""
    aspp_dilate = [12, 24, 36] if output_stride == 8 else [6, 12, 18]
    backbone = mobilenetv3.mobilenet_v3_large(
        pretrained=pretrained_backbone, output_stride=output_stride
    )

    # MobileNetV3-Large feature structure:
    # features[0]: first conv layer (16 channels, stride 2)
    # features[1]: 16 channels
    # features[2-3]: 24 channels (C1, stride 4)
    # features[4-6]: 40 channels (C2, stride 8)
    # features[7-10]: 80 channels
    # features[11-12]: 112 channels (C3)
    # features[13-15]: 160 channels (C4)
    # features[16]: last conv (960 channels)

    # For low_level_features: use layers 0-3 (outputs 24 channels at stride 4)
    # For high_level_features: layers 4-15 (outputs 160 channels)
    backbone.low_level_features = backbone.features[:4]  # Outputs 24 channels
    backbone.high_level_features = backbone.features[4:-1]  # Outputs 160 channels
    backbone.features = None
    backbone.classifier = None
    backbone.avgpool = None

    inplanes = 160
    if name == "deeplabv3plus_attention":
        return_layers = {
            "high_level_features": "out",
            "low_level_features": "low_level",
        }
        low_level_planes = 24  # MobileNetV3-Large has 24 channels at layer 3

        classifier = DeepLabHeadV3PlusWithECA(
            inplanes, low_level_planes, num_classes, aspp_dilate,
            use_shuffle_attention=use_shuffle_attention,
            shuffle_attention_groups=shuffle_attention_groups
        )
    else:
        raise ValueError(f"Unknown architecture: {name}")
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    return DeepLabV3(backbone, classifier)


def _segm_resnet_attention(
    name, backbone_name, num_classes, output_stride, pretrained_backbone,
    use_shuffle_attention=True, shuffle_attention_groups=64
):
    """ResNet backbone with attention-augmented DeepLabV3+ head."""
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation,
    )

    inplanes = 2048
    if name == "deeplabv3plus_attention":
        return_layers = {"layer4": "out", "layer1": "low_level"}
        low_level_planes = 256

        classifier = DeepLabHeadV3PlusWithECA(
            inplanes, low_level_planes, num_classes, aspp_dilate,
            use_shuffle_attention=use_shuffle_attention,
            shuffle_attention_groups=shuffle_attention_groups
        )
    else:
        raise ValueError(f"Unknown architecture: {name}")
    
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    return DeepLabV3(backbone, classifier)


def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):
    if backbone == "mobilenetv2":
        model = _segm_mobilenet(
            arch_type,
            backbone,
            num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
        )
    elif backbone == "mobilenetv3_large":
        model = _segm_mobilenet_v3(
            arch_type,
            backbone,
            num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
        )
    elif backbone.startswith("resnet"):
        model = _segm_resnet(
            arch_type,
            backbone,
            num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
        )
    elif backbone.startswith("hrnetv2"):
        model = _segm_hrnet(
            arch_type, backbone, num_classes, pretrained_backbone=pretrained_backbone
        )
    elif backbone == "xception":
        model = _segm_xception(
            arch_type,
            backbone,
            num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
        )
    else:
        raise NotImplementedError
    return model


# Deeplab v3
def deeplabv3_hrnetv2_48(
    num_classes=21, output_stride=4, pretrained_backbone=False
):  # no pretrained backbone yet
    return _load_model(
        "deeplabv3",
        "hrnetv2_48",
        output_stride,
        num_classes,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True):
    return _load_model(
        "deeplabv3",
        "hrnetv2_32",
        output_stride,
        num_classes,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model(
        "deeplabv3",
        "resnet50",
        num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model(
        "deeplabv3",
        "resnet101",
        num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3_mobilenet(
    num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs
):
    """Constructs a DeepLabV3 model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model(
        "deeplabv3",
        "mobilenetv2",
        num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3_mobilenet_v3_large(
    num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs
):
    """Constructs a DeepLabV3 model with a MobileNetV3-Large backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model(
        "deeplabv3",
        "mobilenetv3_large",
        num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3_xception(
    num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs
):
    """Constructs a DeepLabV3 model with a Xception backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model(
        "deeplabv3",
        "xception",
        num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


# Deeplab v3+
def deeplabv3plus_hrnetv2_48(
    num_classes=21, output_stride=4, pretrained_backbone=False
):  # no pretrained backbone yet
    return _load_model(
        "deeplabv3plus",
        "hrnetv2_48",
        num_classes,
        output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3plus_hrnetv2_32(num_classes=21, output_stride=4, pretrained_backbone=True):
    return _load_model(
        "deeplabv3plus",
        "hrnetv2_32",
        num_classes,
        output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model(
        "deeplabv3plus",
        "resnet50",
        num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model(
        "deeplabv3plus",
        "resnet101",
        num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model(
        "deeplabv3plus",
        "mobilenetv2",
        num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3plus_mobilenet_v3_large(
    num_classes=21, output_stride=8, pretrained_backbone=True
):
    """Constructs a DeepLabV3+ model with a MobileNetV3-Large backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model(
        "deeplabv3plus",
        "mobilenetv3_large",
        num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def deeplabv3plus_xception(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a Xception backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model(
        "deeplabv3plus",
        "xception",
        num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


# DeepLabV3+ with Attention
def deeplabv3plus_mobilenet_attention(
    num_classes=21, output_stride=8, pretrained_backbone=True,
    use_shuffle_attention=True, shuffle_attention_groups=64
):
    """Constructs a DeepLabV3+ model with MobileNetV2 backbone and attention modules.

    Features Shuffle Attention in ASPP (after 5-branch concatenation) and 
    ECA Attention after encoder-decoder feature fusion.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        use_shuffle_attention (bool): Use ShuffleAttention in ASPP. Default: True.
        shuffle_attention_groups (int): Number of groups for ShuffleAttention. Default: 64.
    """
    return _segm_mobilenet_attention(
        "deeplabv3plus_attention",
        "mobilenetv2",
        num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
        use_shuffle_attention=use_shuffle_attention,
        shuffle_attention_groups=shuffle_attention_groups,
    )


def deeplabv3plus_resnet50_attention(
    num_classes=21, output_stride=8, pretrained_backbone=True,
    use_shuffle_attention=True, shuffle_attention_groups=64
):
    """Constructs a DeepLabV3+ model with ResNet-50 backbone and attention modules.

    Features Shuffle Attention in ASPP (after 5-branch concatenation) and 
    ECA Attention after encoder-decoder feature fusion.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        use_shuffle_attention (bool): Use ShuffleAttention in ASPP. Default: True.
        shuffle_attention_groups (int): Number of groups for ShuffleAttention. Default: 64.
    """
    return _segm_resnet_attention(
        "deeplabv3plus_attention",
        "resnet50",
        num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
        use_shuffle_attention=use_shuffle_attention,
        shuffle_attention_groups=shuffle_attention_groups,
    )


def deeplabv3plus_resnet101_attention(
    num_classes=21, output_stride=8, pretrained_backbone=True,
    use_shuffle_attention=True, shuffle_attention_groups=64
):
    """Constructs a DeepLabV3+ model with ResNet-101 backbone and attention modules.

    Features Shuffle Attention in ASPP (after 5-branch concatenation) and 
    ECA Attention after encoder-decoder feature fusion.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        use_shuffle_attention (bool): Use ShuffleAttention in ASPP. Default: True.
        shuffle_attention_groups (int): Number of groups for ShuffleAttention. Default: 64.
    """
    return _segm_resnet_attention(
        "deeplabv3plus_attention",
        "resnet101",
        num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
        use_shuffle_attention=use_shuffle_attention,
        shuffle_attention_groups=shuffle_attention_groups,
    )


def deeplabv3plus_mobilenet_v3_large_attention(
    num_classes=21, output_stride=8, pretrained_backbone=True,
    use_shuffle_attention=True, shuffle_attention_groups=64
):
    """Constructs a DeepLabV3+ model with MobileNetV3-Large backbone and attention modules.

    Features Shuffle Attention in ASPP (after 5-branch concatenation) and 
    ECA Attention after encoder-decoder feature fusion.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
        use_shuffle_attention (bool): Use ShuffleAttention in ASPP. Default: True.
        shuffle_attention_groups (int): Number of groups for ShuffleAttention. Default: 64.
    """
    return _segm_mobilenet_v3_attention(
        "deeplabv3plus_attention",
        "mobilenetv3_large",
        num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
        use_shuffle_attention=use_shuffle_attention,
        shuffle_attention_groups=shuffle_attention_groups,
    )
