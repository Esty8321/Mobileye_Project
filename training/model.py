from typing import Optional
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


def build_model(num_classes: int,
                dropout: float = 0.3,
                pretrained: bool = True) -> nn.Module:
    """
    Build EfficientNetV2-S with a custom classifier head.

    Args:
        num_classes: Number of output classes for the classifier head.
        dropout: Dropout probability before the final linear layer.
        pretrained: If True, load ImageNet-1K weights for the backbone.

    Returns:
        nn.Module: EfficientNetV2-S model whose head matches `num_classes`.
    """
    weights: Optional[EfficientNet_V2_S_Weights] = (
        EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    )
    m = efficientnet_v2_s(weights=weights)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
    return m


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """
    Freeze/unfreeze all backbone layers (everything except the classifier head).

    Args:
        model: The EfficientNet model.
        freeze: If True, freeze backbone; if False, unfreeze backbone.
    """
    for name, p in model.named_parameters():
        if not name.startswith("classifier."):
            p.requires_grad = not freeze


def count_trainable_params(model: nn.Module) -> int:
    """
    Count trainable parameters of a model.

    Args:
        model: Any nn.Module.

    Returns:
        int: Number of parameters with `requires_grad=True`.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
