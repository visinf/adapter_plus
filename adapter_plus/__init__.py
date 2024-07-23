import timm
from .vit_adapter import *
from .vit_adapter import _create_vision_transformer_adapter

__all__ = [
    "Adapter",
    "LoRAAttention",
    "AdapterBlock",
    "AdapterResPostBlock",
    "VisionTransformerAdapter",
]


def patch_timm():
    timm.models.vision_transformer._create_vision_transformer = (
        _create_vision_transformer_adapter
    )


patch_timm()
