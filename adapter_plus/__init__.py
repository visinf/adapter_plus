import logging
import timm

try:
    import open_clip
except:
    logging.warning("Failed to import open_clip")
    open_clip = None
import importlib
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


def patch_clip():
    if open_clip is None:
        return
    from .clip_adapter import CLIPAdapter

    # patch the CLIP class instead of only the build_vision_tower function
    # CLIP doesn't allow to pass additional kwargs to the constructor.
    # and it uses load_state_dict with strict=True as default, which causes problems
    # when creating a model with adapters
    open_clip.model.CLIP = CLIPAdapter
    # had to reload the module because importing open_clip initializes also this module
    # and is has a static reference to the original CLIP class
    # this has to be overwritten
    importlib.reload(open_clip.factory)


patch_timm()
patch_clip()
