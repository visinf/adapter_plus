from typing import Callable, Optional, Tuple, Dict, Any
from timm.models._builder import (
    resolve_pretrained_cfg,
    _update_default_model_kwargs,
    adapt_model_from_file,
    load_pretrained,
    load_custom_pretrained,
    pretrained_cfg_for_features,
    FeatureHookNet,
    FeatureDictNet,
    FeatureGraphNet,
    FeatureGetterNet,
    FeatureListNet,
)

# modified from timm.models._builder.py to accept urls with .npz checkpoints
def build_model_with_cfg_compat(
    model_cls: Callable,
    variant: str,
    pretrained: bool,
    pretrained_cfg: Optional[Dict] = None,
    pretrained_cfg_overlay: Optional[Dict] = None,
    model_cfg: Optional[Any] = None,
    feature_cfg: Optional[Dict] = None,
    pretrained_strict: bool = True,
    pretrained_filter_fn: Optional[Callable] = None,
    kwargs_filter: Optional[Tuple[str]] = None,
    **kwargs,
):
    """Build model with specified default_cfg and optional model_cfg

    This helper fn aids in the construction of a model including:
      * handling default_cfg and associated pretrained weight loading
      * passing through optional model_cfg for models with config based arch spec
      * features_only model adaptation
      * pruning config / model adaptation

    Args:
        model_cls: model class
        variant: model variant name
        pretrained: load pretrained weights
        pretrained_cfg: model's pretrained weight/task config
        model_cfg: model's architecture config
        feature_cfg: feature extraction adapter config
        pretrained_strict: load pretrained weights strictly
        pretrained_filter_fn: filter callable for pretrained weights
        kwargs_filter: kwargs to filter before passing to model
        **kwargs: model args passed through to model __init__
    """
    pruned = kwargs.pop("pruned", False)
    features = False
    feature_cfg = feature_cfg or {}

    # resolve and update model pretrained config and model kwargs
    pretrained_cfg = resolve_pretrained_cfg(
        variant,
        pretrained_cfg=pretrained_cfg,
        pretrained_cfg_overlay=pretrained_cfg_overlay,
    )

    # FIXME converting back to dict, PretrainedCfg use should be propagated further, but not into model
    pretrained_cfg = pretrained_cfg.to_dict()

    _update_default_model_kwargs(pretrained_cfg, kwargs, kwargs_filter)

    # Setup for feature extraction wrapper done at end of this fn
    if kwargs.pop("features_only", False):
        features = True
        feature_cfg.setdefault("out_indices", (0, 1, 2, 3, 4))
        if "out_indices" in kwargs:
            feature_cfg["out_indices"] = kwargs.pop("out_indices")
        if "feature_cls" in kwargs:
            feature_cfg["feature_cls"] = kwargs.pop("feature_cls")

    # Instantiate the model
    if model_cfg is None:
        model = model_cls(**kwargs)
    else:
        model = model_cls(cfg=model_cfg, **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg  # alias for backwards compat

    if pruned:
        model = adapt_model_from_file(model, variant)

    # For classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = (
        0
        if features
        else getattr(model, "num_classes", kwargs.get("num_classes", 1000))
    )
    if pretrained:
        if "npz" in pretrained_cfg["url"]:
            # FIXME improve custom load trigger
            load_custom_pretrained(model, pretrained_cfg=pretrained_cfg)
        else:
            load_pretrained(
                model,
                pretrained_cfg=pretrained_cfg,
                num_classes=num_classes_pretrained,
                in_chans=kwargs.get("in_chans", 3),
                filter_fn=pretrained_filter_fn,
                strict=pretrained_strict,
            )

    # Wrap the model in a feature extraction module if enabled
    if features:
        use_getter = False
        if "feature_cls" in feature_cfg:
            feature_cls = feature_cfg.pop("feature_cls")
            if isinstance(feature_cls, str):
                feature_cls = feature_cls.lower()

                # flatten_sequential only valid for some feature extractors
                if feature_cls not in ("dict", "list", "hook"):
                    feature_cfg.pop("flatten_sequential", None)

                if "hook" in feature_cls:
                    feature_cls = FeatureHookNet
                elif feature_cls == "list":
                    feature_cls = FeatureListNet
                elif feature_cls == "dict":
                    feature_cls = FeatureDictNet
                elif feature_cls == "fx":
                    feature_cls = FeatureGraphNet
                elif feature_cls == "getter":
                    use_getter = True
                    feature_cls = FeatureGetterNet
                else:
                    assert False, f"Unknown feature class {feature_cls}"
        else:
            feature_cls = FeatureListNet

        output_fmt = getattr(model, "output_fmt", None)
        if (
            output_fmt is not None and not use_getter
        ):  # don't set default for intermediate feat getter
            feature_cfg.setdefault("output_fmt", output_fmt)

        model = feature_cls(model, **feature_cfg)
        model.pretrained_cfg = pretrained_cfg_for_features(
            pretrained_cfg
        )  # add back pretrained cfg
        model.default_cfg = (
            model.pretrained_cfg
        )  # alias for rename backwards compat (default_cfg -> pretrained_cfg)

    return model


