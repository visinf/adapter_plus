from typing import Optional, Callable, Tuple
from functools import partial

from open_clip.transformer import (
    LayerNorm,
    LayerNormFp32,
    QuickGELU,
    ResidualAttentionBlock,
    Transformer,
    VisionTransformer,
)
from open_clip.model import CLIP, CLIPVisionCfg, CLIPTextCfg
from open_clip.timm_model import TimmModel
from open_clip.modified_resnet import ModifiedResNet

import torch
from torch import Tensor, nn

import numpy as np
import math
import logging

from adapter_plus.vit_adapter import Adapter


# TODO: add support for LoRA in CLIP
# this is work in progress
class LoRAAttention(nn.MultiheadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
        # LoRA specific parameters
        lora_config=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        qkv_bias=False,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.lora_config = lora_config
        self.adapters = nn.ModuleDict(
            {
                l: Adapter(
                    embed_dim=embed_dim,
                    bottleneck_dim=lora_config.dim,
                    drop_path=lora_config.drop_path,
                    dropout=lora_config.dropout,
                    act_layer=act_layer if lora_config.act_layer else None,
                    norm_layer=norm_layer if lora_config.norm_layer else None,
                    scaling=lora_config.scaling,
                    init=lora_config.init,
                    bias=((qkv_bias or l == "o") and lora_config.bias),
                    pre_dropout=lora_config.pre_dropout,
                )
                for l in lora_config.location
            }
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # TODO: check if this is correct

        # apply adapters to query, key and value
        if "q" in self.adapters:
            query = self.adapters["q"](query, skip=query)
        if "k" in self.adapters:
            key = self.adapters["k"](key, skip=key)
        if "v" in self.adapters:
            value = self.adapters["v"](value, skip=value)

        attn_output, attn_output_weights = super().forward(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )

        if "o" in self.adapters:
            attn_output = self.adapters["o"](attn_output, skip=attn_output)

        return attn_output, attn_output_weights


###### CLIP Vision Transformer Adapter ######


class AdapterResidualAttentionBlock(ResidualAttentionBlock):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        is_cross_attention: bool = False,
        batch_first: bool = True,
        # Add adapter specific parameters
        adapter_config=None,
        lora_config=None,
        prompt_config=None,
        # Lora specific parameters
        qkv_bias: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        # Prompt specific parameters
        patch_size: int = 16,
    ):
        super().__init__(
            d_model=d_model,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            is_cross_attention=is_cross_attention,
            batch_first=batch_first,
        )
        # Add adapter specific parameters
        self.adapter_config = adapter_config
        self.lora_config = lora_config
        self.prompt_config = prompt_config

        if adapter_config is not None:
            self.adapter = Adapter(
                d_model,
                bottleneck_dim=adapter_config.dim,
                dropout=adapter_config.dropout,
                drop_path=adapter_config.drop_path,
                act_layer=act_layer if adapter_config.act_layer else None,
                norm_layer=norm_layer if adapter_config.norm_layer else None,
                bias=adapter_config.bias,
                scaling=adapter_config.scaling,
                init=adapter_config.init,
            )
            if adapter_config.attn_adapter:
                self.adapter_attn = Adapter(
                    d_model,
                    bottleneck_dim=adapter_config.dim,
                    dropout=adapter_config.dropout,
                    drop_path=adapter_config.drop_path,
                    act_layer=act_layer if adapter_config.act_layer else None,
                    norm_layer=norm_layer if adapter_config.norm_layer else None,
                    bias=adapter_config.bias,
                    scaling=adapter_config.scaling,
                    init=adapter_config.init,
                )

        # TODO: Add LoRA support
        if lora_config is not None and lora_config.config == "attention":
            raise NotImplementedError("LoRA is not implemented yet")
            # self.attn = LoRAAttention(
            #     d_model,
            #     num_heads=n_head,
            #     qkv_bias=qkv_bias,
            #     attn_drop=attn_drop,
            #     proj_drop=proj_drop,
            #     act_layer=act_layer,
            #     norm_layer=norm_layer,
            #     lora_config=lora_config,
            # )

        if prompt_config is not None:
            self.prompt = nn.Parameter(torch.zeros(prompt_config.num_tokens, d_model))
            self.prompt_dropout = nn.Dropout(prompt_config.dropout)

            val = math.sqrt(6.0 / float(3 * patch_size**2 + prompt_config.num_tokens))
            nn.init.uniform_(self.prompt, -val, val)

    def forward_post(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = (
            self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        )
        v_x = (
            self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        )
        x = q_x + self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        x = self.adapter(x, skip=x)
        return x

    def forward_pre(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = (
            self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        )
        v_x = (
            self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        )
        x = q_x + self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        x = self.adapter(x, skip=x)
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x

    def forward_pfeiffer(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = (
            self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        )
        v_x = (
            self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        )
        y = q_x + self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        skip = self.ls_2(self.mlp(self.ln_2(y)))
        x = y + skip
        x = self.adapter(x, skip=skip)
        x = x + y
        return x

    def forward_intermediate(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = (
            self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        )
        v_x = (
            self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        )
        x = q_x + self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        y = self.ls_2(self.mlp(self.ln_2(x)))
        return x + self.adapter(y, skip=y)

    def forward_houlsby(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = (
            self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        )
        v_x = (
            self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        )
        y = self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        x = q_x + self.adapter_attn(y, skip=y)
        y = self.ls_2(self.mlp(self.ln_2(x)))
        return x + self.adapter(y, skip=y)

    def forward_parallel(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = (
            self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        )
        v_x = (
            self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        )
        y = q_x + self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        z = y + self.ls_2(self.mlp(self.ln_2(y)))
        return self.adapter(y, skip=z)

    # original forward function
    def forward_no_adapter(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = (
            self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        )
        v_x = (
            self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        )
        x = q_x + self.ls_1(
            self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        )
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x

    def include_prompt(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = torch.cat(
            (
                x[:, :1, :],
                self.prompt_dropout(self.prompt.expand(B, -1, -1)),
                x[:, (1 + self.prompt_config.num_tokens) :, :],
            ),
            dim=1,
        )
        return x

    def forward(
        self,
        q_x: torch.Tensor,
        k_x: Optional[torch.Tensor] = None,
        v_x: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        if self.prompt_config:
            q_x = self.include_prompt(q_x)

        if self.adapter_config is None:
            return self.forward_no_adapter(q_x, k_x, v_x, attn_mask)
        elif self.adapter_config.config == "pfeiffer":
            return self.forward_pfeiffer(q_x, k_x, v_x, attn_mask)
        elif self.adapter_config.config == "post":
            return self.forward_post(q_x, k_x, v_x, attn_mask)
        elif self.adapter_config.config == "pre":
            return self.forward_pre(q_x, k_x, v_x, attn_mask)
        elif self.adapter_config.config == "houlsby":
            return self.forward_houlsby(q_x, k_x, v_x, attn_mask)
        elif self.adapter_config.config == "intermediate":
            return self.forward_intermediate(q_x, k_x, v_x, attn_mask)
        elif self.adapter_config.config == "parallel":
            return self.forward_parallel(q_x, k_x, v_x, attn_mask)
        else:
            raise ValueError(f"Unknown adapter config: {self.adapter_config.config}")


# modified from open_clip/transformer.py class Transformer
class TransformerAdapter(Transformer):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        batch_first: bool = True,
        adapter_config=None,
        lora_config=None,
        prompt_config=None,
    ):
        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            batch_first=batch_first,
        )
        # Add adapter specific parameters
        self.adapter_config = adapter_config
        self.lora_config = lora_config
        self.prompt_config = prompt_config

        self.resblocks = nn.ModuleList(
            [
                AdapterResidualAttentionBlock(
                    d_model=width,
                    n_head=heads,
                    mlp_ratio=mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    batch_first=batch_first,
                    adapter_config=adapter_config,
                    lora_config=lora_config,
                    prompt_config=prompt_config,
                )
                for _ in range(layers)
            ]
        )


class VisionTransformerAdapter(VisionTransformer):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        attentional_pool: bool = False,
        attn_pooler_queries: int = 256,
        attn_pooler_heads: int = 8,
        output_dim: int = 512,
        patch_dropout: float = 0.0,
        no_ln_pre: bool = False,
        pos_embed_type: str = "learnable",
        pool_type: str = "tok",
        final_ln_after_pool: bool = False,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        output_tokens: bool = False,
        # Add adapter specific parameters
        adapter_config=None,
        lora_config=None,
        prompt_config=None,
        **kwargs,
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            attentional_pool=attentional_pool,
            attn_pooler_queries=attn_pooler_queries,
            attn_pooler_heads=attn_pooler_heads,
            output_dim=output_dim,
            patch_dropout=patch_dropout,
            no_ln_pre=no_ln_pre,
            pos_embed_type=pos_embed_type,
            pool_type=pool_type,
            final_ln_after_pool=final_ln_after_pool,
            act_layer=act_layer,
            norm_layer=norm_layer,
            output_tokens=output_tokens,
        )
        # init adapter specific parameters
        self.adapter_config = adapter_config
        self.lora_config = lora_config
        self.prompt_config = prompt_config

        # overwrite the transformer with the adapter transformer
        self.transformer = TransformerAdapter(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            adapter_config=adapter_config,
            lora_config=lora_config,
            prompt_config=prompt_config,
        )


# NOTE: adapted from open_clip.model._build_vision_tower
def _build_vision_tower_adapter(
    embed_dim: int,
    vision_cfg: CLIPVisionCfg,
    quick_gelu: bool = False,
    cast_dtype: Optional[torch.dtype] = None,
    adapter=True,
    adapter_config=None,
    lora_config=None,
    prompt_config=None,
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            drop=vision_cfg.timm_drop,
            drop_path=vision_cfg.timm_drop_path,
            patch_drop=(
                vision_cfg.patch_dropout if vision_cfg.patch_dropout > 0 else None
            ),
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    elif adapter:
        # use adapters
        # rest of this function is unchanged
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = (
            LayerNormFp32
            if cast_dtype in (torch.float16, torch.bfloat16)
            else LayerNorm
        )
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformerAdapter(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            adapter_config=adapter_config,
            lora_config=lora_config,
            prompt_config=prompt_config,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = (
            LayerNormFp32
            if cast_dtype in (torch.float16, torch.bfloat16)
            else LayerNorm
        )
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)

        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


class CLIPAdapter(CLIP):
    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        init_logit_scale: float = np.log(1 / 0.07),
        init_logit_bias: Optional[float] = None,
        nonscalar_logit_scale: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
        output_dict: bool = False,
        # add adapter specific parameters
        adapter=True,
        adapter_config=None,
        lora_config=None,
        prompt_config=None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            init_logit_scale=init_logit_scale,
            init_logit_bias=init_logit_bias,
            nonscalar_logit_scale=nonscalar_logit_scale,
            cast_dtype=cast_dtype,
            output_dict=output_dict,
        )
        # add adapter specific parameters
        self.adapter = adapter
        self.adapter_config = adapter_config
        self.lora_config = lora_config
        self.prompt_config = prompt_config

        # replace the vision tower with the adapter vision tower
        self.visual = _build_vision_tower_adapter(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
            adapter=adapter,
            adapter_config=adapter_config,
            lora_config=lora_config,
            prompt_config=prompt_config,
        )

    def load_state_dict(self, state_dict, strict: bool = False, assign: bool = False):
        # overwrite load_state_dict from nn.Module to allow loading pretrained weights with strict=False
        # open_clip does not use strict=False by default
        # but we need to load pretrained weights with strict=False to include the adapter weights
        strict = False
        load_result = super().load_state_dict(
            state_dict=state_dict, strict=strict, assign=assign
        )
        if load_result.missing_keys:
            logging.info(
                f'Missing keys ({", ".join(load_result.missing_keys)}) discovered while loading pretrained weights.'
                f" This is expected if model is being adapted."
            )
        if load_result.unexpected_keys:
            logging.warning(
                f'Unexpected keys ({", ".join(load_result.unexpected_keys)}) found while loading pretrained weights.'
                f" This may be expected if model is being adapted."
            )
        return load_result
