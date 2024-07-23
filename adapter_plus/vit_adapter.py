import math
from functools import partial
from typing import Optional

import torch
from torch import nn

from timm.models.layers import DropPath, PatchEmbed, Mlp
from timm.models.helpers import checkpoint_seq
from timm.models.vision_transformer import (
    Block,
    ResPostBlock,
    VisionTransformer,
    checkpoint_filter_fn,
    build_model_with_cfg,
)

# for Adapter+ set norm_layer to None and scaling to "channel"
class Adapter(nn.Module):
    def __init__(
        self,
        embed_dim,
        bottleneck_dim=8,
        drop_path=0.0,
        dropout=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        scaling=1.0,
        init="houlsby",
        bias=True,
        pre_dropout=False,
    ):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 and pre_dropout else nn.Identity(),
            nn.Linear(embed_dim, bottleneck_dim, bias=bias),
            act_layer() if act_layer else nn.Identity(),
            nn.Dropout(dropout) if dropout > 0 and not pre_dropout else nn.Identity(),
            nn.Linear(bottleneck_dim, embed_dim, bias=bias),
        )
        self.norm_a = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.drop_path_a = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.bottleneck_dim = bottleneck_dim
        if scaling == "learned":
            self.scaling = nn.Parameter(torch.ones(1))
        elif scaling == "channel":
            self.scaling = nn.Parameter(torch.ones(embed_dim))
        else:
            self.scaling = scaling

        # init following (Houslby 2019)
        if init == "houlsby":
            std = 0.01  # paper value, houlsby code implementation: std = 0.001
            nn.init.trunc_normal_(
                self.bottleneck[1].weight, std=std, a=-2 * std, b=2 * std
            )
            if self.bottleneck[1].bias is not None:
                nn.init.zeros_(self.bottleneck[1].bias)
            nn.init.trunc_normal_(
                self.bottleneck[4].weight, std=std, a=-2 * std, b=2 * std
            )
            if self.bottleneck[4].bias is not None:
                nn.init.zeros_(self.bottleneck[4].bias)

        # LoRa init
        elif init == "lora":
            # leave in projection with default init
            nn.init.kaiming_uniform_(self.bottleneck[1].weight, a=math.sqrt(5))
            if self.bottleneck[1].bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                    self.bottleneck[1].weight
                )
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bottleneck[1].bias, -bound, bound)
            # set out projection to zeros
            nn.init.zeros_(self.bottleneck[4].weight)
            if self.bottleneck[4].bias is not None:
                nn.init.zeros_(self.bottleneck[4].bias)

        elif init == "bert":
            nn.init.normal_(self.bottleneck[1].weight, mean=0.0, std=0.02)
            if self.bottleneck[1].bias is not None:
                nn.init.zeros_(self.bottleneck[1].bias)
            nn.init.normal_(self.bottleneck[4].weight, mean=0.0, std=0.02)
            if self.bottleneck[4].bias is not None:
                nn.init.zeros_(self.bottleneck[4].bias)

        else:
            raise ValueError(f"Initialization {init} not implemented!")

    def forward(
        self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.norm_a(x)
        x = self.drop_path_a(self.bottleneck(x))
        x = x * self.scaling

        y = x
        if skip is not None:
            y = y + skip

        return y


class LoRAAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        lora_config=None,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.lora_config = lora_config
        self.adapters = nn.ModuleDict(
            {
                l: Adapter(
                    dim,
                    bottleneck_dim=lora_config.dim,
                    act_layer=act_layer if lora_config.act_layer else None,
                    norm_layer=norm_layer if lora_config.norm_layer else None,
                    scaling=lora_config.scaling,
                    bias=((qkv_bias or l == "o") and lora_config.bias),
                    drop_path=lora_config.drop_path,
                    dropout=lora_config.dropout,
                    pre_dropout=lora_config.pre_dropout,
                )
                for l in lora_config.location
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)  # 3, B, num_heads, N, C // num_heads
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if "q" in self.adapters:
            q = q + (
                self.adapters["q"](x)
                .reshape(B, N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)  # B, num_heads, N, C // num_heads
            )
        if "k" in self.adapters:
            k = k + (
                self.adapters["k"](x)
                .reshape(B, N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)  # B, num_heads, N, C // num_heads
            )
        if "v" in self.adapters:
            v = v + (
                self.adapters["v"](x)
                .reshape(B, N, self.num_heads, C // self.num_heads)
                .permute(0, 2, 1, 3)  # B, num_heads, N, C // num_heads
            )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if "o" in self.adapters:
            x = x + self.adapters["o"](x)
        x = self.proj_drop(x)
        return x


class AdapterBlock(Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        adapter_config=None,
        lora_config=None,
        prompt_config=None,
        patch_size=16,
    ):
        super().__init__(
            dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )

        self.adapter_config = adapter_config
        self.lora_config = lora_config
        self.prompt_config = prompt_config

        if adapter_config is not None:
            self.adapter = Adapter(
                dim,
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
                    dim,
                    bottleneck_dim=adapter_config.dim,
                    dropout=adapter_config.dropout,
                    drop_path=adapter_config.drop_path,
                    act_layer=act_layer if adapter_config.act_layer else None,
                    norm_layer=norm_layer if adapter_config.norm_layer else None,
                    bias=adapter_config.bias,
                    scaling=adapter_config.scaling,
                    init=adapter_config.init,
                )

        if lora_config is not None and lora_config.config == "attention":
            self.attn = LoRAAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                act_layer=act_layer,
                norm_layer=norm_layer,
                lora_config=lora_config,
            )

        if prompt_config is not None:
            self.prompt = nn.Parameter(torch.zeros(prompt_config.num_tokens, dim))
            self.prompt_dropout = nn.Dropout(prompt_config.dropout)

            val = math.sqrt(6.0 / float(3 * patch_size**2 + prompt_config.num_tokens))
            nn.init.uniform_(self.prompt, -val, val)

    def forward_post(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        x = self.adapter(x, skip=x)
        return x

    def forward_pre(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = self.adapter(x, skip=x)
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

    def forward_pfeiffer(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        skip = self.drop_path2(self.ls2(self.mlp(self.norm2(y))))
        x = y + skip
        x = self.adapter(x, skip=skip)
        x = x + y
        return x

    def forward_intermediate(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        y = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x + self.adapter(y, skip=y)

    def forward_houlsby(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.adapter_attn(y, skip=y)
        y = self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x + self.adapter(y, skip=y)

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        z = y + self.drop_path2(self.ls2(self.mlp(self.norm2(y))))
        return self.adapter(y, skip=z)

    def forward_no_adapter(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prompt_config:
            x = self.include_prompt(x)
        if self.adapter_config is None:
            return self.forward_no_adapter(x)
        elif self.adapter_config.config == "pfeiffer":
            return self.forward_pfeiffer(x)
        elif self.adapter_config.config == "post":
            return self.forward_post(x)
        elif self.adapter_config.config == "pre":
            return self.forward_pre(x)
        elif self.adapter_config.config == "houlsby":
            return self.forward_houlsby(x)
        elif self.adapter_config.config == "intermediate":
            return self.forward_intermediate(x)
        elif self.adapter_config.config == "parallel":
            return self.forward_parallel(x)
        else:
            raise ValueError(f"Unknown adapter config: {self.adapter_config.config}")


class AdapterResPostBlock(ResPostBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        adapter_config=None,
        lora_config=None,
        prompt_config=None,
        patch_size=16,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )

        self.adapter_config = adapter_config
        self.lora_config = lora_config
        self.prompt_config = prompt_config

        if adapter_config is not None:
            self.adapter = Adapter(
                dim,
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
                    dim,
                    bottleneck_dim=adapter_config.dim,
                    dropout=adapter_config.dropout,
                    drop_path=adapter_config.drop_path,
                    act_layer=act_layer if adapter_config.act_layer else None,
                    norm_layer=norm_layer if adapter_config.norm_layer else None,
                    bias=adapter_config.bias,
                    scaling=adapter_config.scaling,
                    init=adapter_config.init,
                )

        if lora_config is not None and lora_config.config == "attention":
            self.attn = LoRAAttention(
                dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                act_layer=act_layer,
                norm_layer=norm_layer,
                lora_config=lora_config,
            )

        if prompt_config is not None:
            self.prompt = nn.Parameter(torch.zeros(prompt_config.num_tokens, dim))
            self.prompt_dropout = nn.Dropout(prompt_config.dropout)

            val = math.sqrt(6.0 / float(3 * patch_size**2 + prompt_config.num_tokens))
            nn.init.uniform_(self.prompt, -val, val)

    def forward_post(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        x = self.adapter(x, skip=x)
        return x

    def forward_pre(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = self.adapter(x, skip=x)
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x

    def forward_pfeiffer(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.drop_path1(self.norm1(self.attn(x)))
        skip = self.drop_path2(self.norm2(self.mlp(y)))
        x = y + skip
        x = self.adapter(x, skip=skip)
        x = x + y
        return x

    def forward_intermediate(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        y = self.drop_path2(self.norm2(self.mlp(x)))
        return x + self.adapter(y, skip=y)

    def forward_houlsby(self, x: torch.Tensor) -> torch.Tensor:
        y = self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.adapter_attn(y, skip=y)
        y = self.drop_path2(self.norm2(self.mlp(x)))
        return x + self.adapter(y, skip=y)

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.drop_path1(self.norm1(self.attn(x)))
        z = y + self.drop_path2(self.norm2(self.mlp(y)))
        return self.adapter(y, skip=z)

    def forward_no_adapter(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prompt_config:
            x = self.include_prompt(x)
        if self.adapter_config is None:
            return self.forward_no_adapter(x)
        elif self.adapter_config.config == "pfeiffer":
            return self.forward_pfeiffer(x)
        elif self.adapter_config.config == "post":
            return self.forward_post(x)
        elif self.adapter_config.config == "pre":
            return self.forward_pre(x)
        elif self.adapter_config.config == "houlsby":
            return self.forward_houlsby(x)
        elif self.adapter_config.config == "intermediate":
            return self.forward_intermediate(x)
        elif self.adapter_config.config == "parallel":
            return self.forward_parallel(x)
        else:
            raise ValueError(f"Unknown adapter config: {self.adapter_config.config}")


# modified from timm.models.vision_transformer.VisionTransformer
class VisionTransformerAdapter(VisionTransformer):
    """Vision Transformer with Adapter support

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values=None,
        class_token=True,
        pos_embed="learn",
        no_embed_class=False,
        reg_tokens=0,
        pre_norm=False,
        fc_norm=None,
        dynamic_img_size=False,
        dynamic_img_pad=False,
        drop_rate=0.0,
        pos_drop_rate=0.0,
        patch_drop_rate=0.0,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        fix_init=False,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        mlp_layer=Mlp,
        block_fn=AdapterBlock,
        adapter_config=None,
        lora_config=None,
        prompt_config=None,
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            class_token=class_token,
            pos_embed=pos_embed,
            no_embed_class=no_embed_class,
            reg_tokens=reg_tokens,
            pre_norm=pre_norm,
            fc_norm=fc_norm,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            drop_rate=drop_rate,
            pos_drop_rate=pos_drop_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            fix_init=fix_init,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn,
            mlp_layer=mlp_layer,
        )
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.prompt_config = prompt_config
        self.num_cls_token = 0
        if self.num_cls_token is not None:
            self.num_cls_token += 1
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                    adapter_config=adapter_config,
                    lora_config=lora_config,
                    prompt_config=prompt_config,
                    patch_size=patch_size,
                )
                for i in range(depth)
            ]
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if self.prompt_config:
            B, _, C = x.shape
            x = torch.cat(
                (
                    x[:, : self.num_cls_token, :],
                    torch.zeros(B, self.prompt_config.num_tokens, C).to(x.device),
                    x[:, self.num_cls_token :, :],
                ),
                dim=1,
            )

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x


# modified from timm.models.vision_transformer._create_vision_transformer
def _create_vision_transformer_adapter(
    variant: str, pretrained: bool = False, adapter=False, **kwargs
) -> VisionTransformer:
    out_indices = kwargs.pop("out_indices", 3)
    if "flexi" in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(
            checkpoint_filter_fn, interpolation="bilinear", antialias=False
        )
    else:
        _filter_fn = checkpoint_filter_fn

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    strict = True
    if "siglip" in variant and kwargs.get("global_pool", None) != "map":
        strict = False

    if adapter:
        # rewrite block_fn to use adapters
        block_fn = kwargs.pop("block_fn", Block)
        if block_fn == Block:
            block_fn = AdapterBlock
        elif block_fn == ResPostBlock:
            block_fn = AdapterResPostBlock
        else:
            raise ValueError(f"Adapters not implemented for {block_fn}!")
        return build_model_with_cfg(
            VisionTransformerAdapter,
            variant,
            pretrained,
            pretrained_filter_fn=_filter_fn,
            pretrained_strict=strict,
            feature_cfg=dict(out_indices=out_indices, feature_cls="getter"),
            block_fn=block_fn,
            **kwargs,
        )
    else:
        return build_model_with_cfg(
            VisionTransformer,
            variant,
            pretrained,
            pretrained_filter_fn=_filter_fn,
            pretrained_strict=strict,
            feature_cfg=dict(out_indices=out_indices, feature_cls="getter"),
            **kwargs,
        )
