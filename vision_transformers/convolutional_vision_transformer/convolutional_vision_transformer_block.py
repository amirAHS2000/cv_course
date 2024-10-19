from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from convolutional_token_embedding import ConvEmbed
from timm.layers import trunc_normal_
from vision_transformer_block import VisionTransformer


class ConvolutionalVisionTransformer(nn.Module):
    def __init__(
      self,
      in_chans=3,
      num_classes=1000,
      act_layer=nn.GELU,
      norm_layer=nn.LayerNorm,
      init="trunc_norm",
      spec=None,      
    ):
        super().__init__()
        self.num_classes = num_classes

        self.num_stages = spec["NUM_STAGES"]
        for i in range(self.num_stages):
            """
            The model consists of multiple stages, each represented by an instance
            of the VisionTransformer class.
            Each stage has different configurations such as patch size, stride,
            depth, number of heads, etc., specified in the spec dictionary.
            """
            kwargs = {
                "patch_size": spec["PATCH_SIZE"][i],
                "patch_stride": spec["PATCH_STRIDE"][i],
                "patch_padding": spec["PATCH_PADDING"][i],
                "embed_dim": spec["DIM_EMBED"][i],
                "depth": spec["DEPTH"][i],
                "num_heads": spec["NUM_HEADS"][i],
                "mlp_ratio": spec["MLP_RATIO"][i],
                "qkv_bias": spec["QKV_BIAS"][i],
                "drop_rate": spec["DROP_RATE"][i],
                "attn_drop_rate": spec["ATTN_DROP_RATE"][i],
                "drop_path_rate": spec["DROP_PATH_RATE"][i],
                "with_cls_token": spec["CLS_TOKEN"][i],
                "method": spec["QKV_PROJ_METHOD"][i],
                "kernel_size": spec["KERNEL_QKV"][i],
                "padding_q": spec["PADDING_Q"][i],
                "padding_kv": spec["PADDING_KV"][i],
                "stride_kv": spec["STRIDE_KV"][i],
                "stride_q": spec["STRIDE_Q"][i],
            }
            
            """
            The vision transformer stages are sequentially named as stage0,
            stage1, etc., and each stage is an instance of the VisionTransformer
            class forming a hierarchy of transformers.
            """
            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs,
            )
            setattr(self, f"stage{i}", stage)

            in_chans = spec["DIM_EMBED"][i]

        dim_embed = spec["DIM_EMBED"][-1]
        self.norm = norm_layer(dim_embed)
        self.cls_token = spec["CLS_TOKEN"][-1]

        """
        The class has a classifier head that performs a linear transformation
        to produce the final output.
        """
        self.head = (
            nn.Linear(dim_embed, num_classes) if num_classes > 0 else nn.Identity()
        )
        trunc_normal_(self.head.weight, std=0.02)

    """
    The forward_features method processes the input x through all the stages,
    and it aggregates the final representation.
    """
    def forward_features(self, x):
        for i in range(self.num_stages):
            x, cls_tokens = getattr(self, f"stage{i}")(x)

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = torch.squeeze(x)
        else:
            x = rearrange(x, "b c h w -> b (h w) c")
            x = self.norm(x)
            x = torch.mean(x, dim=1)

        return x
    
    """
    The forward method calls forward_features and then passes the result through
    the classifier head.
    """
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x