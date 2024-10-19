from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from convolutional_token_embedding import ConvEmbed
from timm.layers import trunc_normal_


class VisionTransformer(nn.Module):
    """
    Vision Transformer with support for patch or hybrid CNN input stage.
    """

    def __init__(
            self,
            patch_size=16,
            patch_stride=16,
            patch_padding=0,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=False,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            init="trunc_norm",
            **kwargs,
    ):
        """
        Initialization: The VisionTransformer class is initialized with various parameters that
        define the model architecture, such as patch size, embedding dimensions,
        number of layers, number of attention heads, dropout rates, etc.
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.rearrage = None


        """
        Patch Embedding: The model includes a patch embedding layer (`patch_embed),
        which processes the input image by dividing it into non-overlapping patches
        and embedding them using Convolutions.
        """
        self.patch_embed = ConvEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )

        """
        Classification Token: Optionally, the model can include a learnable
        classification token (cls_token) appended to the input sequence. This
        token is used for classification tasks.
        """
        with_cls_token = kwargs["with_cls_token"]
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None


        """
        Stochastic Depth: Stochastic depth is applied to the transformer blocks,
        where a random subset of blocks is skipped during training to improve
        regularization. This is controlled by the drop_path_rate parameter.
        """
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ] # stochastic depth decay rule


        """
        Transformer Blocks: The model consists of a stack of transformer blocks
        (Block). The number of blocks is determined by the depth parameter. Each
        block contains multi-head self-attention mechanisms and a feedforward
        neural network.
        """
        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs,
                )
            )
            self.blocks = nn.ModuleList(blocks)

            if self.cls_token is not None:
                trunc_normal_(self.cls_token, std=0.02)

            """
            Initialization of Weights: The model weights are initialized using
            either truncated normal distribution (trunc_norm) or Xavier initialization
            (xavier).
            """
            if init == "xavier":
                self.apply(self._init_weights_xavier)
            else:
                self.apply(self._init_weights_trunc_normal)

    """
    Forward Method: The forward method processes the input through the patch
    embedding, rearranges the dimensions, adds the classification token if
    present, applies dropout, and then passes the data through the stack of
    transformer blocks. Finally, the output is rearranged back to the original
    shape, and the classification token (if present) is separated from the rest
    of the sequence before returning the output.
    """
    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.size()

        x = rearrange(x, "b c h w -> b (h w) c")

        cls_tokens = None
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)

        if self.cls_token is not None:
            cls_tokens, x = torch.split(x, [1, H * W], 1)
        
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x, cls_tokens
    