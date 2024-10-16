from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from timm.layers import to_2tuple


"""
this module is designed for patch-wise embedding of images, where each patch is processed independently 
through a convolutional layer, and optional normalization is applied to the embedded features.
"""
class ConvEmbed(nn.Module):
    """
    The __init__ method initializes the module with parameters such as patch_size (size of the image patches), 
    in_chans (number of input channels), embed_dim (dimensionality of the embedded patches), stride (stride for 
    the convolution operation), padding (padding for the convolution operation), and norm_layer (a normalization 
    layer, which is optional).

    In the constructor, a 2D convolutional layer (nn.Conv2d) is created with specified parameters, including
    the patch size, input channels, embedding dimension, stride, and padding. This convolutional layer is 
    assigned to self.proj.

    If a normalization layer is provided, an instance of the normalization layer is created with embed_dim 
    channels, and it is assigned to self.norm.

    The forward method takes an input tensor x and applies the convolution operation using self.proj. The 
    output is reshaped using the rearrange function to flatten the spatial dimensions. If a normalization 
    layer is present, it is applied to the flattened representation. Finally, the tensor is reshaped back 
    to the original spatial dimensions and returned.
    """
    def __init__(self, patch_size=7, in_chans=3, embed_dim=64, stride=4, padding=2, 
                 norm_layer=None):
        super().__init__()

        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x
