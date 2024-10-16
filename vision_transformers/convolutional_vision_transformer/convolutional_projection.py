from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


# implementation of Convolutional Projection
def _build_projection(self, dim_in, dim_out, kernel_size, padding, stride, method):
    """
    1- If the method is dw_bn (depthwise separable with batch normalization), it creates a Sequential
       block consisting of a depthwise separable convolutional layer followed by batch normalization
       and rearranges the dimensions.

    2- If the method is avg (average pooling), it creates a Sequential block with an average pooling
       layer followed by rearranging the dimensions.

    3- If the method is linear, it returns None, indicating that no projection is applied.
    
    * The rearrangement of dimensions is performed using the Rearrange operation, which reshapes 
      the input tensor. The resulting projection block is then returned.
    """


    if method == "dw_bn":
        proj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        nn.Conv2d(
                            dim_in,
                            dim_in,
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride,
                            bias=False,
                            groups=dim_in,
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(dim_in)),
                    ("rearrage", Rearrange("b c h w -> b (h w) c")),
                ]
            )
        )
    elif method == "avg":
        proj = nn.Sequential(
            OrderedDict(
                [
                    (
                        "avg",
                        nn.AvgPool2d(
                            kernel_size=kernel_size,
                            padding=padding,
                            stride=stride,
                            ceil_mode=True,
                        ),
                    ),
                    ("rearrage", Rearrange("b c h w -> b (h w) c")),
                ]
            )
        )
    elif method == "linear":
        proj = None
    else:
        raise ValueError("Unknown method ({})".format(method))
    

    return proj