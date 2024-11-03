import torch
from torch import nn
from torchvision.models import resnet50

# DETR class defines a transformer-based model for object detection.
class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers):
        super().__init__()
        # Feature extractor: ResNet-50 backbone without the last two layers.
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        # Convolution to match hidden_dim size.
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        # Transformer module for encoding and decoding spatial features.
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        # Linear layer for predicting class probabilities.
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        # Linear layer for predicting bounding box coordinates.
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        # Learnable positional embeddings for queries.
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        # Learnable embeddings for row and column positions.
        self.row_embed = nn.Parameter(torch.rand(100, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # Pass input through backbone to extract feature map.
        x = self.backbone(inputs)
        # Apply 1x1 convolution to reduce feature dimensions.
        h = self.conv(x)
        # Get feature map dimensions.
        H, W = h.shape[-2:]
        # Combine row and column embeddings for positional encoding.
        pos = (
            torch.cat(
                [self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                 self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)],
                dim=-1
            ).flatten(0, 1).unsqueeze(1)
        )
        # Transformer encoding and decoding with positional encoding and query embedding.
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1))
        # Return class predictions and normalized bounding box predictions.
        return self.linear_class(h), self.linear_bbox(h).sigmoid()
