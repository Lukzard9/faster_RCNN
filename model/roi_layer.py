import torch
import torch.nn as nn
from torchvision.ops import roi_align


class RoIAlignLayer(nn.Module):
    def __init__(self, output_size=(7, 7), spatial_scale=1.0 / 16):
        super(RoIAlignLayer, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, features, proposals):
        aligned_features = roi_align(
            input=features,
            boxes=proposals,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale
        )

        return aligned_features
