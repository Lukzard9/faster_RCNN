import torch
import torch.nn as nn
from torchvision.ops import roi_align

class RoIAlignLayer(nn.Module):
    def __init__(self, output_size=(7, 7), spatial_scale=1.0/16):
        """
        Args:
            output_size (tuple): The fixed resolution (H, W) we want for every object.
            spatial_scale (float): Scaling factor to map input coordinates to feature coordinates.
                                   Since our Backbone stride is 32, scale is 1/32.
        """
        super(RoIAlignLayer, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, features, proposals):
        """
        Args:
            features (Tensor): Feature map from Backbone. Shape (N, 2048, H_feat, W_feat).
            proposals (List[Tensor]): List of N tensors. Each tensor is (M, 4) containing 
                                      box coordinates [x1, y1, x2, y2] in ORIGINAL IMAGE PIXELS.
        
        Returns:
            aligned_features (Tensor): Shape (Total_Proposals, 2048, 7, 7).
        """
        
        # torchvision.ops.roi_align expects the proposals to be in a specific format:
        # A list of tensors, or a single tensor with an added batch index column.
        # Passing a list of tensors (one per image) is the easiest way.
        
        aligned_features = roi_align(
            input=features,
            boxes=proposals,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=-1  # -1 means "adaptive", usually samples 2 points per bin
        )
        
        return aligned_features