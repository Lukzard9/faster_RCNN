import torch
import numpy as np

class AnchorGenerator:
    def __init__(self, stride=32, scales=[128, 256, 512], ratios=[0.5, 1, 2]):
        self.stride = stride
        self.scales = torch.tensor(scales)
        self.ratios = torch.tensor(ratios)
        self.base_anchors = self._generate_base_anchors()

    def _generate_base_anchors(self):
        """
        Generates 9 anchors (3 scales x 3 ratios) centered at (0,0).
        """
        base_anchors = []
        for scale in self.scales:
            for ratio in self.ratios:
                w = scale * torch.sqrt(ratio)
                h = scale / torch.sqrt(ratio)
                # Create box: [-w/2, -h/2, w/2, h/2]
                base_anchors.append([-w/2, -h/2, w/2, h/2])
        return torch.tensor(base_anchors, dtype=torch.float32)

    def generate(self, feature_map_size, device):
        """
        Shifts the base anchors over the entire feature map grid.
        Input: feature_map_size = (height, width) of the features (e.g., 15, 20)
        """
        grid_h, grid_w = feature_map_size
        
        # 1. Generate Grid Coordinates
        shifts_x = torch.arange(0, grid_w, dtype=torch.float32, device=device) * self.stride
        shifts_y = torch.arange(0, grid_h, dtype=torch.float32, device=device) * self.stride
        
        # Create meshgrid
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        
        # Flatten to (K, 4) where K is total pixels in feature map
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)

        # 2. Add Shifts to Base Anchors
        # (A = 9 anchors, K = grid pixels)
        A = self.base_anchors.shape[0]
        K = shifts.shape[0]
        
        # Reshape to broadcast: (1, A, 4) + (K, 1, 4) -> (K, A, 4)
        all_anchors = self.base_anchors.to(device).view(1, A, 4) + shifts.view(K, 1, 4)
        
        # Flatten to (N, 4) where N = K * A
        all_anchors = all_anchors.view(-1, 4)
        
        return all_anchors