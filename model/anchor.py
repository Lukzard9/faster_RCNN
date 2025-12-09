import torch
import numpy as np

class AnchorGenerator:
    def __init__(self, stride=16, scales=[12, 24, 48, 96, 192], ratios=[0.5, 1, 2]):
        self.stride = stride
        self.scales = torch.tensor(scales)
        self.ratios = torch.tensor(ratios)
        self.base_anchors = self._generate_base_anchors()

    def _generate_base_anchors(self):
        base_anchors = []
        for scale in self.scales:
            for ratio in self.ratios:
                w = scale * torch.sqrt(ratio)
                h = scale / torch.sqrt(ratio)
                base_anchors.append([-w/2, -h/2, w/2, h/2])
        return torch.tensor(base_anchors, dtype=torch.float32)

    def generate(self, feature_map_size, device):
        grid_h, grid_w = feature_map_size
        
        ctrs_x = torch.arange(0, grid_w, dtype=torch.float32, device=device) * self.stride
        ctrs_y = torch.arange(0, grid_h, dtype=torch.float32, device=device) * self.stride
        
        grid_y, grid_x = torch.meshgrid(ctrs_y, ctrs_x, indexing='ij')
        grid_x = grid_x.reshape(-1)
        grid_y = grid_y.reshape(-1)
        shifts = torch.stack((grid_x, grid_y, grid_x, grid_y), dim=1) # stack for (x1, y1, x2, y2) format

        A = self.base_anchors.shape[0] 
        K = shifts.shape[0] # 40x30
        
        all_anchors = self.base_anchors.to(device).view(1, A, 4) + shifts.view(K, 1, 4) # (1, A, 4) + (K, 1, 4) -> (K, A, 4)
        
        all_anchors = all_anchors.view(-1, 4)
        
        return all_anchors