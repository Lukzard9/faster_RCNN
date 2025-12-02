import torch
import torch.nn as nn
from backbone import ResNetBackbone
from rpn import RPN
from roi_layer import RoIAlignLayer
from detector import DetectorHead

class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        """
        Args:
            num_classes: Number of classes (excluding background). 
                         If you have Car, Person -> num_classes = 2.
        """
        super(FasterRCNN, self).__init__()
        
        # 1. Backbone (ResNet50)
        self.backbone = ResNetBackbone(pretrained=True, freeze_params=True)
        
        # 2. RPN (Region Proposal Network)
        # Input: 2048 channels from ResNet
        self.rpn = RPN(in_channels=2048, mid_channels=512)
        
        # 3. RoI Align
        # Maps proposals to fixed 7x7 features
        self.roi_align = RoIAlignLayer(output_size=(7, 7), spatial_scale=1.0/32)
        
        # 4. Detector Head (RoI Head)
        # Input: 2048 channels, Output: num_classes
        self.detector = DetectorHead(num_classes=num_classes, in_channels=2048)

    def forward(self, images, targets=None):
        """
        images: (Batch, 3, H, W) Tensor
        targets: List of Dicts (optional, only for training debug)
        """
        # --- Stage 1: Feature Extraction ---
        # Shape: (Batch, 2048, H/32, W/32)
        features = self.backbone(images)
        
        # --- Stage 2: RPN ---
        # rpn_scores: (Batch, N_anchors, 1)
        # rpn_deltas: (Batch, N_anchors, 4)
        # proposals: List[Tensor] (one tensor per image in batch)
        rpn_scores, rpn_deltas, proposals = self.rpn(features, images.shape[-2:])
        
        # IMPORTANT: For the Loss function, we need the raw anchors.
        # We ask the RPN's generator to give us the grid for this feature map size.
        feat_size = (features.shape[2], features.shape[3])
        anchors = self.rpn.anchor_generator.generate(feat_size, images.device)

        # --- Stage 3: RoI Align ---
        # Extract features for every proposal
        # Shape: (Total_Proposals_in_Batch, 2048, 7, 7)
        roi_features = self.roi_align(features, proposals)
        
        # --- Stage 4: Detector Head ---
        # cls_scores: (Total_Proposals, Num_Classes + 1)
        # bbox_deltas: (Total_Proposals, (Num_Classes + 1) * 4)
        cls_scores, bbox_deltas = self.detector(roi_features)
        
        # Return everything needed for Loss or Post-Processing
        return {
            "rpn_scores": rpn_scores,
            "rpn_deltas": rpn_deltas,
            "anchors": anchors,
            "proposals": proposals,
            "cls_scores": cls_scores,
            "bbox_deltas": bbox_deltas
        }