import torch.nn as nn
from model.backbone import ResNetBackbone
from model.rpn import RPN
from model.roi_layer import RoIAlignLayer
from model.detector import DetectorHead

class FasterRCNN(nn.Module):
    def __init__(self, num_classes):
        super(FasterRCNN, self).__init__()

        self.backbone = ResNetBackbone(pretrained=True, freeze_params=False)
        self.rpn = RPN(in_channels=2048, mid_channels=512, n_anchors=15)
        self.roi_align = RoIAlignLayer(output_size=(7, 7), spatial_scale=1.0/16)        
        self.detector = DetectorHead(num_classes=num_classes, in_channels=2048)

    def forward(self, images):
        features = self.backbone(images)
        
        rpn_scores, rpn_deltas, proposals = self.rpn(features, images.shape[-2:])
        
        feat_size = (features.shape[2], features.shape[3])
        anchors = self.rpn.anchor_generator.generate(feat_size, images.device) # for loss

        roi_features = self.roi_align(features, proposals)
        
        cls_scores, bbox_deltas = self.detector(roi_features)
        
        return {
            "rpn_scores": rpn_scores,
            "rpn_deltas": rpn_deltas,
            "anchors": anchors,
            "proposals": proposals,
            "cls_scores": cls_scores,
            "bbox_deltas": bbox_deltas
        }