import torch
import torch.nn as nn
from torchvision.ops import nms # Standard Allowed Op
from model.anchor import AnchorGenerator

class RPN(nn.Module):
    def __init__(self, in_channels=2048, mid_channels=512, n_anchors=9):
        super(RPN, self).__init__()
        
        # 1. Sliding Window (Intermediate layer)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # 2. Classification Layer (Objectness)
        # Output: 1 score per anchor (Probability of being foreground)
        self.cls_layer = nn.Conv2d(mid_channels, n_anchors, kernel_size=1, stride=1)
        
        # 3. Regression Layer (Bounding Box Offsets)
        # Output: 4 coords (dx, dy, dw, dh) per anchor
        self.reg_layer = nn.Conv2d(mid_channels, n_anchors * 4, kernel_size=1, stride=1)

        # Utils
        self.anchor_generator = None # Will be set in main class or passed
        
    def _apply_deltas(self, anchors, deltas):
        """
        Applies the learned offsets (deltas) to the anchors.
        This uses the standard Faster R-CNN equations.
        """
        widths  = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        ctr_x   = anchors[:, 0] + 0.5 * widths
        ctr_y   = anchors[:, 1] + 0.5 * heights

        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]

        # Predict center
        pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
        pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
        # Predict w, h (exp ensures positive size)
        pred_w     = torch.exp(dw) * widths.unsqueeze(1)
        pred_h     = torch.exp(dh) * heights.unsqueeze(1)

        # Convert back to [x1, y1, x2, y2]
        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes

    def forward(self, features, image_shape):
        """
        features: (Batch, 2048, H, W)
        image_shape: (H_img, W_img) e.g., (480, 640)
        """
        batch_size = features.shape[0]
        
        # 1. Pass through layers
        x = self.relu(self.conv1(features))
        
        # Pred Scores: (Batch, 9, H, W) -> Permute to (Batch, H, W, 9)
        rpn_scores = self.cls_layer(x).permute(0, 2, 3, 1).contiguous()
        # Pred Deltas: (Batch, 36, H, W) -> Permute to (Batch, H, W, 36)
        rpn_deltas = self.reg_layer(x).permute(0, 2, 3, 1).contiguous()

        # Reshape for easier processing
        # Scores: (Batch, N_anchors, 1)
        rpn_scores = rpn_scores.view(batch_size, -1, 1) 
        # Deltas: (Batch, N_anchors, 4)
        rpn_deltas = rpn_deltas.view(batch_size, -1, 4)
        
        # 2. Generate Anchors (On the fly based on feature map size)
        feat_h, feat_w = features.shape[2], features.shape[3]
        if self.anchor_generator is None:
             # Initialize with default stride=32
            self.anchor_generator = AnchorGenerator() 
            
        anchors = self.anchor_generator.generate((feat_h, feat_w), features.device)
        
        # 3. Process Proposals (Only needed for the Next Stage)
        # We perform this "per image" in the batch
        proposals_list = []
        
        for i in range(batch_size):
            # Apply deltas to anchors
            # Note: We detach() here because we don't backprop through proposal coordinates
            # We only backprop through the rpn_deltas loss
            decoded_boxes = self._apply_deltas(anchors, rpn_deltas[i].detach())
            
            # Clip boxes to image dimensions
            decoded_boxes[:, 0::2].clamp_(min=0, max=image_shape[1]) # x
            decoded_boxes[:, 1::2].clamp_(min=0, max=image_shape[0]) # y
            
            # Get scores (sigmoid for probability)
            scores = torch.sigmoid(rpn_scores[i]).squeeze()
            
            # --- FILTERING (Important for Exam) ---
            # 1. Keep top K scores (e.g., 2000 pre-NMS)
            k = min(2000, len(scores))
            top_k_scores, top_k_indices = torch.topk(scores, k)
            top_k_boxes = decoded_boxes[top_k_indices]
            
            # 2. NMS (Non-Maximum Suppression)
            # Remove overlapping boxes (IoU > 0.7)
            keep_indices = nms(top_k_boxes, top_k_scores, iou_threshold=0.7)
            
            # 3. Keep top N after NMS (e.g., 1000 post-NMS)
            keep_indices = keep_indices[:1000]
            final_proposals = top_k_boxes[keep_indices]
            
            proposals_list.append(final_proposals)
            
        return rpn_scores, rpn_deltas, proposals_list