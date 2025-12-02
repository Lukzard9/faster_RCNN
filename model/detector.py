import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

class DetectorHead(nn.Module):
    def __init__(self, num_classes, in_channels=2048, roi_size=7, hidden_dim=1024):
        """
        Args:
            num_classes (int): Number of real classes (excluding background).
            in_channels (int): Depth of feature maps (2048 for ResNet50).
            roi_size (int): Height/Width of RoI Align output (7).
            hidden_dim (int): Size of the dense layers.
        """
        super(DetectorHead, self).__init__()
        
        # 1. Flatten the features
        # Input: (N, 2048, 7, 7) -> Flatten -> (N, 2048*7*7)
        self.input_dim = in_channels * roi_size * roi_size
        
        # 2. Two Fully Connected Layers (The "MLP Head")
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)

        # 3. The Output Layers
        # Class Scores: +1 is for the Background Class (always index 0)
        self.cls_score = nn.Linear(hidden_dim, num_classes + 1)
        
        # Box Deltas: 4 coordinates * (num_classes + 1)
        # We predict a separate box adjustment for EVERY class (Class-Specific Regression)
        self.bbox_pred = nn.Linear(hidden_dim, (num_classes + 1) * 4)

    def forward(self, x):
        # x shape: (N_proposals, 2048, 7, 7)
        x = x.view(x.size(0), -1) # Flatten
        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        scores = self.cls_score(x)      # Shape: (N, num_classes + 1)
        bbox_deltas = self.bbox_pred(x) # Shape: (N, (num_classes+1) * 4)
        
        return scores, bbox_deltas

    def post_process(self, scores, bbox_deltas, proposals, img_shape):
        """
        Converts raw network outputs into final bounding boxes for mIoU calculation.
        This effectively reverses the RPN/Dataset logic.
        """
        # 1. Apply Softmax to get probabilities
        probs = F.softmax(scores, dim=1)
        
        # 2. Decode Boxes (Apply Deltas to Proposals)
        # We reuse the same logic as RPN, but now for specific classes
        pred_boxes = self._apply_deltas_to_proposals(proposals, bbox_deltas)
        
        # 3. Clip boxes to image
        pred_boxes[:, 0::2].clamp_(min=0, max=img_shape[1]) # x
        pred_boxes[:, 1::2].clamp_(min=0, max=img_shape[0]) # y

        final_results = []
        
        # Iterate over every class (skip 0 because it's background)
        num_classes = scores.shape[1]
        for class_id in range(1, num_classes):
            
            # Get probabilities for this class
            cls_probs = probs[:, class_id]
            
            # Get boxes for this class (extract the 4 cols relevant to this class)
            # stored as [bg_x, bg_y, ..., class1_x, class1_y, ...]
            box_idx = class_id * 4
            cls_boxes = pred_boxes[:, box_idx : box_idx + 4]
            
            # 4. Filter by Confidence Threshold (Essential for good mIoU)
            # If the model isn't at least 5% sure, ignore it.
            mask = cls_probs > 0.05
            cls_boxes = cls_boxes[mask]
            cls_probs = cls_probs[mask]
            
            if len(cls_boxes) == 0:
                continue
                
            # 5. Apply NMS (Non-Maximum Suppression) PER CLASS
            # If we predict 5 boxes for the same Car, keep only the best one.
            keep = nms(cls_boxes, cls_probs, iou_threshold=0.5)
            
            valid_boxes = cls_boxes[keep]
            valid_scores = cls_probs[keep]
            
            for box, score in zip(valid_boxes, valid_scores):
                # Append result: [Class_ID, x1, y1, x2, y2, Score]
                # Note: We subtract 1 from class_id to match your Original YOLO labels 
                # (which were 0-indexed)
                final_results.append((class_id - 1, box, score))
                
        return final_results

    def _apply_deltas_to_proposals(self, proposals, deltas):
        """
        Manual implementation of box decoding.
        proposals: (N, 4) -> [x1, y1, x2, y2]
        deltas: (N, 4 * n_classes) -> [dx, dy, dw, dh, ...]
        """
        # Convert proposals to center/size
        widths  = proposals[:, 2] - proposals[:, 0]
        heights = proposals[:, 3] - proposals[:, 1]
        ctr_x   = proposals[:, 0] + 0.5 * widths
        ctr_y   = proposals[:, 1] + 0.5 * heights
        
        # We need to reshape deltas to handle the broadcasting easily
        # or just repeat the proposals to match delta shape.
        # For simplicity in exam code, let's loop or expand:
        
        pred_boxes = torch.zeros_like(deltas)
        
        # The deltas are arranged as [bg, bg, bg, bg, c1, c1, c1, c1, c2...]
        for i in range(deltas.shape[1] // 4):
            dx = deltas[:, i*4]
            dy = deltas[:, i*4+1]
            dw = deltas[:, i*4+2]
            dh = deltas[:, i*4+3]
            
            pred_ctr_x = dx * widths + ctr_x
            pred_ctr_y = dy * heights + ctr_y
            pred_w     = torch.exp(dw) * widths
            pred_h     = torch.exp(dh) * heights
            
            pred_boxes[:, i*4]   = pred_ctr_x - 0.5 * pred_w # x1
            pred_boxes[:, i*4+1] = pred_ctr_y - 0.5 * pred_h # y1
            pred_boxes[:, i*4+2] = pred_ctr_x + 0.5 * pred_w # x2
            pred_boxes[:, i*4+3] = pred_ctr_y + 0.5 * pred_h # y2
            
        return pred_boxes