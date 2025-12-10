import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

class DetectorHead(nn.Module):
    def __init__(self, num_classes, in_channels=2048, roi_size=7, hidden_dim=1024):
        super(DetectorHead, self).__init__()
        self.input_dim = in_channels * roi_size * roi_size
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.cls_score = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_pred = nn.Linear(hidden_dim, (num_classes + 1) * 4)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

    def post_process(self, scores, bbox_deltas, proposals, img_shape, score_thresh=0.5, iou_threshold=0.5):
        probs = F.softmax(scores, dim=1)
        pred_boxes = self._apply_deltas_to_proposals(proposals, bbox_deltas)
        
        pred_boxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        pred_boxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        final_results = []
        num_classes = scores.shape[1]
        
        for class_id in range(1, num_classes): # skip 0
            cls_probs = probs[:, class_id]
            box_idx = class_id * 4
            cls_boxes = pred_boxes[:, box_idx : box_idx + 4]
            
            mask = cls_probs > score_thresh
            cls_boxes = cls_boxes[mask]
            cls_probs = cls_probs[mask]
            
            if len(cls_boxes) == 0:
                continue
            
            keep = nms(cls_boxes, cls_probs, iou_threshold)
            
            valid_boxes = cls_boxes[keep]
            valid_scores = cls_probs[keep]
            
            for box, score in zip(valid_boxes, valid_scores):
                final_results.append((class_id - 1, box, score))
                
        return final_results

    def _apply_deltas_to_proposals(self, proposals, deltas):
        widths  = proposals[:, 2] - proposals[:, 0]
        heights = proposals[:, 3] - proposals[:, 1]
        ctr_x   = proposals[:, 0] + 0.5 * widths
        ctr_y   = proposals[:, 1] + 0.5 * heights
        
        pred_boxes = torch.zeros_like(deltas)
        
        for i in range(deltas.shape[1] // 4):
            dx = deltas[:, i*4]
            dy = deltas[:, i*4+1]
            dw = deltas[:, i*4+2]
            dh = deltas[:, i*4+3]
            
            pred_ctr_x = dx * widths + ctr_x
            pred_ctr_y = dy * heights + ctr_y
            pred_w     = torch.exp(dw) * widths
            pred_h     = torch.exp(dh) * heights
            
            pred_boxes[:, i*4]   = pred_ctr_x - 0.5 * pred_w
            pred_boxes[:, i*4+1] = pred_ctr_y - 0.5 * pred_h
            pred_boxes[:, i*4+2] = pred_ctr_x + 0.5 * pred_w
            pred_boxes[:, i*4+3] = pred_ctr_y + 0.5 * pred_h
            
        return pred_boxes