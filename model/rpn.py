import torch
import torch.nn as nn
from torchvision.ops import nms
from model.anchor import AnchorGenerator


class RPN(nn.Module):
    def __init__(self, in_channels=2048, mid_channels=512, n_anchors=15):
        super(RPN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Objectness
        self.cls_layer = nn.Conv2d(mid_channels, n_anchors, kernel_size=1, stride=1)

        # BBox Offsets
        self.reg_layer = nn.Conv2d(mid_channels, n_anchors * 4, kernel_size=1, stride=1)

        self.anchor_generator = AnchorGenerator()

    def _apply_deltas(self, anchors, deltas):

        # Anchors: [x1, y1, x2, y2]
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        center_x = anchors[:, 0] + 0.5 * widths
        center_y = anchors[:, 1] + 0.5 * heights

        # deltas for anchor0: dx0 dy0 dw0 dh0
        #            anchor1: dx1 dy1 dw1 dh1
        #                     ...
        dx = deltas[:, 0::4]
        dy = deltas[:, 1::4]
        dw = deltas[:, 2::4]
        dh = deltas[:, 3::4]

        pred_center_x = dx * widths.unsqueeze(1) + center_x.unsqueeze(1)
        pred_center_y = dy * heights.unsqueeze(1) + center_y.unsqueeze(1)
        pred_w = torch.exp(dw) * widths.unsqueeze(1)
        pred_h = torch.exp(dh) * heights.unsqueeze(1)

        # [x,y,w,h] -> [x1,y1,x2,y2]
        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_center_x - 0.5 * pred_w
        pred_boxes[:, 1::4] = pred_center_y - 0.5 * pred_h
        pred_boxes[:, 2::4] = pred_center_x + 0.5 * pred_w
        pred_boxes[:, 3::4] = pred_center_y + 0.5 * pred_h

        return pred_boxes

    def forward(self, features, image_shape):
        batch_size = features.shape[0]

        x = self.relu(self.conv1(features))

        rpn_scores = self.cls_layer(x).permute(0, 2, 3, 1).contiguous()  # (B, 15, H, W) -> (B, H, W, 15)
        rpn_deltas = self.reg_layer(x).permute(0, 2, 3, 1).contiguous()  # (B, 60, H, W) -> (B, H, W, 60)

        rpn_scores = rpn_scores.view(batch_size, -1, 1)  # (B, H, W, 15) -> (B, (H*W*15), 1)
        rpn_deltas = rpn_deltas.view(batch_size, -1, 4)  # (B, H, W, 60) -> (B, (H*W*15), 4)

        feat_h, feat_w = features.shape[2], features.shape[3]

        anchors = self.anchor_generator.generate((feat_h, feat_w), features.device)

        proposals_list = []

        for i in range(batch_size):
            rp_boxes = self._apply_deltas(anchors, rpn_deltas[i].detach())

            rp_boxes[:, 0::2].clamp_(min=0, max=image_shape[1])
            rp_boxes[:, 1::2].clamp_(min=0, max=image_shape[0])

            scores = torch.sigmoid(rpn_scores[i]).squeeze()

            k = min(2000, len(scores))
            top_k_scores, top_k_indices = torch.topk(scores, k)
            top_k_boxes = rp_boxes[top_k_indices]

            keep_indices = nms(top_k_boxes, top_k_scores, iou_threshold=0.6)

            keep_indices = keep_indices[:1000]
            final_proposals = top_k_boxes[keep_indices]

            proposals_list.append(final_proposals)

        return rpn_scores, rpn_deltas, proposals_list
