import torch
import torch.nn as nn
import torch.nn.functional as F


class FasterRCNNLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.rpn_batch_size = 256
        self.roi_batch_size = 128

    def forward(self, rpn_out, detector_out, anchors, proposals, gt_boxes, gt_labels):
        """
        rpn_out: (rpn_scores, rpn_deltas)
        detector_out: (cls_scores, bbox_deltas)
        anchors: Tensor (N_anchors, 4)
        proposals: List[Tensor] per image
        gt_boxes: List[Tensor] per image
        gt_labels: List[Tensor] per image
        """
        rpn_scores, rpn_deltas = rpn_out
        cls_scores, bbox_deltas = detector_out

        # RPN LOSS
        rpn_cls_loss = 0
        rpn_reg_loss = 0

        for i in range(len(gt_boxes)):
            target_labels, target_deltas = self._assign_rpn_targets(
                anchors, gt_boxes[i], rpn_deltas.device
            )

            mask = target_labels != -1
            rpn_cls_loss += F.binary_cross_entropy_with_logits(
                rpn_scores[i][mask].squeeze(),
                target_labels[mask].float()
            )

            pos_mask = target_labels == 1
            if pos_mask.sum() > 0:
                rpn_reg_loss += F.smooth_l1_loss(
                    rpn_deltas[i][pos_mask],
                    target_deltas[pos_mask]
                )

        # DETECTOR LOSS
        det_cls_loss = 0
        det_reg_loss = 0

        num_proposals_per_img = [p.shape[0] for p in proposals]
        cls_scores_split = cls_scores.split(num_proposals_per_img, dim=0)
        bbox_deltas_split = bbox_deltas.split(num_proposals_per_img, dim=0)

        for i in range(len(gt_boxes)):

            sampled_idxs, target_labels, target_deltas = self._assign_detector_targets(
                proposals[i], gt_boxes[i], gt_labels[i], bbox_deltas.device
            )

            curr_cls_scores = cls_scores_split[i][sampled_idxs]
            curr_bbox_deltas = bbox_deltas_split[i][sampled_idxs]

            det_cls_loss += F.cross_entropy(curr_cls_scores, target_labels)

            pos_mask = target_labels > 0
            if pos_mask.sum() > 0:
                pos_deltas_pred = curr_bbox_deltas[pos_mask]
                pos_deltas_target = target_deltas[pos_mask]
                pos_labels = target_labels[pos_mask]

                pos_deltas_pred = pos_deltas_pred.view(-1, 3, 4)  # (N_pos, Num_Classes, 4)

                final_preds = pos_deltas_pred[
                    torch.arange(pos_deltas_pred.size(0)), pos_labels]  # correct deltas for specific class

                det_reg_loss += F.smooth_l1_loss(final_preds, pos_deltas_target)

        return rpn_cls_loss + rpn_reg_loss + det_cls_loss + det_reg_loss

    def _assign_rpn_targets(self, anchors, gt_boxes, device):
        """Find anchors for GT boxes and sample them."""
        ious = self._box_iou(anchors, gt_boxes)
        max_iou, argmax_iou = ious.max(dim=1)  # best GT for each anchor

        labels = torch.full((len(anchors),), -1, dtype=torch.float32, device=device)

        labels[max_iou < 0.3] = 0
        labels[max_iou > 0.7] = 1

        _, gt_argmax_iou = ious.max(dim=0)
        labels[gt_argmax_iou] = 1  # at least one

        num_pos = int(self.rpn_batch_size * 0.5)  # 128
        pos_indices = torch.where(labels == 1)[0]
        if len(pos_indices) > num_pos:
            disable_inds = pos_indices[torch.randperm(len(pos_indices))[num_pos:]]
            labels[disable_inds] = -1

        num_neg = self.rpn_batch_size - (labels == 1).sum()
        neg_indices = torch.where(labels == 0)[0]
        if len(neg_indices) > num_neg:
            disable_inds = neg_indices[torch.randperm(len(neg_indices))[num_neg:]]
            labels[disable_inds] = -1

        matched_gt_boxes = gt_boxes[argmax_iou]
        deltas = self._encode_boxes(anchors, matched_gt_boxes)

        return labels, deltas

    def _assign_detector_targets(self, proposals, gt_boxes, gt_labels, device):
        """Find anchors for GT boxes and sample them."""
        ious = self._box_iou(proposals, gt_boxes)
        max_iou, argmax_iou = ious.max(dim=1)

        assigned_labels = torch.zeros(len(proposals), dtype=torch.long, device=device)

        pos_indices = torch.where(max_iou >= 0.5)[0]
        assigned_labels[pos_indices] = gt_labels[argmax_iou[pos_indices]]

        num_pos = int(self.roi_batch_size * 0.25)
        pos_indices = torch.where(assigned_labels > 0)[0]
        if len(pos_indices) > num_pos:
            keep_pos = pos_indices[torch.randperm(len(pos_indices))[:num_pos]]
        else:
            keep_pos = pos_indices

        neg_indices = torch.where(assigned_labels == 0)[0]
        actual_num_neg = self.roi_batch_size - len(keep_pos)
        if len(neg_indices) > actual_num_neg:
            keep_neg = neg_indices[torch.randperm(len(neg_indices))[:actual_num_neg]]
        else:
            keep_neg = neg_indices

        sampled_idxs = torch.cat((keep_pos, keep_neg))

        matched_gt_boxes = gt_boxes[argmax_iou[sampled_idxs]]
        sampled_proposals = proposals[sampled_idxs]

        target_deltas = self._encode_boxes(sampled_proposals, matched_gt_boxes)
        target_labels = assigned_labels[sampled_idxs]

        return sampled_idxs, target_labels, target_deltas

    def _encode_boxes(self, anchors, targets):
        w_a = anchors[:, 2] - anchors[:, 0]
        h_a = anchors[:, 3] - anchors[:, 1]
        x_a = anchors[:, 0] + 0.5 * w_a
        y_a = anchors[:, 1] + 0.5 * h_a

        w_t = targets[:, 2] - targets[:, 0]
        h_t = targets[:, 3] - targets[:, 1]
        x_t = targets[:, 0] + 0.5 * w_t
        y_t = targets[:, 1] + 0.5 * h_t

        dx = (x_t - x_a) / w_a
        dy = (y_t - y_a) / h_a
        dw = torch.log(w_t / w_a + 1e-6)
        dh = torch.log(h_t / h_a + 1e-6)

        return torch.stack((dx, dy, dw, dh), dim=1)

    def _box_iou(self, box_a, box_b):
        lt = torch.max(box_a[:, None, :2], box_b[None, :, :2])
        rb = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])

        return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-6)
