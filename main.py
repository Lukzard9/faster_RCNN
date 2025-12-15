import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from pathlib import Path
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F_vis
import numpy as np

from dataset import RecursiveDetDataset, custom_collate_fn
from model.model import FasterRCNN
from model.loss import FasterRCNNLoss
import metrics

CLASS_MAP = {
    1: "Strawberry",
    2: "Olive"
}


def save_img(image_tensor, preds, targets, epoch, save_dir, device, img_name_suffix=""):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    img = image_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    img = (img * 255).type(torch.uint8)

    gt_boxes = targets['boxes']
    if gt_boxes.numel() > 0:
        img = draw_bounding_boxes(img, gt_boxes, colors="green", width=3)

    pred_boxes = [p[1] for p in preds]
    pred_labels = []
    for p in preds:
        cls_id = int(p[0]) + 1
        cls_name = CLASS_MAP.get(cls_id, str(cls_id))
        score = p[2]
        pred_labels.append(f"{cls_name}: {score:.2f}")

    if len(pred_boxes) > 0:
        pred_boxes_tensor = torch.stack(pred_boxes)
        img = draw_bounding_boxes(img, pred_boxes_tensor, labels=pred_labels, colors="red", width=2)

    vis_path = os.path.join(save_dir, f"epoch_{epoch}_vis_{img_name_suffix}.jpg")
    img_pil = F_vis.to_pil_image(img)
    img_pil.save(vis_path)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = torch.stack(images).to(device)
        gt_boxes = [t['boxes'].to(device) for t in targets]
        gt_labels = [t['labels'].to(device) for t in targets]

        out = model(images)

        loss = loss_fn(
            rpn_out=(out['rpn_scores'], out['rpn_deltas']),
            detector_out=(out['cls_scores'], out['bbox_deltas']),
            anchors=out['anchors'],
            proposals=out['proposals'],
            gt_boxes=gt_boxes,
            gt_labels=gt_labels
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(loader)


def validate_during_training(model, dataset, device, epoch, checkpoint_dir, loss_fn):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn)

    print(f"  Validating on VAL set  ")

    total_val_loss = 0

    groups_seen = {'G1': False, 'G2': False, 'G3': False}
    group_ious = {'G1': [], 'G2': [], 'G3': []}

    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            images = torch.stack(images).to(device)
            gt_boxes = [t['boxes'].to(device) for t in targets]
            gt_labels = [t['labels'].to(device) for t in targets]
            img_shape = images.shape[-2:]

            out = model(images)

            loss = loss_fn(
                rpn_out=(out['rpn_scores'], out['rpn_deltas']),
                detector_out=(out['cls_scores'], out['bbox_deltas']),
                anchors=out['anchors'],
                proposals=out['proposals'],
                gt_boxes=gt_boxes,
                gt_labels=gt_labels
            )
            total_val_loss += loss.item()

            results = model.detector.post_process(
                out['cls_scores'],
                out['bbox_deltas'],
                out['proposals'][0],
                img_shape,
                score_thresh=0.6
            )

            idx = targets[0]['image_id'].item()
            original_path = Path(dataset.samples[idx][0])

            for grp in groups_seen.keys():
                if grp in str(original_path.parent):
                    if not groups_seen[grp]:
                        save_img(
                            images[0],
                            results,
                            targets[0],
                            epoch,
                            checkpoint_dir,
                            device,
                            img_name_suffix=grp
                        )
                        groups_seen[grp] = True

                    current_gt_boxes_t = gt_boxes[0]
                    current_gt_labels_t = gt_labels[0]

                    gt_list = []
                    for k in range(len(current_gt_boxes_t)):
                        gt_list.append((
                            int(current_gt_labels_t[k].item()),
                            *current_gt_boxes_t[k].tolist()
                        ))

                    pred_list = []
                    for res in results:
                        p_cls_id, p_box, p_score = res
                        real_cls_id = int(p_cls_id) + 1
                        pred_list.append((real_cls_id, *p_box.tolist()))

                    if len(gt_list) > 0 and len(pred_list) > 0:
                        for p in pred_list:
                            cls_p, x1_p, y1_p, x2_p, y2_p = p
                            p_xyxy = (x1_p, y1_p, x2_p, y2_p)

                            best_iou = 0.0
                            for g in gt_list:
                                cls_g, x1_g, y1_g, x2_g, y2_g = g
                                if cls_p != cls_g:
                                    continue
                                g_xyxy = (x1_g, y1_g, x2_g, y2_g)

                                iou_val = metrics.iou_xyxy(p_xyxy, g_xyxy)
                                best_iou = max(best_iou, iou_val)

                            group_ious[grp].append(best_iou)

    avg_val_loss = total_val_loss / len(loader)

    total_miou = 0
    count = 0
    for grp in ['G1', 'G2', 'G3']:
        ious = group_ious[grp]
        if len(ious) > 0:
            mean_iou = float(np.mean(ious))
        else:
            mean_iou = 0.0

        print(f"    Group {grp} mIoU: {mean_iou:.4f}")
        total_miou += mean_iou
        count += 1

    avg_miou = total_miou / count if count > 0 else 0.0
    print(f"  [Epoch {epoch}] Val Loss: {avg_val_loss:.4f} | Avg mIoU: {avg_miou:.4f}")


def generate_test_predictions(model, root_dir, device):
    print("\n Generating final predictions for TEST Set  ")
    test_dataset = RecursiveDetDataset(root_dir=root_dir, split="TEST", require_labels=False)
    loader = DataLoader(test_dataset, batch_size=1, collate_fn=custom_collate_fn)

    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            img_tensor = torch.stack(images).to(device)
            img_shape = img_tensor.shape[-2:]

            out = model(img_tensor)
            results = model.detector.post_process(out['cls_scores'], out['bbox_deltas'], out['proposals'][0], img_shape,
                                                  score_thresh=0.6)

            idx = targets[0]['image_id'].item()

            img_path = Path(test_dataset.samples[idx][0])

            group_name = img_path.parent.name
            test_root = img_path.parent.parent.parent

            label_dir = test_root / "labels" / group_name
            os.makedirs(label_dir, exist_ok=True)

            save_name = img_path.stem + ".txt"
            save_path = label_dir / save_name

            with open(save_path, 'w') as f:
                for res in results:
                    cls_id, box, score = res
                    x1, y1, x2, y2 = box.tolist()
                    w_abs, h_abs = x2 - x1, y2 - y1

                    x_c_n, y_c_n = (x1 + 0.5 * w_abs) / 640.0, (y1 + 0.5 * h_abs) / 480.0
                    w_n, h_n = w_abs / 640.0, h_abs / 480.0

                    f.write(f"{int(cls_id)} {x_c_n:.6f} {y_c_n:.6f} {w_n:.6f} {h_n:.6f} {score:.6f}\n")

    print(f"Test predictions complete. Saved at {root_dir}/TEST/labels/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    ROOT_DIR = Path(__file__).resolve().parent / "data"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 2
    BATCH_SIZE = 4
    LR = 0.005
    EPOCHS = 20

    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model = FasterRCNN(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 10, 15], gamma=0.1)
    loss_func = FasterRCNNLoss()

    start_epoch = 0

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print(f"Loading checkpoint from {args.checkpoint}...")
            checkpoint = torch.load(args.checkpoint, map_location=DEVICE)

            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                if args.mode == "train":
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    start_epoch = checkpoint["epoch"]
                    print(f"Resuming training from epoch {start_epoch}")
            else:
                model.load_state_dict(checkpoint)
                print("Loaded model weights only.")
        else:
            print(f"Checkpoint file {args.checkpoint} not found")
            return

    if args.mode == "train":
        print(f"Starting training on {DEVICE} from epoch {start_epoch + 1} to {EPOCHS}...")

        train_dataset = RecursiveDetDataset(root_dir=ROOT_DIR, split="Train", require_labels=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
        val_dataset = RecursiveDetDataset(root_dir=ROOT_DIR, split="VAL", require_labels=True)

        for epoch in range(start_epoch, EPOCHS):
            print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_func, DEVICE)
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Epoch Loss: {train_loss:.4f} | LR: {current_lr:.6f}")

            validate_during_training(model, val_dataset, DEVICE, epoch + 1, CHECKPOINT_DIR, loss_func)

            save_path = os.path.join(CHECKPOINT_DIR, f"faster_rcnn_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)

    elif args.mode == "test":
        generate_test_predictions(model, ROOT_DIR, DEVICE)


if __name__ == "__main__":
    main()
