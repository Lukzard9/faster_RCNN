import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from pathlib import Path
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F_vis

# Import our custom classes
from dataset import RecursiveDetDataset, custom_collate_fn
from model.model import FasterRCNN
from model.loss import FasterRCNNLoss
import metrics
import os


def save_visualization(image_tensor, preds, targets, epoch, save_dir, device):
    """
    Draws GT (Green) and Predictions (Red) on the image and saves it.
    """
    # 1. Denormalize Image (Reversing the transforms in dataset.py)
    # Mean and Std from ImageNet (used in your dataset.py)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    # img = (img * std) + mean
    img = image_tensor * std + mean
    # Clip to 0-1 range just in case, then scale to 255 and convert to uint8 (required for drawing)
    img = torch.clamp(img, 0, 1)
    img = (img * 255).type(torch.uint8)

    # 2. Draw Ground Truth Boxes (Green)
    gt_boxes = targets['boxes']
    if len(gt_boxes) > 0:
        img = draw_bounding_boxes(img, gt_boxes, colors="green", width=3)

    # 3. Draw Predictions (Red)
    # preds is list of (class_id, box, score)
    pred_boxes = [p[1] for p in preds]
    pred_scores = [f"{p[2]:.2f}" for p in preds]

    if len(pred_boxes) > 0:
        pred_boxes_tensor = torch.stack(pred_boxes)
        img = draw_bounding_boxes(img, pred_boxes_tensor, labels=pred_scores, colors="red", width=2)

    # 4. Save Image
    vis_path = os.path.join(save_dir, f"epoch_{epoch}_vis.jpg")
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


def validate_and_save_results(model, dataset, device, output_dir, epoch, checkpoint_dir):
    """
    Runs inference, saves YOLO .txt files for metrics, and saves a visualization.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn)

    print(f"  Running validation... (Saving preds to {output_dir})")

    vis_done = False  # Flag to ensure we only save ONE visualization per epoch

    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            img_tensor = torch.stack(images).to(device)
            img_shape = img_tensor.shape[-2:]

            # Forward
            out = model(img_tensor)

            # Post-Process
            results = model.detector.post_process(
                out['cls_scores'],
                out['bbox_deltas'],
                out['proposals'][0],
                img_shape
            )

            # --- VISUALIZATION BLOCK ---
            if not vis_done:
                # Visualize the first image of the batch
                save_visualization(
                    img_tensor[0],
                    results,
                    targets[0],  # Using the target directly (CPU/GPU handled inside)
                    epoch,
                    checkpoint_dir,
                    device
                )
                vis_done = True
            # ---------------------------

            # Save Text Files for Metrics
            idx = targets[0]['image_id'].item()
            original_path = Path(dataset.samples[idx][1])
            save_name = original_path.name

            with open(os.path.join(output_dir, save_name), 'w') as f:
                for res in results:
                    cls_id, box, score = res
                    x1, y1, x2, y2 = box.tolist()

                    # Normalize for YOLO format
                    w_abs = x2 - x1
                    h_abs = y2 - y1
                    x_c_n = (x1 + 0.5 * w_abs) / 640.0
                    y_c_n = (y1 + 0.5 * h_abs) / 480.0
                    w_n = w_abs / 640.0
                    h_n = h_abs / 480.0

                    f.write(f"{int(cls_id)} {x_c_n:.6f} {y_c_n:.6f} {w_n:.6f} {h_n:.6f} {score:.6f}\n")

    # Run Metrics
    gt_root_base = dataset.root / dataset.split / "Labels"
    total_miou = 0
    groups = ['G1', 'G2', 'G3']
    count = 0

    for grp in groups:
        gt_dir = gt_root_base / grp
        if not gt_dir.exists(): continue

        # Calculate mIoU
        # Note: Suppressing print inside loop if you want cleaner logs,
        # or keep it to see per-group progress.
        mean_iou, _ = metrics.compute_mean_iou(str(gt_dir), output_dir)
        print(f"Mean IoU for group {grp}: {mean_iou}")
        total_miou += mean_iou
        count += 1

    avg_miou = total_miou / count if count > 0 else 0.0
    print(f"  [Epoch {epoch}] Validation mIoU: {avg_miou:.4f}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Configuration
    ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 2
    BATCH_SIZE = 4
    LR = 0.005
    EPOCHS = 5

    # 1. Setup Folders
    CHECKPOINT_DIR = "checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # We use a temp folder for txt files so metrics.py can read them
    # We overwrite this every epoch, which is fine.
    TEMP_PRED_DIR = "temp_val_preds"

    # 2. Dataset & Loader
    train_dataset = RecursiveDetDataset(root_dir=ROOT_DIR, split="Train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=custom_collate_fn)

    test_dataset = RecursiveDetDataset(root_dir=ROOT_DIR, split="Test")

    # 3. Model & Optimization
    model = FasterRCNN(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)
    loss_func = FasterRCNNLoss()

    # 4. Training Loop
    print(f"Starting training on {DEVICE}...")

    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_func, DEVICE)
        print(f"  Epoch Loss: {train_loss:.4f}")

        # Save Checkpoint to Folder
        save_path = os.path.join(CHECKPOINT_DIR, f"faster_rcnn_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)

        # Validate & Visualize
        validate_and_save_results(
            model,
            test_dataset,
            DEVICE,
            output_dir=TEMP_PRED_DIR,
            epoch=epoch + 1,
            checkpoint_dir=CHECKPOINT_DIR
        )

    print(f"\nTraining Complete. Checkpoints and visuals are in '{CHECKPOINT_DIR}'.")