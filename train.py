import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from pathlib import Path
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F_vis

from dataset import RecursiveDetDataset, custom_collate_fn
from model.model import FasterRCNN
from model.loss import FasterRCNNLoss
import metrics 

# MAP: Dataset adds 1 to YOLO class. 
# YOLO 0 (Strawberry) -> Model 1
# YOLO 1 (Olive)      -> Model 2
CLASS_MAP = {
    1: "Strawberry",
    2: "Olive"
}

def save_visualization(image_tensor, preds, targets, epoch, save_dir, device, img_name_suffix=""):
    """
    Draws GT (Green) and Predictions (Red) with Class Names.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    img = image_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    img = (img * 255).type(torch.uint8)

    # Draw GT (Green)
    gt_boxes = targets['boxes']
    if gt_boxes.numel() > 0:
        img = draw_bounding_boxes(img, gt_boxes, colors="green", width=3)

    # Draw Predictions (Red)
    pred_boxes = [p[1] for p in preds]
    # NEW: Show Class Name + Score
    pred_labels = []
    for p in preds:
        cls_id = int(p[0]) + 1 # Model outputs 0-indexed class (0=Strawberry), map back to 1/2
        cls_name = CLASS_MAP.get(cls_id, str(cls_id))
        score = p[2]
        pred_labels.append(f"{cls_name}: {score:.2f}")
    
    if len(pred_boxes) > 0:
        pred_boxes_tensor = torch.stack(pred_boxes)
        img = draw_bounding_boxes(img, pred_boxes_tensor, labels=pred_labels, colors="red", width=2)

    # Save with specific suffix (e.g., "G1", "G2")
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
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(loader)

def validate_during_training(model, dataset, device, output_dir, epoch, checkpoint_dir, loss_fn):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn)
    
    print(f"  Validating on VAL set... (Results -> {output_dir})")
    
    total_val_loss = 0
    
    # Track which groups we have visualized this epoch
    groups_seen = {'G1': False, 'G2': False, 'G3': False}
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            images = torch.stack(images).to(device)
            gt_boxes = [t['boxes'].to(device) for t in targets]
            gt_labels = [t['labels'].to(device) for t in targets]
            img_shape = images.shape[-2:]
            
            # Forward
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

            # Post-Process (Use 0.3 thresh to see weak detections)
            results = model.detector.post_process(
                out['cls_scores'], 
                out['bbox_deltas'], 
                out['proposals'][0], 
                img_shape,
                score_thresh=0.3 
            )
            
            # --- INTELLIGENT VISUALIZATION ---
            idx = targets[0]['image_id'].item()
            original_path = Path(dataset.samples[idx][0]) 
            
            # Detect group from filename path (e.g. .../G1/img.jpg)
            # We assume folder structure is .../G1/...
            # If path contains 'G1', visualize it if we haven't yet.
            for grp in groups_seen.keys():
                if grp in str(original_path.parent) and not groups_seen[grp]:
                    save_visualization(
                        images[0], 
                        results, 
                        targets[0], 
                        epoch, 
                        checkpoint_dir, 
                        device,
                        img_name_suffix=grp # Appends G1, G2, or G3 to filename
                    )
                    groups_seen[grp] = True
            # ---------------------------------

            # Save Predictions
            save_name = original_path.stem + ".txt"
            with open(os.path.join(output_dir, save_name), 'w') as f:
                for res in results:
                    cls_id, box, score = res
                    x1, y1, x2, y2 = box.tolist()
                    w_abs, h_abs = x2 - x1, y2 - y1
                    x_c_n, y_c_n = (x1 + 0.5 * w_abs) / 640.0, (y1 + 0.5 * h_abs) / 480.0
                    w_n, h_n = w_abs / 640.0, h_abs / 480.0
                    f.write(f"{int(cls_id)} {x_c_n:.6f} {y_c_n:.6f} {w_n:.6f} {h_n:.6f} {score:.6f}\n")

    avg_val_loss = total_val_loss / len(loader)
    
    # Metrics
    gt_root_base = dataset.root / dataset.split / "LABELS"
    total_miou = 0
    groups = ['G1', 'G2', 'G3']
    count = 0
    
    for grp in groups:
        gt_dir = gt_root_base / grp
        if not gt_dir.exists(): continue
        mean_iou, _ = metrics.compute_mean_iou(str(gt_dir), output_dir)
        # Print per-group mIoU to see if Strawberries (G2/G3) are failing
        print(f"    Group {grp} mIoU: {mean_iou:.4f}")
        total_miou += mean_iou
        count += 1
        
    avg_miou = total_miou / count if count > 0 else 0.0
    print(f"  [Epoch {epoch}] Val Loss: {avg_val_loss:.4f} | Avg mIoU: {avg_miou:.4f}")

def generate_test_predictions(model, root_dir, device, output_dir):
    print("\n--- Generating Final Predictions for TEST Set ---")
    test_dataset = RecursiveDetDataset(root_dir=root_dir, split="TEST", require_labels=False)
    loader = DataLoader(test_dataset, batch_size=1, collate_fn=custom_collate_fn)
    
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            img_tensor = torch.stack(images).to(device)
            img_shape = img_tensor.shape[-2:]
            
            out = model(img_tensor)
            results = model.detector.post_process(out['cls_scores'], out['bbox_deltas'], out['proposals'][0], img_shape, score_thresh=0.05) 
            
            idx = targets[0]['image_id'].item()
            img_path = Path(test_dataset.samples[idx][0])
            save_name = img_path.stem + ".txt"
            
            with open(os.path.join(output_dir, save_name), 'w') as f:
                for res in results:
                    cls_id, box, score = res
                    x1, y1, x2, y2 = box.tolist()
                    w_abs, h_abs = x2 - x1, y2 - y1
                    x_c_n, y_c_n = (x1 + 0.5 * w_abs) / 640.0, (y1 + 0.5 * h_abs) / 480.0
                    w_n, h_n = w_abs / 640.0, h_abs / 480.0
                    f.write(f"{int(cls_id)} {x_c_n:.6f} {y_c_n:.6f} {w_n:.6f} {h_n:.6f} {score:.6f}\n")
    print("Test predictions complete.")

if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parent / "data"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 2 
    BATCH_SIZE = 4
    LR = 0.005
    EPOCHS = 15 # Kept at 15
    
    CHECKPOINT_DIR = "checkpoints"
    VAL_PRED_DIR = "temp_val_preds"
    TEST_PRED_DIR = "final_test_predictions"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_dataset = RecursiveDetDataset(root_dir=ROOT_DIR, split="Train", require_labels=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    val_dataset = RecursiveDetDataset(root_dir=ROOT_DIR, split="VAL", require_labels=True)

    model = FasterRCNN(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)
    loss_func = FasterRCNNLoss()

    print(f"Starting training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_func, DEVICE)
        print(f"  Epoch Loss: {train_loss:.4f}")
        
        save_path = os.path.join(CHECKPOINT_DIR, f"faster_rcnn_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        
        validate_during_training(model, val_dataset, DEVICE, VAL_PRED_DIR, epoch+1, CHECKPOINT_DIR, loss_func)
        
    generate_test_predictions(model, ROOT_DIR, DEVICE, TEST_PRED_DIR)