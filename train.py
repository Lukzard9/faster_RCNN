import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from pathlib import Path

# Import our custom classes
from dataset import RecursiveDetDataset, custom_collate_fn
from model import FasterRCNN
from loss import FasterRCNNLoss
import metrics # Your provided metrics file

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (images, targets) in enumerate(loader):
        # 1. Move data to GPU
        # images is a tuple/list from collate, need to stack for backbone
        images = torch.stack(images).to(device)
        
        # Unpack targets
        gt_boxes = [t['boxes'].to(device) for t in targets]
        gt_labels = [t['labels'].to(device) for t in targets]

        # 2. Forward Pass
        out = model(images)
        
        # 3. Compute Loss
        loss = loss_fn(
            rpn_out=(out['rpn_scores'], out['rpn_deltas']),
            detector_out=(out['cls_scores'], out['bbox_deltas']),
            anchors=out['anchors'],
            proposals=out['proposals'],
            gt_boxes=gt_boxes,
            gt_labels=gt_labels
        )
        
        # 4. Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(loader)

def validate_and_save_results(model, dataset, device, output_dir):
    """
    Runs inference, converts results to YOLO format, saves .txt files,
    and runs the provided metrics.py evaluation.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # Use batch_size=1 for validation to easily map back to filenames
    loader = DataLoader(dataset, batch_size=1, collate_fn=custom_collate_fn)
    
    print(f"Generating predictions in {output_dir}...")
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            img_tensor = torch.stack(images).to(device)
            img_shape = img_tensor.shape[-2:] # H, W
            
            # Forward
            out = model(img_tensor)
            
            # Post-Process (NMS, Score Thresholding)
            # This returns [(class_id, [x1, y1, x2, y2], score), ...]
            results = model.detector.post_process(
                out['cls_scores'], 
                out['bbox_deltas'], 
                out['proposals'][0], # Only 1 image
                img_shape
            )
            
            # Retrieve original filename to save matching .txt
            # We access the dataset via the original index stored in target
            idx = targets[0]['image_id'].item()
            # dataset.samples is [(img_path, lbl_path), ...]
            original_path = Path(dataset.samples[idx][1]) 
            save_name = original_path.name # e.g., "img1.txt"
            
            # Write to file in YOLO format for metrics.py
            # Format: cls x_c y_c w h conf
            with open(os.path.join(output_dir, save_name), 'w') as f:
                for res in results:
                    cls_id, box, score = res
                    x1, y1, x2, y2 = box.tolist()
                    
                    # Convert Absolute Corners -> Normalized Center (YOLO)
                    # Note: metrics.py expects normalized coordinates
                    w_abs = x2 - x1
                    h_abs = y2 - y1
                    x_c_abs = x1 + 0.5 * w_abs
                    y_c_abs = y1 + 0.5 * h_abs
                    
                    x_c_n = x_c_abs / 640.0
                    y_c_n = y_c_abs / 480.0
                    w_n = w_abs / 640.0
                    h_n = h_abs / 480.0
                    
                    # Write line: "cls x y w h score"
                    # Note: We use cls_id directly (0, 1...) 
                    # post_process already subtracted 1 to match ground truth 0-indexing
                    f.write(f"{int(cls_id)} {x_c_n:.6f} {y_c_n:.6f} {w_n:.6f} {h_n:.6f} {score:.6f}\n")

    print("Predictions saved. Running metrics.py logic...")
    
    # Now we call your provided metric function
    # We need the ground truth folder. 
    # Since your dataset is nested (G1, G2, G3), we might need to point to specific folders 
    # or aggregate them. For the exam, usually, you test on one Group or a flattened folder.
    # Here is an example checking "G1" specifically, or you can loop over groups.
    
    gt_root_base = dataset.root / dataset.split / "Labels"
    
    total_miou = 0
    groups = ['G1', 'G2', 'G3']
    for grp in groups:
        gt_dir = gt_root_base / grp
        if not gt_dir.exists(): continue
        
        # calculate mean IoU for this group
        # Note: Your metrics.py compares files in pred_dir matching files in gt_dir.
        # Since we dumped ALL predictions into 'output_dir', it will find them.
        mean_iou, per_image = metrics.compute_mean_iou(str(gt_dir), output_dir)
        print(f"Group {grp} mIoU: {mean_iou:.4f}")
        total_miou += mean_iou
        
    print(f"Average mIoU across groups: {total_miou / len(groups):.4f}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Configuration
    ROOT_DIR = "C:\\Users\\User\\Desktop\\CodeCave\\Deep\\faster_rcnn\\data" # Your specific path
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 2
    BATCH_SIZE = 4
    LR = 0.005
    EPOCHS = 5

    # 1. Dataset & Loader
    train_dataset = RecursiveDetDataset(root_dir=ROOT_DIR, split="Train")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, collate_fn=custom_collate_fn)
    
    test_dataset = RecursiveDetDataset(root_dir=ROOT_DIR, split="Test")

    # 2. Model & Optimization
    model = FasterRCNN(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Optimizer: SGD with Momentum is standard for Object Detection
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)
    
    loss_func = FasterRCNNLoss()

    # 3. Training Loop
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_func, DEVICE)
        print(f"Epoch Loss: {train_loss:.4f}")
        
        # Save Checkpoint
        torch.save(model.state_dict(), f"faster_rcnn_epoch_{epoch}.pth")

    # 4. Final Evaluation
    print("\n--- Starting Evaluation ---")
    validate_and_save_results(
        model, 
        test_dataset, 
        DEVICE, 
        output_dir="validation_preds"
    )