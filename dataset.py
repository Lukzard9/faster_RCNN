import torch
import os
from pathlib import Path
from PIL import Image
from torchvision import transforms
import metrics 

class RecursiveDetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split="Train", transform=None):
        """
        Args:
            root_dir (str): Path to the root dataset folder.
            split (str): "Train", "Val", or "Test".
            transform (callable, optional): Transform to be applied on the image.
        """
        self.root = Path(root_dir)
        self.split = split
        self.img_width = 640
        self.img_height = 480
        
        # Sub-folders structure
        self.groups = ['G1', 'G2', 'G3']
        
        # List to store (image_path, label_path) tuples
        self.samples = []
        self._load_samples()

        # Define default transform for ResNet50 if none is provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_height, self.img_width)),
                transforms.ToTensor(),
                # Standard ImageNet normalization required for pre-trained ResNet
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def _load_samples(self):
        """Recursively loads images and matches them with labels."""
        img_root = self.root / self.split / "IMAGES"
        lbl_root = self.root / self.split / "LABELS"

        for group in self.groups:
            img_group_path = img_root / group
            lbl_group_path = lbl_root / group

            if not img_group_path.exists():
                continue

            # Sort to ensure alignment
            img_files = sorted(list(img_group_path.glob("*.jpg")) + list(img_group_path.glob("*.png")))
            
            for img_path in img_files:
                # Construct corresponding label path (change extension to .txt)
                lbl_name = img_path.stem + ".txt"
                lbl_path = lbl_group_path / lbl_name
                
                if lbl_path.exists():
                    self.samples.append((str(img_path), str(lbl_path)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        # 1. Load Image
        image = Image.open(img_path).convert("RGB")
        
        # 2. Read Labels using YOUR metrics.py function
        # Returns list of (cls, x, y, w, h) normalized
        raw_boxes = metrics.read_yolo_boxes(lbl_path, has_conf=False)

        boxes = []
        labels = []

        for item in raw_boxes:
            cls_id, x_c, y_c, w, h = item
            
            # 3. Convert Normalized Center (YOLO) -> Normalized Corners
            # Using YOUR metrics.py function
            x_min_n, y_min_n, x_max_n, y_max_n = metrics.xywh_to_xyxy(x_c, y_c, w, h)

            # 4. Denormalize: Scale to Absolute Pixel Coordinates
            # Faster R-CNN expects: [x1, y1, x2, y2] in pixels
            x_min = x_min_n * self.img_width
            y_min = y_min_n * self.img_height
            x_max = x_max_n * self.img_width
            y_max = y_max_n * self.img_height

            boxes.append([x_min, y_min, x_max, y_max])
            
            # IMPORTANT: Faster R-CNN reserves class 0 for background.
            # If your dataset classes are 0-indexed (0, 1, 2...), 
            # we typically add 1 so the model treats 0 as background.
            labels.append(int(cls_id) + 1)

        # Handle images with no objects (background only)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # 5. Wrap in target dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        # Apply Transforms to Image
        if self.transform:
            image = self.transform(image)

        return image, target

def custom_collate_fn(batch):
    """
    Standard PyTorch collate fails because different images have 
    different numbers of bounding boxes.
    """
    return tuple(zip(*batch))