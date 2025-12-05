import torch
import torch.utils.data
from pathlib import Path
from PIL import Image
from torchvision import transforms
import metrics 

class RecursiveDetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split="Train", transform=None, require_labels=True):
        """
        Args:
            root_dir (str): Path to the root dataset folder.
            split (str): "Train", "Val", or "Test".
            transform (callable, optional): Transform to be applied on the image.
            require_labels (bool): If True, skips images without corresponding .txt files.
                                   Set False for the final TEST set generation.
        """
        self.root = Path(root_dir)
        self.split = split
        self.require_labels = require_labels
        self.img_width = 640
        self.img_height = 480
        
        # Sub-folders structure
        self.groups = ['G1', 'G2', 'G3']
        
        # List to store (image_path, label_path) tuples
        # label_path will be None if require_labels=False and no label is found
        self.samples = []
        self._load_samples()

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((self.img_height, self.img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def _load_samples(self):
        img_root = self.root / self.split / "IMAGES"
        lbl_root = self.root / self.split / "LABELS"

        for group in self.groups:
            img_group_path = img_root / group
            lbl_group_path = lbl_root / group

            if not img_group_path.exists():
                continue

            # Sort to ensure consistent order
            img_files = sorted(list(img_group_path.glob("*.jpg")) + list(img_group_path.glob("*.png")))
            
            for img_path in img_files:
                lbl_name = img_path.stem + ".txt"
                lbl_path = lbl_group_path / lbl_name
                
                if lbl_path.exists():
                    self.samples.append((str(img_path), str(lbl_path)))
                elif not self.require_labels:
                    # Keep image even if label is missing (for TEST inference)
                    self.samples.append((str(img_path), None))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        # 1. Load Image
        image = Image.open(img_path).convert("RGB")
        
        boxes = []
        labels = []

        # 2. Read Labels (if they exist)
        if lbl_path is not None:
            raw_boxes = metrics.read_yolo_boxes(lbl_path, has_conf=False)

            for item in raw_boxes:
                cls_id, x_c, y_c, w, h = item
                x_min_n, y_min_n, x_max_n, y_max_n = metrics.xywh_to_xyxy(x_c, y_c, w, h)

                x_min = x_min_n * self.img_width
                y_min = y_min_n * self.img_height
                x_max = x_max_n * self.img_width
                y_max = y_max_n * self.img_height

                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(cls_id) + 1)

        # 3. Handle No Labels / Empty Labels
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        if self.transform:
            image = self.transform(image)

        return image, target

def custom_collate_fn(batch):
    return tuple(zip(*batch))