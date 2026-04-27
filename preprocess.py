import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class PairedSketchDataset(Dataset):
    """
    Expects side-by-side images (sketch | photo) as produced by
    the Berkeley pix2pix datasets. Each image is split 50/50.
    """
    def __init__(self, root: str, split: str = "train", img_size: int = 256):
        self.files = sorted([
            os.path.join(root, split, f)
            for f in os.listdir(os.path.join(root, split))
            if f.lower().endswith((".jpg", ".png"))
        ])
        assert len(self.files) > 0, f"No images found in {root}/{split}"

        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),   # → [-1, 1]
        ])

        # Augmentation only for training
        self.aug = T.RandomHorizontalFlip() if split == "train" else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        w, h = img.size
        mid = w // 2

        sketch = self.transform(img.crop((0,   0, mid, h)))
        photo  = self.transform(img.crop((mid, 0, w,   h)))

        if self.aug:
            # apply identical flip to both
            seed = torch.randint(0, 2, (1,)).item()
            if seed:
                sketch = T.functional.hflip(sketch)
                photo  = T.functional.hflip(photo)

        return sketch, photo


def get_loaders(root: str, img_size: int = 256, batch_size: int = 8, num_workers: int = 2):
    train_ds = PairedSketchDataset(root, "train", img_size)
    val_ds   = PairedSketchDataset(root, "val",   img_size)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
