import os
import re
from glob import glob
from PIL import Image
import torch
from torch.utils import data
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF
from torchvision.transforms.v2 import InterpolationMode
import random
import numpy as np

IMAGE_EXTS = ("*.jpg", "*.png", "*.tif", "*.gif", "*.ppm")

def list_images(folder):
    files = []
    for ext in IMAGE_EXTS:
        files.extend(glob(os.path.join(folder, ext)))
    return sorted(files)

class ImageFolder(data.Dataset):
    """
    Generic dataset loader that supports:
      - CHASE_DB1 (image.jpg -> image_1stHO.png)
      - DRIVE (21_training.tif -> 21_manual1.gif)
      - STARE (im0001.ppm -> im0001.vk.ppm)
      - Lung (image and mask have identical names)
      - SkinCancer (ISIC_xxx.jpg -> ISIC_xxx_segmentation.png)

    Returns:
      image (C,H,W) float normalized to ~[-1,1],
      GT    (1,H,W) float {0,1}
    """
    def __init__(self, root, image_size=256, mode="train", augmentation_prob=0.7):
        self.root = root
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        self.image_size = image_size

        # Load only image files from the requested folder (don't pick up *_GT or other files)
        self.img_paths = list_images(root)
        self.gt_dir = root.replace(mode, mode + "_GT")
        if not os.path.isdir(self.gt_dir):
            # some setups may have same folder names; ensure path exists
            self.gt_dir = self.gt_dir  # keep as-is, will test existence per-file

        print(f"[{mode}] {len(self.img_paths)} images loaded from {root}")

        # transforms
        self.image_transform = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # GT: single channel, resized, converted to binary 0/1
        self.gt_transform = T.Compose([
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True)
        ])

    def __len__(self):
        return len(self.img_paths)

    def paired_augment(self, image, GT):
        # ---- Flip ----
        if random.random() < 0.5:
            image = TF.horizontal_flip(image)
            GT = TF.horizontal_flip(GT)
        if random.random() < 0.5:
            image = TF.vertical_flip(image)
            GT = TF.vertical_flip(GT)

        # ---- Rotation ----
        if random.random() < 0.5:
            angle = random.choice([0, 90, 180, 270])
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BICUBIC)
            GT = TF.rotate(GT, angle, interpolation=InterpolationMode.NEAREST)

        # ---- Random Resized Crop (ZOOM) ----
        if random.random() < 0.5:
            i, j, h, w = T.RandomResizedCrop.get_params(
                image,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1)
            )
            image = TF.resized_crop(image, i, j, h, w,
                                    (self.image_size, self.image_size),
                                    interpolation=InterpolationMode.BICUBIC)
            GT = TF.resized_crop(GT, i, j, h, w,
                                   (self.image_size, self.image_size),
                                   interpolation=InterpolationMode.NEAREST)

        # ---- Color Jitter (image only) ----
        if random.random() < 0.5:
            color_aug = T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
            )
            image = color_aug(image)

        # ---- Gaussian Blur (image only) ----
        if random.random() < 0.5:
            image = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))(image)

        # ---- Gaussian Noise (image only) ----
        if random.random() < 0.5:
            image = np.array(image)
            noise = np.random.normal(0, 5, image.shape).astype(np.uint8)
            image = Image.fromarray(np.clip(image + noise, 0, 255))

        # ---- Random Brightness drop (image only) ----
        if random.random() < 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.5, 0.9))

        return image, GT

    def _chase_gt_name(self, img_name):
        # image.jpg -> image_1stHO.png
        base = os.path.basename(img_name)
        name_no_ext = os.path.splitext(base)[0]
        return f"{name_no_ext}_1stHO.png"

    def _drive_gt_name(self, img_name):
        # DRIVE images like "21_training.tif" or "01_test.tif" -> want "21_manual1.gif"
        base = os.path.basename(img_name)
        m = re.match(r"(\d+)_", base)
        if m:
            idx = m.group(1)
            return f"{idx}_manual1.gif"
        # fallback: try replace suffix
        return base.replace("_training.tif", "_manual1.gif").replace("_test.tif", "_manual1.gif")

    def _stare_gt_name(self, img_name):
        # im0001.ppm -> im0001.vk.ppm
        base = os.path.basename(img_name)
        name_no_ext = os.path.splitext(base)[0]
        return f"{name_no_ext}.vk.ppm"

    def _skin_gt_name(self, img_name):
        # ISIC naming: e.g. ISIC_0000000.jpg -> ISIC_0000000_segmentation.png
        base = os.path.basename(img_name)
        name_no_ext = os.path.splitext(base)[0]
        return f"{name_no_ext}_segmentation.png"

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_base = os.path.basename(img_path)

        # choose gt path based on dataset type in root path
        if "CHASE_DB1" in self.root:
            gt_name = self._chase_gt_name(img_base)
        elif "DRIVE" in self.root:
            gt_name = self._drive_gt_name(img_base)
        elif "STARE" in self.root:
            gt_name = self._stare_gt_name(img_base)
        elif "Lung" in self.root:
            gt_name = img_base  # identical filenames
        elif "SkinCancer" in self.root:
            gt_name = self._skin_gt_name(img_base)
        else:
            # generic: assume same basename with png extension in GT folder
            gt_name = os.path.splitext(img_base)[0] + ".png"

        gt_path = os.path.join(self.gt_dir, gt_name)
        image = Image.open(img_path).convert("RGB")

        missing_gt_for_image = []
        missing_gt = []
        if os.path.exists(gt_path):
            GT = Image.open(gt_path).convert("L")
        else:
            missing_gt_for_image.append(img_path)
            missing_gt.append(gt_path)
            GT = Image.new("L", image.size, 0)
        
        if missing_gt:
            print(f"Warning: missing GTs for {missing_gt_for_image}. Expected at: {missing_gt}! Using empty GTs.")

        # Data augmentation (image & GT must be transformed identically)
        if self.mode == "train" and random.random() < self.augmentation_prob:
            image, GT = self.paired_augment(image, GT)

        image = self.image_transform(image)     # (3, H, W), normalized to [-1,1]
        GT = self.gt_transform(GT)              # (1, H, W), in [0,1]

        # Binarize GT: any non-zero -> 1.0 (makes evaluation and BCE straightforward)
        GT = (GT > 0.5).float()

        return image, GT


def get_loader(image_path, image_size, batch_size, mode, num_workers=2, augmentation_prob=0.7):
    dataset = ImageFolder(image_path, image_size, mode, augmentation_prob)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=(mode=="train"), num_workers=num_workers)
