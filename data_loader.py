import os
from glob import glob
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
import random


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=256, mode="train", augmentation_prob=0.4):
        self.root = root
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        self.image_size = image_size

        self.img_paths = sorted(glob(os.path.join(root, "*")))
        self.gt_paths = sorted(glob(os.path.join(root.replace(mode, mode + "_GT"), "*")))

        print(f"[{mode}] {len(self)} images loaded")

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        name = os.path.basename(img_path)

        gt_dir = self.root.replace(self.mode, self.mode + "_GT")
        # CHASE_DB1, DRIVE, STARE, Lung, SkinCancer
        if "CHASE_DB1" in self.root:
            gt_path = os.path.join(gt_dir, name.replace(".jpg", "_1stHO.png"))
        elif "DRIVE" in self.root:
            gt_path = os.path.join(gt_dir, name.replace("_training.tif", "_manual1.gif").replace("_test.tif", "_manual1.gif"))
        elif "STARE" in self.root:
            gt_path = os.path.join(gt_dir, name.replace(".ppm", ".vk.ppm"))
        elif "Lung" in self.root:
            gt_path = os.path.join(gt_dir, name)
        elif "SkinCancer" in self.root:
            gt_path = os.path.join(gt_dir, name.replace(".jpg", "_segmentation.png"))

        image = Image.open(img_path).convert("RGB")

        if os.path.exists(gt_path):
            GT = Image.open(gt_path).convert("L")
        else:
            print(f"Warning: GT mask not found for {img_path}. Using empty mask.")
            GT = Image.new("L", image.size)

        # Augment
        if self.mode == "train" and random.random() < self.augmentation_prob:
            if random.random() < 0.5:
                image = F.hflip(image)
                GT = F.hflip(GT)
            if random.random() < 0.5:
                image = F.vflip(image)
                GT = F.vflip(GT)

        transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor()
        ])

        image = transform(image)
        GT = transform(GT)

        image = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)

        return image, GT

    def __len__(self):
        return len(self.img_paths)


def get_loader(image_path, image_size, batch_size, mode, num_workers=2, augmentation_prob=0.4):
    dataset = ImageFolder(image_path, image_size, mode, augmentation_prob)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=(mode=="train"), num_workers=num_workers)
