import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class DualFocusDataset(Dataset):
    def __init__(self, root, mode="train", single_prob=0.15, transform=None):
        """
        root/
          daytime/
            drop/00001/*.png
            blur/00001/*.png
            clear/00001/*.png
          nighttime/
            same structure
        """
        self.root = root
        self.mode = mode
        self.single_prob = single_prob
        self.transform = transform

        self.samples = []
        for time in ["daytime", "nighttime"]:
            base = os.path.join(root, time)
            if not os.path.exists(base):
                continue

            drop_root = os.path.join(base, "Drop")
            for scene in sorted(os.listdir(drop_root)):
                drop_scene = os.path.join(base, "Drop", scene)
                blur_scene = os.path.join(base, "Blur", scene)
                clear_scene = os.path.join(base, "Clear", scene)

                for name in sorted(os.listdir(drop_scene)):
                    self.samples.append({
                        "Drop": os.path.join(drop_scene, name),
                        "Blur": os.path.join(blur_scene, name),
                        "Clear": os.path.join(clear_scene, name),
                        "is_night": time == "nighttime"
                    })

    def __len__(self):
        return len(self.samples)

    def _read(self, path):
        img = Image.open(path).convert("RGB")
        return TF.to_tensor(img)

    def __getitem__(self, idx):
        s = self.samples[idx]

        drop = self._read(s["Drop"])
        blur = self._read(s["Blur"])
        clear = self._read(s["Clear"])

        # fake single-image simulation
        if self.mode == "train" and random.random() < self.single_prob:
            if random.random() < 0.5:
                blur = drop.clone()
            else:
                drop = blur.clone()

        if self.transform:
            drop, blur, clear = self.transform(drop, blur, clear)

        inp = torch.cat([drop, blur], dim=0)
        return inp, clear
