import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class CodabenchDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = sorted(os.listdir(root))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.root, name)).convert("RGB")
        img = TF.to_tensor(img)

        # duplicate image
        inp = torch.cat([img, img], dim=0)
        return inp, name
