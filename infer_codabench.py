import os
import torch
from torch.utils.data import DataLoader
from datasets.codabench_dataset import CodabenchDataset
from models.backbone import Model  # you will define later
from torchvision.utils import save_image

def main():
    model = Model().cuda().eval()
    model.load_state_dict(torch.load("ckpt.pth"))

    dataset = CodabenchDataset("data/codabench")
    loader = DataLoader(dataset, batch_size=1)

    os.makedirs("results", exist_ok=True)

    with torch.no_grad():
        for inp, name in loader:
            inp = inp.cuda()
            out = model(inp)
            save_image(out, os.path.join("results", name[0]))

if __name__ == "__main__":
    main()
