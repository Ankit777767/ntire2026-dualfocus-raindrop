import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets.codabench_dataset import CodabenchDataset
from models.backbone import Model

def main():
    # -------- paths --------
    input_dir = os.path.join("data", "codabench")
    output_dir = "results"
    checkpoint = os.path.join("checkpoints", "epoch_5.pth")

    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- model --------
    model = Model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # -------- dataset --------
    dataset = CodabenchDataset(input_dir)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    # -------- inference --------
    with torch.no_grad():
        for inp, name in loader:
            inp = inp.to(device)
            pred = model(inp).clamp(0, 1)

            save_image(
                pred,
                os.path.join(output_dir, name[0])
            )

    print("Inference completed. Results saved in:", output_dir)

if __name__ == "__main__":
    main()
