import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from datasets.codabench_dataset import CodabenchDataset
from models.backbone import Model

def main():
    # -------- paths --------
    input_dir = os.path.join("data", "codabench")
    output_dir = "results"
    checkpoint = os.path.join("checkpoints", "epoch_2.pth")

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

    total_images = len(loader)
    start_time = time.time()
    total_runtime = 0.0

    # -------- inference --------
    with torch.no_grad():
        for idx, (inp, name) in enumerate(loader):
            iter_start = time.time()

            inp = inp.to(device)
            pred = model(inp).clamp(0, 1)

            save_image(
                pred,
                os.path.join(output_dir, name[0])
            )

            # -------- timing --------
            iter_time = time.time() - iter_start
            total_runtime += iter_time

            avg_time = total_runtime / (idx + 1)
            remaining = total_images - (idx + 1)
            eta = remaining * avg_time

            print(
                f"[{idx+1}/{total_images}] "
                f"Time/img: {iter_time:.4f}s | "
                f"Avg: {avg_time:.4f}s | "
                f"ETA: {eta/60:.2f} min"
            )

    total_time = time.time() - start_time

    print("\nInference completed.")
    print(f"Total images       : {total_images}")
    print(f"Total time         : {total_time/60:.2f} min")
    print(f"Average time/image : {total_runtime/total_images:.4f} sec")
    print("Results saved in   :", output_dir)

if __name__ == "__main__":
    main()