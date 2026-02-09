import os
import torch
import time
from torch.utils.data import DataLoader
from datasets.dual_focus_dataset import DualFocusDataset
from models.backbone import Model
from losses.charbonnier import CharbonnierLoss
from losses.y_charbonnier import YCharbonnierLoss
from losses.ssim import SSIMLoss

def main():
    # -------- config --------
    data_root = os.path.join("data", "train")
    batch_size = 4
    epochs = 10
    lr = 1e-4
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- dataset --------
    train_set = DualFocusDataset(
        root=data_root,
        mode="train",
        single_prob=0.15
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # -------- model --------
    model = Model().to(device)

    # -------- loss & optimizer --------
    loss_rgb = CharbonnierLoss()
    loss_y = YCharbonnierLoss()
    loss_ssim = SSIMLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------- training loop --------
    model.train()
    start_time = time.time()

    num_batches = len(train_loader)

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0

        for i, (inp, target) in enumerate(train_loader):
            iter_start = time.time()

            inp = inp.to(device)
            target = target.to(device)

            pred = model(inp)
            loss = (
                loss_rgb(pred, target) +
                0.5 * loss_y(pred, target) +
                0.2 * loss_ssim(pred, target)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # -------- ETA calculation --------
            elapsed = time.time() - start_time
            completed_iters = epoch * num_batches + (i + 1)
            total_iters = epochs * num_batches

            iters_left = total_iters - completed_iters
            avg_time_per_iter = elapsed / completed_iters
            eta_seconds = iters_left * avg_time_per_iter

            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Iter [{i+1}/{num_batches}] "
                    f"Loss: {loss.item():.4f} | "
                    f"ETA: {eta_seconds/60:.1f} min"
                )

        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch+1} finished | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Epoch Time: {epoch_time/60:.2f} min"
        )

        # save checkpoint
        torch.save(
            model.state_dict(),
            os.path.join(save_dir, f"epoch_{epoch+1}.pth")
        )

if __name__ == "__main__":
    main()