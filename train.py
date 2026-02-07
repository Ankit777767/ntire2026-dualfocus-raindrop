import os
import torch
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
    epochs = 5
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
    for epoch in range(epochs):
        epoch_loss = 0.0

        for i, (inp, target) in enumerate(train_loader):
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

            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"Iter [{i+1}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")

        # save checkpoint
        torch.save(
            model.state_dict(),
            os.path.join(save_dir, f"epoch_{epoch+1}.pth")
        )

if __name__ == "__main__":
    main()
