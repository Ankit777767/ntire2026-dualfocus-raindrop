import os
import time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from datasets.dual_focus_dataset import DualFocusDataset
from models.backbone import Model
from losses.charbonnier import CharbonnierLoss
from losses.y_charbonnier import YCharbonnierLoss
from losses.ssim import SSIMLoss

from metrics.psnr_y import psnr_y
from metrics.ssim_y import ssim_y
from metrics.lpips_metric import LPIPSMetric


def main():
    # -------- config --------
    train_root = os.path.join("data", "train")
    val_root = os.path.join("data", "val")

    batch_size = 4
    epochs = 10
    lr = 1e-4
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- dataset --------
    train_set = DualFocusDataset(
        root=train_root,
        mode="train",
        single_prob=0.15
    )

    val_set = DualFocusDataset(
        root=val_root,
        mode="val",
        single_prob=0.0
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    # -------- model --------
    model = Model()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model = model.to(device)

    # -------- loss & optimizer --------
    loss_rgb = CharbonnierLoss()
    loss_y = YCharbonnierLoss()
    loss_ssim = SSIMLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # -------- tracking --------
    train_losses = []
    val_scores = []
    val_psnrs = []
    val_ssims = []
    val_lpips = []

    best_score = -1
    best_epoch = -1

    start_time = time.time()
    num_batches = len(train_loader)

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for i, (inp, target) in enumerate(train_loader):

            inp = inp.to(device)
            target = target.to(device)

            pred = model(inp)

            loss_rgb_val = loss_rgb(pred, target)
            loss_y_val = loss_y(pred, target)
            loss_ssim_val = loss_ssim(pred, target)

            loss = (
                loss_rgb_val +
                0.5 * loss_y_val +
                0.2 * loss_ssim_val
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # -------- ETA --------
            elapsed = time.time() - start_time
            completed_iters = epoch * num_batches + (i + 1)
            total_iters = epochs * num_batches
            iters_left = total_iters - completed_iters
            avg_time = elapsed / completed_iters
            eta_seconds = iters_left * avg_time

            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Iter [{i+1}/{num_batches}] | "
                    f"RGB: {loss_rgb_val.item():.4f} | "
                    f"Y: {loss_y_val.item():.4f} | "
                    f"SSIM: {loss_ssim_val.item():.4f} | "
                    f"Total: {loss.item():.4f} | "
                    f"ETA: {eta_seconds/60:.2f} min"
                )

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        epoch_time = time.time() - epoch_start

        print(
            f"\nEpoch {epoch+1} finished | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Epoch Time: {epoch_time/60:.2f} min"
        )

        # =========================
        # VALIDATION
        # =========================
        model.eval()
        lpips_fn = LPIPSMetric(device)

        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        count = 0

        with torch.no_grad():
            for inp, gt in val_loader:
                inp = inp.to(device)
                gt = gt.to(device)

                pred = model(inp).clamp(0, 1)

                total_psnr += psnr_y(pred, gt).item()
                total_ssim += ssim_y(pred, gt).item()
                total_lpips += lpips_fn(pred, gt).item()
                count += 1

        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_lpips = total_lpips / count

        score = avg_psnr + 10 * avg_ssim - 5 * avg_lpips

        print(f"\nValidation Results - Epoch {epoch+1}")
        print(f"PSNR(Y): {avg_psnr:.4f}")
        print(f"SSIM(Y): {avg_ssim:.4f}")
        print(f"LPIPS  : {avg_lpips:.4f}")
        print(f"NTIRE Score: {score:.4f}\n")

        val_scores.append(score)
        val_psnrs.append(avg_psnr)
        val_ssims.append(avg_ssim)
        val_lpips.append(avg_lpips)

        # -------- Save Best Model --------
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), os.path.join(save_dir, "best_model.pth"))
            else:
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"🔥 New Best Model Saved at Epoch {best_epoch}")

        model.train()

    # =========================
    # FINAL SUMMARY
    # =========================
    print(f"\nTraining Completed.")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best NTIRE Score: {best_score:.4f}")

    # -------- Plot Curves --------
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(val_scores)
    plt.title("Validation NTIRE Score")
    plt.xlabel("Epoch")
    plt.ylabel("Score")

    plt.subplot(1, 2, 2)
    plt.plot(val_psnrs, label="PSNR")
    plt.plot(val_ssims, label="SSIM")
    plt.plot(val_lpips, label="LPIPS")
    plt.legend()
    plt.title("Validation Metrics")

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()


if __name__ == "__main__":
    main()