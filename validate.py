import os
import torch
from torch.utils.data import DataLoader
import time
from datasets.dual_focus_dataset import DualFocusDataset
from models.backbone import Model
from metrics.psnr_y import psnr_y
from metrics.ssim_y import ssim_y
from metrics.lpips_metric import LPIPSMetric

def main():
    # -------- config --------
    data_root = os.path.join("data", "val")
    checkpoint = os.path.join("checkpoints", "epoch_5.pth")
    batch_size = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- dataset --------
    val_set = DualFocusDataset(
        root=data_root,
        mode="val",
        single_prob=0.0
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    # -------- model --------
    model = Model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # -------- metrics --------
    lpips_fn = LPIPSMetric(device)

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    count = 0
    total_time = 0.0

    with torch.no_grad():
        for idx, (inp, gt) in enumerate(val_loader):
            start = time.time()

            inp = inp.to(device)
            gt = gt.to(device)

            pred = model(inp).clamp(0, 1)

            total_psnr += psnr_y(pred, gt).item()
            total_ssim += ssim_y(pred, gt).item()
            total_lpips += lpips_fn(pred, gt).item()
            count += 1

            elapsed = time.time() - start
            total_time += elapsed

            if (idx + 1) % 50 == 0:
                print(
                    f"Validated {idx+1}/{len(val_loader)} | "
                    f"Time per image: {elapsed:.4f} sec"
                )


    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_lpips = total_lpips / count

    score = avg_psnr + 10 * avg_ssim - 5 * avg_lpips
    avg_time = total_time / count
    print(f"Average runtime per image: {avg_time:.4f} sec")

    print("Validation Results")
    print(f"PSNR (Y): {avg_psnr:.4f}")
    print(f"SSIM (Y): {avg_ssim:.4f}")
    print(f"LPIPS   : {avg_lpips:.4f}")
    print(f"NTIRE Score: {score:.4f}")

if __name__ == "__main__":
    main()