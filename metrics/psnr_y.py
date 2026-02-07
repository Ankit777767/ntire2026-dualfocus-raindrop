import torch
from utils.color import rgb_to_y

def psnr_y(pred, gt, eps=1e-8):
    # pred, gt: [B,3,H,W]
    psnrs = []
    for p, g in zip(pred, gt):
        y_p = rgb_to_y(p)
        y_g = rgb_to_y(g)
        mse = torch.mean((y_p - y_g) ** 2)
        psnr = 10 * torch.log10(1.0 / (mse + eps))
        psnrs.append(psnr)
    return torch.stack(psnrs).mean()
