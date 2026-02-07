import torch
from utils.color import rgb_to_y

def psnr_y(pred, gt):
    y_pred = rgb_to_y(pred)
    y_gt = rgb_to_y(gt)
    mse = torch.mean((y_pred - y_gt) ** 2)
    return 10 * torch.log10(1.0 / mse)
