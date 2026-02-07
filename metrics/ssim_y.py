import torch
from utils.color import rgb_to_y

def ssim_y(pred, gt):
    ssims = []
    for p, g in zip(pred, gt):
        y1 = rgb_to_y(p)
        y2 = rgb_to_y(g)

        mu1 = y1.mean()
        mu2 = y2.mean()
        sigma1 = ((y1 - mu1) ** 2).mean()
        sigma2 = ((y2 - mu2) ** 2).mean()
        sigma12 = ((y1 - mu1) * (y2 - mu2)).mean()

        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1)*(sigma1 + sigma2 + C2))
        ssims.append(ssim)

    return torch.stack(ssims).mean()
