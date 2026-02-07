import torch
import lpips

class LPIPSMetric:
    def __init__(self, device):
        self.fn = lpips.LPIPS(net='alex').to(device)
        self.fn.eval()

    @torch.no_grad()
    def __call__(self, pred, gt):
        # pred, gt: [B,3,H,W] in [0,1]
        pred = pred * 2 - 1
        gt = gt * 2 - 1
        val = self.fn(pred, gt)
        return val.mean()
