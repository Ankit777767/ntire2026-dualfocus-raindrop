import torch
import torch.nn as nn
from utils.color import rgb_to_y

class YCharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # pred, target: [B,3,H,W]
        loss = 0.0
        for p, t in zip(pred, target):
            y_p = rgb_to_y(p)
            y_t = rgb_to_y(t)
            diff = y_p - y_t
            loss += torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss / pred.size(0)
