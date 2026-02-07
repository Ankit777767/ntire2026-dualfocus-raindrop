import torch
import torch.nn as nn
from models.restormer_blocks import TransformerBlock

class Model(nn.Module):
    def __init__(self, dim=48, num_blocks=6):
        super().__init__()

        self.embedding = nn.Conv2d(6, dim, 3, padding=1)

        self.blocks = nn.Sequential(
            *[TransformerBlock(dim) for _ in range(num_blocks)]
        )

        self.output = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.output(x)
        return x
