import torch
import torch.nn as nn

class PlainCNN(nn.Module):
    def __init__(self, layer=12):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.SiLU(),
        )
        layers = []
        for i in range(layer-4):
            layers.append(nn.Conv2d(64, 64, 3, 1, 1))
            layers.append(nn.SiLU())
        self.middle_part = nn.ModuleList(layers)
        self.tail = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, x):
        # x :: (B, 3, 96, 96)
        res = x
        x = self.head(x)
        for b in self.middle_part:
            x = b(x)
        x = self.tail(x)
        x = x + res
        return x


