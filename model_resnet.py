import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channel=64):
        super().__init__()
        self.channel = channel
        self.conv1 = nn.Conv2d(channel, channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(channel, channel, 3, 1, 1)
        
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = F.silu(x)
        x = self.conv2(x)
        x = x + res
        x = F.silu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, channel=64, n_block=8):
        super().__init__()
        self.channel = channel
        self.n_block = n_block
        self.head = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 1, 1), nn.SiLU(),
        )
        self.middle_part = nn.ModuleList([
            ResBlock(channel=channel) for _ in range(n_block)
        ])
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


