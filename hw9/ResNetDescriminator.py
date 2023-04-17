import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional


class ResNetConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 shortcut: Optional[Callable] = None,
        ) -> None:
        super().__init__()

        if shortcut:
            self.shortcut = shortcut
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride = 1, padding = 1),
            )

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                               out_channels=in_channels*2,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding),
            nn.BatchNorm2d(num_features=in_channels*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels*2, 
                               out_channels=in_channels*3,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels*3, 
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding),
            nn.ReLU(),
        )


    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        out = self.block(x)
        
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape, hidden_channel:int = 8, block_count: int = 4):
        super().__init__()
        self.input_shape = input_shape
        self.hidden_channel = hidden_channel

        self.blocks = []


        prev_channel_num = input_shape[0]
        for _ in range(block_count):
            block = ResNetConvBlock(prev_channel_num, self.hidden_channel)
            prev_channel_num = self.hidden_channel
            
            self.hidden_channel = min(128, self.hidden_channel*2)

            self.blocks.append(block)

        self.blocks = nn.ModuleList(self.blocks)

        self.linear = nn.Linear(
            in_features=self.hidden_channel * input_shape[1] * input_shape[2], 
            out_features=1,
            )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x
        for block in self.blocks:
            print(out.isnan().any())
            out = block(out)

        out = out.reshape(-1, self.linear.in_features)
        print(out)
        out = self.linear(out)

        out = self.sigmoid(out)

        return out
