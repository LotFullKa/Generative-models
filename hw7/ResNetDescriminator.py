import torch
import torch.nn as nn
from typing import Callable, Optional

class ResNetConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reshaper: Optional[Callable] = None,
    ):
        super(ResNetConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.conv3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)

        self.reshaper = reshaper

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += x if self.reshaper is None else self.reshaper(x)
        out = self.relu(out)

        return out

class Discriminator(nn.Module):
    def __init__(self, input_shape, block_count: int = 4):
        super().__init__()
        self.input_shape = input_shape

        channels_num = 32
        self.blocks = []

        for i in range(block_count):
            if i % 2 == 0:
                channels_num *= 2

                reshaper = nn.Sequential(
                    nn.Conv2d(
                        in_channels=channels_num // 2 if i else self.input_shape[0],
                        out_channels=channels_num,
                        kernel_size=1,
                        stride=2,
                    ),
                    nn.BatchNorm2d(num_features=channels_num)
                )
                block = ResNetConvBlock(
                    channels_num // 2 if i else self.input_shape[0],
                    channels_num,
                    stride=2,
                    reshaper=reshaper,
                )
            else:
                block = ResNetConvBlock(channels_num, channels_num)

            self.blocks.append(block)

        self.blocks = nn.ModuleList(self.blocks)

        self.linear = nn.Linear(
            in_features=self.input_shape[1] * self.input_shape[2] * 2 ** (5 - block_count // 2), 
            out_features=1,
            )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x

        for i, block in enumerate(self.blocks):
            out = block(out)

        out = out.reshape(-1, self.linear.in_features)
        out = self.linear(out)

        out = self.sigmoid(out)

        return out