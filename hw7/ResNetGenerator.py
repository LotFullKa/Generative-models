import torch
import torch.nn as nn
from typing import Callable, Optional


class ResNetUpConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        reshaper: Optional[Callable] = None,
    ):
        super().__init__()

        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.ConvTranspose2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.conv3 = nn.ConvTranspose2d(
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


class Generator(nn.Module):
    def __init__(self, noise_dim, output_shape, block_count: int = 10):
        super().__init__()
        self.noise_dim = noise_dim
        self.output_shape = output_shape

        self.base_size = (int(64 * 2 ** (block_count // 2 - 1)), output_shape[1] // 2 ** (
            block_count // 2), output_shape[2] // 2 ** (block_count // 2))

        self.linear = nn.Linear(in_features=noise_dim, out_features=64 *
                                output_shape[1] * output_shape[2] // 2 ** (block_count // 2 + 1))

        channels_num = 64 * 2 ** (block_count // 2 - 1)
        self.blocks = []

        for i in range(block_count):
            if i % 2 == 0 and i:
                channels_num //= 2

                reshaper = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=channels_num * 2,
                        out_channels=channels_num,
                        kernel_size=2,
                        stride=2,
                    ),
                    nn.BatchNorm2d(num_features=channels_num)
                )
                block = ResNetUpConvBlock(
                    channels_num * 2,
                    channels_num,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    reshaper=reshaper,
                )
            else:
                block = ResNetUpConvBlock(channels_num, channels_num)

            self.blocks.append(block)

        self.blocks = nn.ModuleList(self.blocks)

        # now shape is (64, output_shape[0] // 2, output_shape[1] // 2)
        self.up_conv = nn.ConvTranspose2d(
            in_channels=channels_num,
            out_channels=self.output_shape[0],
            kernel_size=2,
            stride=2,
        )

    def forward(self, z):
        out = self.linear(z)
        out = out.reshape(-1, *self.base_size)

        for i, block in enumerate(self.blocks):
            out = block(out)

        out = self.up_conv(out)
        out = nn.functional.sigmoid(out)

        return out
