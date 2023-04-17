import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional
# global vars
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

class ResNetUpConvBlock(nn.Module):
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
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride = 1, padding = 1),
            )

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, 
                               out_channels=in_channels*2,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding),
            nn.BatchNorm2d(num_features=in_channels*2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=in_channels*2, 
                               out_channels=in_channels*3,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=in_channels*3, 
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
        
class Generator(nn.Module):
    def __init__(self,
                 noise_dim,
                 out_shape,
                 hidden_channels:int = 32,
                 block_count:int = 5) -> None:
        super().__init__()

        self.noise = nn.Parameter(torch.Tensor(hidden_channels, out_shape[1], out_shape[2]), requires_grad=True).to(device)

        self.noise_dim = noise_dim
        self.base_size = out_shape
        self.hidden_channels = hidden_channels

        self.linear = nn.Linear(noise_dim, hidden_channels * out_shape[1] * out_shape[2])


        self.blocks = []
        

        for i in range(block_count):
            if i % 2 == 0 and i:
                shortcut = nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels * 2,
                        kernel_size=1,
                    ),
                    nn.BatchNorm2d(num_features=hidden_channels*2)
                )
                block = ResNetUpConvBlock(
                    hidden_channels,
                    hidden_channels * 2,
                    shortcut=shortcut,
                )
                hidden_channels *= 2
            else:
                block = ResNetUpConvBlock(hidden_channels, hidden_channels//2)
                hidden_channels //=2

            self.blocks.append(block)
        
        self.blocks = nn.ModuleList(self.blocks)
        
        # now shape is (hidden_channels, output_shape[1], output_shape[2])
        self.up_conv = nn.ConvTranspose2d(
            in_channels=hidden_channels,
            out_channels=self.base_size[0],
            kernel_size=1
        )
        
        

    def forward(self, z, noise:bool=False):
        out = self.linear(z)
        out = out.reshape(-1, self.hidden_channels, self.base_size[1], self.base_size[2])
        
        if noise:
            out += self.noise

        for block in self.blocks:
            out = block(out)

        out = self.up_conv(out)
        out = F.sigmoid(out)

        return out