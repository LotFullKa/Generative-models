import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size=5):
        assert mask_type in ['A', 'B']
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(mask_type)

    def forward(self, input):
        # ====
        # your code
        # with torch.no_grad():
        self.weight.data *= self.mask
        return super().forward(input)
        # ====

    def create_mask(self, mask_type):
        # ====
        # your code
        # do not forget about mask_type
        
        assert self.kernel_size[0] % 2 == 1

        center = self.kernel_size[0] // 2
        self.mask[:, :, :center, :] = 1
        self.mask[:, :, :center+1, :center] = 1

        if mask_type == "B":
            self.mask[:, :, center, center] = 1
        # ====


class LayerNorm(nn.LayerNorm):
    def __init__(self, n_filters):
        super().__init__(n_filters)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = super().forward(x)
        return x.permute(0, 3, 1, 2).contiguous()


class PixelCNN(nn.Module):
    def __init__(
        self, 
        input_shape, 
        n_filters=64, 
        kernel_size=7, 
        n_layers=5, 
        use_layer_norm=True
    ):
      
        super().__init__()
        self.input_shape = input_shape
        # ====
        # your code
        # apply the sequence of MaskedConv2d -> LayerNorm (it is optional) -> ReLU
        # the last layer should be MaskedConv2d (not ReLU)
        # Note 1: the first conv layer should be of type 'A'
        # Note 2: final output_dim in MaskedConv2d must be 2

        if use_layer_norm:
          block = nn.Sequential(
              MaskedConv2d('B', n_filters, n_filters, kernel_size=kernel_size),
              LayerNorm(n_filters),
              nn.ReLU(),
          )
        else:
          block = nn.Sequential(
              MaskedConv2d('B', n_filters, n_filters, kernel_size=kernel_size),
              nn.ReLU(),
          )
        
        packed_blocks = [block] * (n_layers-3)
        self.net = nn.Sequential(
            MaskedConv2d('A', 3, n_filters // 2, kernel_size=kernel_size),
            nn.ReLU(),
            MaskedConv2d('B', n_filters // 2, n_filters, kernel_size=kernel_size),
            *packed_blocks,
            MaskedConv2d('B', n_filters, n_filters // 2, kernel_size=kernel_size),
            nn.ReLU(), 
            MaskedConv2d('B', n_filters // 2, 2, kernel_size=kernel_size)
        )
        
        # ====

    def forward(self, x):
        # read the forward method carefully
        batch_size = x.shape[0]
        out = (x.float() - 0.5) / 0.5
        out = self.net(out)
        return out.view(batch_size, 2, 1, *self.input_shape)

    def loss(self, x):
        # ====
        # your code
        x = x.long()
        total_loss = F.cross_entropy(self.forward(x), x)
        # ====
        return {'total_loss': total_loss}

    def sample(self, n):
        # read carefully the sampling process
        samples = torch.zeros(n, 3, *self.input_shape).cuda()
        with torch.no_grad():
            for r in range(self.input_shape[0]):
                for c in range(self.input_shape[1]):
                    logits = self(samples)[:, :, :, r, c]
                    probs = F.softmax(logits, dim=1).squeeze(-1)
                    samples[:, 0, r, c] = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return samples
    
