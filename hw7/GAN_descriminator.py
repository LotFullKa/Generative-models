import torch
import torch.nn as nn

class ConvDescriminator(nn.Module):
    def __init__(self, input_shape, output_shape = 1):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = output_shape
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
        )
        conv_out_dim = input_shape[1] // 4 * input_shape[2] // 4 * 64
        self.fc = nn.Linear(conv_out_dim, self.latent_dim)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = (x.float() - 0.5) * 2
        out = self.convs(x)
        out = out.view(out.shape[0], -1)
        logits = self.fc(out)
        probs = self.sigm(logits)
    
        return probs.type(torch.FloatTensor)