import sys, os
if __name__ == '__main__':
    sys.path.append(os.path.abspath(f'{__file__}/../../../'))

import torch
import torch.nn as nn

from model.module import Patcher, SelfAttention


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, num_heads: int, dropout_p: float=0):
        super(Encoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        
        latent_dim = torch.tensor([latent_dim])
        self.norm = nn.LayerNorm(latent_dim, eps=1.0E-8)
        self.self_attention = SelfAttention(
            latent_dim,
            num_heads=num_heads,
            bias=True,
            kv_bias=True)
        self.block = nn.Sequential(
            nn.LayerNorm(latent_dim, eps=1.0E-8),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, x):
        feat1 = self.norm(x)
        feat1 = self.self_attention(x)

        feat2 = feat1 + x
        feat2 = self.block(feat2)

        out = feat2 + x
        return out


def test():
    num_channels, image_size = 3, torch.tensor([384, 384])
    patch_size = torch.tensor(16)

    patcher = Patcher(image_size=image_size, patch_size=patch_size)
    flattener = nn.Flatten(start_dim=2)

    sample = torch.randn(4, num_channels, image_size[0], image_size[1])
    latent_dim = patch_size * patch_size * num_channels
    hidden_dim = latent_dim * 4

    embeded_sample = flattener(patcher((sample)))
    encoder = Encoder(latent_dim=latent_dim, hidden_dim=hidden_dim, num_heads=12)

    out = encoder(embeded_sample)
    assert out.shape == embeded_sample.shape
    print('PASSED: model/module/Encoder.py')


if __name__ == '__main__':
    test()
