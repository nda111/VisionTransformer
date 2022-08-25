import sys, os
if __name__ == '__main__':
    sys.path.append(os.path.abspath(f'{__file__}/../../'))

from typing import Union, Tuple

import torch
import torch.nn as nn

from model.module import Patcher, Encoder

class VisionTransformer(nn.Module):
    def __init__(self, 
                num_classes: int, 
                image_size: Union[int, Tuple[int, int], torch.IntTensor], num_channels: int,
                patch_size: int, hidden_dim: int, num_heads: int, depth: int, drop_rate: float=0):
        super(VisionTransformer, self).__init__()

        self.num_classes = num_classes
        self.image_size = image_size
        self.patch_size = patch_size

        self.latent_dim = patch_size * patch_size * num_channels
        self.hidden_dim = hidden_dim

        self.num_heads = num_heads
        self.depth = depth
        self.dropout_p = drop_rate

        self.patcher = Patcher(image_size=image_size, patch_size=patch_size)
        self.flattener = nn.Flatten(start_dim=2)

        self.class_token = torch.zeros(1, 1, self.latent_dim, requires_grad=True)
        num_patches = self.patcher.num_patches[0] * self.patcher.num_patches[1]
        self.pos_embed = torch.zeros(1, num_patches + 1, self.latent_dim, requires_grad=True)
        self.pos_dropout = nn.Dropout(p=drop_rate)

        self.encoders = nn.Sequential(*[
            Encoder(latent_dim=self.latent_dim, 
                    hidden_dim=self.hidden_dim, 
                    num_heads=self.num_heads, 
                    dropout_p=self.dropout_p) for _ in range(self.depth)
        ])
        self.norm = nn.LayerNorm(torch.tensor([self.latent_dim]), eps=1.0E-8)
        self.classifier = nn.Linear(self.latent_dim, self.num_classes)

    def forward(self, x):
        class_token = self.class_token.expand(x.size(0), 1, self.latent_dim).to(x.device)

        x = self.flattener(self.patcher(x))
        x = torch.cat([class_token, x], dim=1) + self.pos_embed.to(x.device)
        x = self.pos_dropout(x)

        x = self.encoders(x)
        x = self.norm(x)
        x = x[:, 0] # extract class tokens
        out = self.classifier(x)

        return out


def test():
    num_channels, image_size = 3, torch.tensor([384, 384])
    patch_size = torch.tensor(16)

    sample = torch.randn(4, num_channels, image_size[0], image_size[1])
    latent_dim = patch_size * patch_size * num_channels
    hidden_dim = latent_dim * 4

    model = VisionTransformer(
        num_classes=1000, image_size=image_size, num_channels=num_channels, 
        patch_size=patch_size, hidden_dim=hidden_dim, 
        num_heads=12, depth=12)
    out = model(sample)
    assert out.shape == (4, 1000)
    print('PASSED: model/module/Encoder.py')


if __name__ == '__main__':
    test()
