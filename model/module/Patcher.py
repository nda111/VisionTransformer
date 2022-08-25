from typing import Union, Tuple

import torch
import torch.nn as nn

class Patcher(nn.Module):
    def __init__(self, image_size: Union[int, Tuple[int, int], torch.IntTensor], patch_size: int):
        super(Patcher, self).__init__()

        size_type = type(image_size)
        if size_type is int:
            image_size = torch.tensor([image_size, image_size])
        elif size_type is tuple:
            image_size = torch.tensor(image_size)
        
        if image_size.shape != (2,):
            raise RuntimeError('The size of an image must be one of an integer, a 2-tuple or an integer tensor.')
        elif torch.any(image_size <= 0):
            raise RuntimeError('The image size must be positive.')
        self.image_size = image_size

        if torch.any(image_size % patch_size != 0):
            raise RuntimeError('The image size must be divisible by the patch size.')
        self.patch_size = patch_size
        self.num_patches = (image_size / patch_size).int()
        

    def forward(self, img):
        """    
        INPUT: tensor of size (N, C, W, H) or (C, W, H)
        OUTPUT: tensor of size (N, M, C, P, P) or(M, C, P, P) where M = HW / P^2

        :param img: The image to split into patches.
        :return: Splitted patches from the input image.
        """
        if img.ndim == 3:
            _, width, height = img.shape
            img = img.unsqueeze(0)
            squeeze_later = True
        elif img.ndim == 4:
            _, _, width, height = img.shape
            squeeze_later = False
        else:
            raise RuntimeError('Please follow the input shape.')
            
        width, height = self.image_size
        x_indices = torch.linspace(0, width - self.patch_size, self.num_patches[0]).long()
        y_indices = torch.linspace(0, height - self.patch_size, self.num_patches[1]).long()
        
        patches = []
        for x in x_indices:
            for y in y_indices:
                patch = img[:, :, x:x + self.patch_size, y:y + self.patch_size]
                patches.append(patch.unsqueeze(1))
        patches = torch.cat(patches, dim=1)
        if squeeze_later:
            patches = patches.squeeze()
        return patches


def test():
    image_size = 384
    sample = torch.randn(2, 3, image_size, image_size)
    patcher = Patcher(image_size=(image_size, image_size), patch_size=16)
    out = patcher(sample)

    assert out.shape == (2, 576, 3, 16, 16)
    print('PASSED: dataset/utils/Patcher.py')


if __name__ == '__main__':
    test()
