import torch
from PIL import ImageFilter
import PIL
import random
import numpy as np
import albumentations
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# class TwoCropsTransform:
#     """Take two random crops of one image as the query and key."""
#
#     def __init__(self, base_transform):
#         self.base_transform = base_transform
#
#     def __call__(self, x, x_h=None, x_e=None):
#         q = self.base_transform(x)
#         k = self.base_transform(x)
#
#         return [q, k]

# class TwoCropsTransform:
#     """Take two random crops of one image as the query and key."""
#
#     def __init__(self, base_transform, is_eval=False):
#         self.base_transform = base_transform
#         self.is_eval = is_eval
#         self.album_transform = albumentations.RandomResizedCrop(224, 224, scale=(0.2, 1.))
#
#     def __call__(self, x_h, x_e):
#         image_q = self.album_transform(image=np.array(x_h), mask=np.array(x_e))
#         q_h = self.base_transform(PIL.Image.fromarray(image_q['image']))
#         q_e = self.base_transform(PIL.Image.fromarray(image_q['mask']))
#         if self.is_eval:
#             return [q_h, q_e]
#         image_k = self.album_transform(image=np.array(x_h), mask=np.array(x_e))
#         k_h = self.base_transform(PIL.Image.fromarray(image_k['image']))
#         k_e = self.base_transform(PIL.Image.fromarray(image_k['mask']))
#         return [q_h, q_e, k_h, k_e]

import torchvision.transforms.functional as TF

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, is_eval=False):
        self.base_transform = base_transform
        self.is_eval = is_eval
        if isinstance(self.base_transform, list):

            self.strong_transform = self.base_transform[0]
            self.weak_transform = self.base_transform[1]

            print(len(self.base_transform), len(self.strong_transform))

    def __call__(self, x=None, x_h=None, x_e=None):
        if x is not None:
            q = self.base_transform(image=x)['image']
            k = self.base_transform(image=x)['image']
            return [q, k]
        else:
            # x_h = PIL.Image.fromarray(x_h)
            # x_e = PIL.Image.fromarray(x_e)

            if self.is_eval:
                q_h = self.base_transform(image=x_h)['image']
                q_e = self.base_transform(image=x_e)['image']
                return [q_h, q_e]

            q_h = self.strong_transform(image=x_h)['image']
            q_e = self.weak_transform(image=x_e)['image']

            k_h = self.strong_transform(image=x_h)['image']
            k_e = self.weak_transform(image=x_e)['image']

            return [q_h, q_e, k_h, k_e]


def my_transforms(H_prime, E_prime):
    # transforms.RandomResizedCrop(224, scale=(0.2, 1.0))
    i, j, h, w = transforms.RandomResizedCrop.get_params(H_prime, scale=(0.2, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0))
    H_prime = TF.resized_crop(H_prime, i, j, h, w, (224, 224))
    E_prime = TF.resized_crop(E_prime, i, j, h, w, (224, 224))

    # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
    if torch.rand(1) <= 0.8:
        cj = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        H_prime = cj(H_prime)
        E_prime = cj(E_prime)

    # transforms.RandomGrayscale(p=0.2)
    if torch.rand(1) < 0.2:
        H_prime = TF.rgb_to_grayscale(H_prime, TF.get_image_num_channels(H_prime))
        E_prime = TF.rgb_to_grayscale(E_prime, TF.get_image_num_channels(E_prime))

    # transforms.RandomApply([pcl.loader.GaussianBlur([.1, 2.])], p=0.5)
    if torch.rand(1) <= 0.5:
        gb = GaussianBlur([.1, 2.])
        H_prime = gb(H_prime)
        E_prime = gb(H_prime)

    if torch.rand(1) < 0.5:
        H_prime = TF.hflip(H_prime)
        E_prime = TF.hflip(E_prime)

    H_prime = TF.to_tensor(H_prime)
    E_prime = TF.to_tensor(E_prime)

    return normalize(H_prime), normalize(E_prime)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    

class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)           
        return sample, index