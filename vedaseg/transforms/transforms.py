import random

import cv2
import torch
import numpy as np
import albumentations as albu
from albumentations import DualTransform
import albumentations.augmentations.functional as F

from .registry import TRANSFORMS


@TRANSFORMS.register_module
class FactorScale(DualTransform):
    def __init__(self, scale=1.0, interpolation=cv2.INTER_LINEAR,
                 always_apply=False,
                 p=1.0):
        super(FactorScale, self).__init__(always_apply, p)
        self.scale = scale
        self.interpolation = interpolation

    def apply(self, image, scale=1.0, **params):
        return F.scale(image, scale, interpolation=self.interpolation)

    def apply_to_mask(self, image, scale=1.0, **params):
        return F.scale(image, scale, interpolation=cv2.INTER_NEAREST)

    def get_params(self):
        return {'scale': self.scale}

    def get_transform_init_args_names(self):
        return ('scale',)


@TRANSFORMS.register_module
class LongestMaxSize(FactorScale):
    def __init__(self, h_max, w_max, interpolation=cv2.INTER_LINEAR,
                 always_apply=False, p=1.0):
        self.h_max = h_max
        self.w_max = w_max
        super(LongestMaxSize, self).__init__(interpolation=interpolation,
                                             always_apply=always_apply,
                                             p=p)

    def update_params(self, params, **kwargs):
        params = super(LongestMaxSize, self).update_params(params, **kwargs)
        rows = params['rows']
        cols = params['cols']

        scale_h = self.h_max / rows
        scale_w = self.w_max / cols
        scale = min(scale_h, scale_w)

        params.update({'scale': scale})
        return params

    def get_transform_init_args_names(self):
        return ('h_max', 'w_max',)


@TRANSFORMS.register_module
class RandomScale(FactorScale):
    def __init__(self, scale_limit=(0.5, 2), interpolation=cv2.INTER_LINEAR,
                 scale_step=None, always_apply=False, p=1.0):
        super(RandomScale, self).__init__(interpolation=interpolation,
                                          always_apply=always_apply,
                                          p=p)
        self.scale_limit = albu.to_tuple(scale_limit)
        self.scale_step = scale_step

    def get_params(self):
        if self.scale_step:
            num_steps = int((self.scale_limit[1] - self.scale_limit[
                0]) / self.scale_step + 1)
            scale_factors = np.linspace(self.scale_limit[0],
                                        self.scale_limit[1], num_steps)
            scale_factor = np.random.choice(scale_factors).item()
        else:
            scale_factor = random.uniform(self.scale_limit[0],
                                          self.scale_limit[1])

        return {'scale': scale_factor}

    def get_transform_init_args_names(self):
        return ('scale_limit', 'scale_step',)


@TRANSFORMS.register_module
class PadIfNeeded(albu.PadIfNeeded):
    def __init__(self, min_height, min_width, border_mode=cv2.BORDER_CONSTANT,
                 value=None, mask_value=None):
        super(PadIfNeeded, self).__init__(min_height=min_height,
                                          min_width=min_width,
                                          border_mode=border_mode,
                                          value=value,
                                          mask_value=mask_value)

    def update_params(self, params, **kwargs):
        params = super(PadIfNeeded, self).update_params(params, **kwargs)
        rows = params['rows']
        cols = params['cols']

        if rows < self.min_height:
            h_pad_bottom = self.min_height - rows
        else:
            h_pad_bottom = 0

        if cols < self.min_width:
            w_pad_right = self.min_width - cols
        else:
            w_pad_right = 0

        params.update({'pad_top': 0,
                       'pad_bottom': h_pad_bottom,
                       'pad_left': 0,
                       'pad_right': w_pad_right})
        return params

    def get_transform_init_args_names(self):
        return ('min_height', 'min_width',)


@TRANSFORMS.register_module
class ToTensor(DualTransform):
    def __init__(self):
        super(ToTensor, self).__init__(always_apply=True)

    def apply(self, image, **params):
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = image[:, :, None]
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1)
        else:
            raise TypeError('img shoud be np.ndarray. Got {}'
                            .format(type(image)))
        return image

    def apply_to_mask(self, image, **params):
        image = torch.from_numpy(image)
        return image

    def apply_to_masks(self, masks, **params):
        masks = [self.apply_to_mask(mask, **params) for mask in masks]
        return torch.stack(masks, dim=0).squeeze()

    def get_transform_init_args_names(self):
        return ()
