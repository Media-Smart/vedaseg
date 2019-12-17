import torch
import random
import numpy as np
import cv2
import torch.nn.functional as F


from .registry import TRANSFORMS


CV2_MODE = {
    'bilinear': cv2.INTER_LINEAR,
    'nearest': cv2.INTER_NEAREST,
    'cubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
}

CV2_BORDER_MODE = {
    'constant': cv2.BORDER_CONSTANT,
    'reflect': cv2.BORDER_REFLECT,
    'reflect101': cv2.BORDER_REFLECT101,
    'replicate': cv2.BORDER_REPLICATE,
}


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


@TRANSFORMS.register_module
class RandomScale(object):
    def __init__(self, min_scale, max_scale, scale_step=0.0, mode='bilinear'):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scale_step = scale_step
        self.mode = mode

    def get_scale_factor(self, min_scale, max_scale, scale_step):
        if min_scale == max_scale:
            return min_scale

        if scale_step == 0:
            return random.uniform(min_scale, max_scale)

        num_steps = int((max_scale - min_scale) / scale_step + 1)
        scale_factors = np.linspace(min_scale, max_scale, num_steps)
        scale_factor = np.random.choice(scale_factors).item()

        return scale_factor

    def __call__(self, image, mask):
        scale_factor = self.get_scale_factor(self.min_scale, self.max_scale, self.scale_step)

        if scale_factor == 1.0:
            return image, mask

        h, w, c = image.shape
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)

        torch_image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0)
        torch_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
        torch_image = F.interpolate(torch_image, size=(new_h, new_w), mode=self.mode, align_corners=True)
        torch_mask = F.interpolate(torch_mask, size=(new_h, new_w), mode='nearest')

        new_image = torch_image.squeeze().permute(1,2,0).numpy()
        new_mask = torch_mask.squeeze().numpy()

        return new_image, new_mask


@TRANSFORMS.register_module
class RandomCrop(object):
    def __init__(self, height, width, image_value, mask_value):
        self.height = height
        self.width = width
        self.image_value = image_value
        self.mask_value = mask_value

    def __call__(self, image, mask):
        h, w, c = image.shape
        target_height = h + max(self.height-h, 0)
        target_width = w + max(self.width-w, 0)

        image_pad_value = np.reshape(np.array(self.image_value, dtype=image.dtype), [1,1,3])
        mask_pad_value = np.reshape(np.array(self.mask_value, dtype=mask.dtype), [1,1])

        new_image = np.tile(image_pad_value, (target_height, target_width, 1))
        new_mask = np.tile(mask_pad_value, (target_height, target_width))

        new_image[:h, :w, :] = image
        new_mask[:h, :w] = mask

        assert np.count_nonzero(mask != self.mask_value) == np.count_nonzero(new_mask != self.mask_value)

        y1 = int(random.uniform(0, target_height-self.height+1))
        y2 = y1 + self.height
        x1 = int(random.uniform(0, target_width-self.width+1))
        x2 = x1 + self.width

        new_image = new_image[y1:y2, x1:x2, :]
        new_mask = new_mask[y1:y2, x1:x2]

        return new_image, new_mask


@TRANSFORMS.register_module
class PadIfNeeded(object):
    def __init__(self, height, width, image_value, mask_value):
        self.height = height
        self.width = width
        self.image_value = image_value
        self.mask_value = mask_value

    def __call__(self, image, mask):
        h, w, c = image.shape

        assert h <= self.height and w <= self.width

        target_height = h + max(self.height-h, 0)
        target_width = w + max(self.width-w, 0)

        image_pad_value = np.reshape(np.array(self.image_value, dtype=image.dtype), [1,1,3])
        mask_pad_value = np.reshape(np.array(self.mask_value, dtype=mask.dtype), [1,1])

        new_image = np.tile(image_pad_value, (target_height, target_width, 1))
        new_mask = np.tile(mask_pad_value, (target_height, target_width))

        new_image[:h, :w, :] = image
        new_mask[:h, :w] = mask

        assert np.count_nonzero(mask!=self.mask_value) == np.count_nonzero(new_mask!=self.mask_value)

        return new_image, new_mask


@TRANSFORMS.register_module
class HorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() > self.p:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        return image, mask


@TRANSFORMS.register_module
class RandomRotate(object):
    def __init__(self, p=0.5, degrees=30, mode='bilinear', border_mode='reflect101', image_value=None, mask_value=None):
        self.p = p
        self.degrees = (-degrees, degrees) if isinstance(degrees, (int, float)) else degrees
        self.mode = CV2_MODE[mode]
        self.border_mode = CV2_BORDER_MODE[border_mode]
        self.image_value = image_value
        self.mask_value = mask_value

    def __call__(self, image, mask):
        if random.random() < self.p:
            h, w, c = image.shape

            angle = random.uniform(*self.degrees)
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

            image = cv2.warpAffine(image, M=matrix, dsize=(w,h), flags=self.mode, borderMode=self.border_mode, borderValue=self.image_value)
            mask = cv2.warpAffine(mask, M=matrix, dsize=(w,h), flags=cv2.INTER_NEAREST, borderMode=self.border_mode, borderValue=self.mask_value)

        return image, mask


@TRANSFORMS.register_module
class GaussianBlur(object):
    def __init__(self, p=0.5, ksize=7):
        self.p = p
        self.ksize = (ksize, ksize) if isinstance(ksize, int) else ksize

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = cv2.GaussianBlur(image, ksize=self.ksize, sigmaX=0)

        return image, mask


@TRANSFORMS.register_module
class Normalize(object):
    def __init__(self, mean=(123.675, 116.280, 103.530), std=(58.395, 57.120, 57.375)):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        mean = np.reshape(np.array(self.mean, dtype=image.dtype), [1,1,3])
        std = np.reshape(np.array(self.std, dtype=image.dtype), [1,1,3])
        denominator = np.reciprocal(std, dtype=image.dtype)

        new_image = (image - mean) * denominator
        new_mask = mask

        return new_image, new_mask


@TRANSFORMS.register_module
class ToTensor(object):
    def __call__(self, image, mask):
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask)

        return image, mask
