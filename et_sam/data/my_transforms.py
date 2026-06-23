import numbers
from collections.abc import Sequence
from typing import List, Optional, Literal

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue


class ColorJitter(object):
    def __init__(self, brightness=0.7, contrast=0.7, saturation=0.7, hue=0.5):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @staticmethod
    def _check_input(value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def _get_params(brightness: Optional[List[float]],
                    contrast: Optional[List[float]],
                    saturation: Optional[List[float]],
                    hue: Optional[List[float]]
                    ):
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def __call__(self, sample):
        image = sample['image']  # np.array, hwc
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(dim=0)
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self._get_params(self.brightness, self.contrast, self.saturation, self.hue)
        for idx in fn_idx:
            if idx == 0 and brightness_factor is not None:
                image = adjust_brightness(image, brightness_factor)
            elif idx == 1 and contrast_factor is not None:
                image = adjust_contrast(image, contrast_factor)
            elif idx == 2 and saturation_factor is not None:
                image = adjust_saturation(image, saturation_factor)
            elif idx == 3 and hue_factor is not None:
                image = adjust_hue(image, hue_factor)
        sample['image'] = image.squeeze(dim=0).permute(1, 2, 0).numpy().astype(np.float32)
        return sample


def create_rotation_matrix(center, angle, bound_w, bound_h, offset):
    center_offset = (center[0] + offset, center[1] + offset)
    rm = cv2.getRotationMatrix2D(tuple(center_offset), angle, 1)
    rot_im_center = cv2.transform(center[None, None, :] + offset, rm)[0, 0, :]
    new_center = np.array([bound_w / 2, bound_h / 2]) + offset - rot_im_center
    rm[:, 2] += new_center
    return rm


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))


def _setup_angle(x, name, req_sizes=(2,)):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


class RandomRotate(object):
    def __init__(self, angle=180):
        self.angle = _setup_angle(angle, name="angle", req_sizes=(2,))

    @staticmethod
    def apply(img, rm_img, bound_w, bound_h, interp=None, border_value=None):
        interp = interp if interp is not None else cv2.INTER_LINEAR
        return cv2.warpAffine(img, rm_img, (bound_w, bound_h), flags=interp, borderValue=border_value)

    @staticmethod
    def get_center_bound(image, angle):
        h, w = image.shape[:2]
        center = np.array((w / 2, h / 2))
        abs_cos, abs_sin = (abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle))))
        bound_w, bound_h = np.rint(
            [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
        ).astype(int)
        return center, bound_w, bound_h

    def __call__(self, sample):
        image, heatmap = sample['image'], sample['heatmap']
        angle = np.random.uniform(self.angle[0], self.angle[1])

        center, bound_w, bound_h = self.get_center_bound(image, angle)
        center_heatmap, bound_w_heatmap, bound_h_heatmap = self.get_center_bound(heatmap, angle)

        rm_image = create_rotation_matrix(center, angle, bound_w, bound_h, offset=-0.5)
        rm_heatmap = create_rotation_matrix(center_heatmap, angle, bound_w_heatmap, bound_h_heatmap, offset=-0.5)

        image = self.apply(image, rm_image, bound_w, bound_h, cv2.INTER_NEAREST, (128, 128, 128))
        heatmap = self.apply(heatmap, rm_heatmap, bound_w_heatmap, bound_h_heatmap, cv2.INTER_NEAREST, 0)

        sample['image'] = image
        sample['heatmap'] = heatmap

        for key in ['word_masks', 'line_word_masks', 'line_masks', 'paragraph_masks']:
            masks = sample.get(key, None)
            if masks is not None:
                masks = self.apply(masks, rm_image, bound_w, bound_h, cv2.INTER_NEAREST, 0)
                if len(masks.shape) < 3:
                    masks = masks[:, :, np.newaxis]
                sample[key] = masks

        return sample


class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
        https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py
    """

    def __init__(self, output_size=1024, aug_scale_min=0.5, aug_scale_max=2, heatmap_scale=4):
        self.desired_size = output_size
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.heatmap_scale = heatmap_scale

    @staticmethod
    def apply_masks(sample, key, scaled_size, crop_x, crop_y, padding):
        masks = sample.get(key, None)
        if masks is not None:
            masks = cv2.resize(masks, dsize=scaled_size[::-1], interpolation=cv2.INTER_LINEAR)
            if len(masks.shape) < 3:
                masks = masks[:, :, np.newaxis]
            masks = masks[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1], :]
            masks = cv2.copyMakeBorder(masks, 0, padding[0], 0, padding[1], cv2.BORDER_CONSTANT, value=[0])
            if len(masks.shape) < 3:
                masks = masks[:, :, np.newaxis]
            sample[key] = masks

    @staticmethod
    def apply_image(image, desired_size, random_scale, offset_h, offset_w, padding_value=(128, 128, 128)):
        image_size = np.array(image.shape[:2])

        # scale
        scaled_size = (random_scale * desired_size).round()
        scale = np.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().astype(np.int32)  # h, w
        scaled_image = cv2.resize(image, dsize=scaled_size[::-1], interpolation=cv2.INTER_LINEAR)

        # crop
        crop_size = (min(desired_size, scaled_size[0]), min(desired_size, scaled_size[1]))
        margin_h = max(scaled_size[0] - crop_size[0], 0)
        margin_w = max(scaled_size[1] - crop_size[1], 0)
        offset_h, offset_w = margin_h * offset_h, margin_w * offset_w
        crop_y = np.array([offset_h, offset_h + crop_size[0]], dtype=np.int32)
        crop_x = np.array([offset_w, offset_w + crop_size[1]], dtype=np.int32)
        scaled_image = scaled_image[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]

        # pad
        padding_h = max(desired_size - scaled_image.shape[0], 0)
        padding_w = max(desired_size - scaled_image.shape[1], 0)
        image = cv2.copyMakeBorder(scaled_image, 0, padding_h, 0, padding_w, cv2.BORDER_CONSTANT, value=padding_value)

        return image, (crop_size, scaled_size, crop_x, crop_y, padding_h, padding_w)

    def __call__(self, sample):
        image, heatmap = sample['image'], sample['heatmap']

        random_scale = np.random.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        offset_h = np.random.rand()
        offset_w = np.random.rand()

        image, other_info = self.apply_image(image, self.desired_size, random_scale, offset_h, offset_w)
        heatmap, heatmap_other_info = self.apply_image(heatmap, self.desired_size // self.heatmap_scale, random_scale,
                                                       offset_h, offset_w, padding_value=(0, 0, 0))
        crop_size, scaled_size, crop_x, crop_y, padding_h, padding_w = other_info
        sample.update(image=image, heatmap=heatmap)
        sample["heatmap_shape"] = torch.tensor(heatmap_other_info[0])
        for key in ['word_masks', 'line_word_masks', 'line_masks', 'paragraph_masks']:
            self.apply_masks(sample, key, scaled_size, crop_x, crop_y, (padding_h, padding_w))

        return sample


class ToTensor(object):
    def __call__(self, sample):
        sample['image'] = torch.as_tensor(sample['image'], dtype=torch.float).permute(2, 0, 1).contiguous()
        sample['heatmap'] = torch.as_tensor(sample['heatmap'][..., None], dtype=torch.float).permute(2, 0, 1).contiguous()
        sample['heatmap_shape'] = torch.as_tensor(sample['heatmap_shape'])

        for key in ['word_masks', 'line_word_masks', 'line_masks', 'paragraph_masks']:
            masks = sample.get(key, None)
            if masks is not None:
                sample[key] = torch.from_numpy(masks).permute(2, 0, 1).contiguous().float()

        return sample


class ResizeLongestSide_ToTensor(object):
    def __init__(self, target_length=1024, heatmap_scale=4):
        self.target_length = target_length
        self.heatmap_target_length = target_length // heatmap_scale

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        scale = long_side_length / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return [newh, neww], scale

    def __call__(self, sample):
        # image: np.array, [h, w, c]
        image, heatmap = sample['image'], sample['heatmap']

        image = torch.as_tensor(image, dtype=torch.float).permute(2, 0, 1)
        heatmap = torch.as_tensor(heatmap, dtype=torch.float)[..., None].permute(2, 0, 1)

        image_new_shape, scale = self.get_preprocess_shape(image.shape[1], image.shape[2], self.target_length)
        heatmap_new_shape, _ = self.get_preprocess_shape(heatmap.shape[1], heatmap.shape[2], self.heatmap_target_length)

        image = (F.interpolate(image.unsqueeze(0), image_new_shape, mode='bilinear')).squeeze(0)
        heatmap = (F.interpolate(heatmap.unsqueeze(0), heatmap_new_shape, mode='bilinear')).squeeze(0)

        # Pad
        h, w = image.shape[-2:]
        padh = self.target_length - h
        padw = self.target_length - w
        image = F.pad(image, (0, padw, 0, padh))

        h, w = heatmap.shape[-2:]
        padh = self.heatmap_target_length - h
        padw = self.heatmap_target_length - w
        heatmap = F.pad(heatmap, (0, padw, 0, padh))

        sample['image'] = image
        sample['heatmap'] = heatmap
        sample['image_shape'] = torch.as_tensor(image_new_shape, dtype=torch.int)
        sample['heatmap_shape'] = torch.as_tensor(heatmap_new_shape, dtype=torch.int)
        sample['scale'] = scale
        return sample
