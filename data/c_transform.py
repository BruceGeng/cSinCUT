import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


def get_c_transform(opt, params=None, method=Image.BICUBIC, convert=True):
    transform_list = []

    if 'zoom' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __random_zoom(img, opt.crop_size, method, factor=params["scale_factor"])))

    if 'patch' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __c_patch(img, params['patch_location'], opt.crop_size)))

    transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if 'flip' in params:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


# 使图像分辨率2的幂
def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    return img.resize((w, h), method)

# 随机缩放，每个batch内缩放相同
def __random_zoom(img, crop_width, method=Image.BICUBIC, factor=None):
    zoom_level = (factor[0], factor[1])
    iw, ih = img.size
    zoomw = max(crop_width, iw * zoom_level[0])
    zoomh = max(crop_width, ih * zoom_level[1])
    img = img.resize((int(round(zoomw)), int(round(zoomh))), method)
    return img

# 根据location取出一个patch
def __c_patch(img, loc, size):
    w, h = img.size
    x, y = loc
    return img.crop((x, y, x + size, y + size))


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img
