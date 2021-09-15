import numpy as np
import os.path
from data.transform import get_transform
from torch.utils.data import Dataset
from PIL import Image
import random
import torch


class SingleImageDataset(Dataset):

    def __init__(self, opt):

        self.opt = opt
        self.root = opt.dataroot
        self.current_epoch = 0

        self.A_path = os.path.join(opt.dataroot, 'A.jpg')
        self.B_path = os.path.join(opt.dataroot, 'B.jpg')
        self.A_img = Image.open(self.A_path).convert('RGB')
        self.B_img = Image.open(self.B_path).convert('RGB')

        print("Image sizes %s and %s" % (str(self.A_img.size), str(self.B_img.size)))

        A_zoom = 1 / self.opt.random_scale_max
        zoom_levels_A = np.random.uniform(A_zoom, 1.0, size=(len(self) // opt.batch_size + 1, 1, 2))
        self.zoom_levels_A = np.reshape(np.tile(zoom_levels_A, (1, opt.batch_size, 1)), [-1, 2])

        # 每个batch内行相同，每行两个数,共batch_size*batch>=10000行

        B_zoom = 1 / self.opt.random_scale_max
        zoom_levels_B = np.random.uniform(B_zoom, 1.0, size=(len(self) // opt.batch_size + 1, 1, 2))
        self.zoom_levels_B = np.reshape(np.tile(zoom_levels_B, (1, opt.batch_size, 1)), [-1, 2])

        # 每个patch的编号，非一一对应
        self.patch_indices_A = list(range(len(self)))
        random.shuffle(self.patch_indices_A)
        self.patch_indices_B = list(range(len(self)))
        random.shuffle(self.patch_indices_B)

    def __getitem__(self, index):

        A_img = self.A_img
        B_img = self.B_img

        # apply image transformation
        if self.opt.phase == "train":
            param = {'scale_factor': self.zoom_levels_A[index],
                     'patch_index': self.patch_indices_A[index],
                     'flip': random.random() > 0.5}

            transform_A = get_transform(self.opt, params=param, method=Image.BILINEAR)
            A = transform_A(A_img)

            param = {'scale_factor': self.zoom_levels_B[index],
                     'patch_index': self.patch_indices_B[index],
                     'flip': random.random() > 0.5}
            transform_B = get_transform(self.opt, params=param, method=Image.BILINEAR)
            B = transform_B(B_img)
        else:
            transform = get_transform(self.opt, method=Image.BILINEAR)
            A = transform(A_img)
            B = transform(B_img)

        return {'A': A, 'B': B}

    def __len__(self):
        """ Let's pretend the single image contains 100,000 crops for convenience.
        """
        return 100000
