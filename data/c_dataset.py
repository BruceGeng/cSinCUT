import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os.path
from data.c_transform import get_c_transform
from torch.utils.data import Dataset
from PIL import Image
import random
import torch
import torchvision.transforms as transforms
import cv2

# 随机缩放，每个batch内缩放相同
def focus_zoom(img, crop_width, method, factor=None):
    zoom_level = (factor[0], factor[1])
    iw, ih = img.size
    zoomw = max(crop_width, iw * zoom_level[0])
    zoomh = max(crop_width, ih * zoom_level[1])
    img = img.resize((int(round(zoomw)), int(round(zoomh))), method)
    return img

def random_patch_locations(focus_img, crop_width, zoom_levels, focus=True):
    loc = []
    for index in range(len(zoom_levels)):
        zoomed_img = focus_zoom(focus_img, crop_width, Image.NEAREST, zoom_levels[index])
        ts = transforms.Compose([transforms.ToTensor()])
        mask = ts(zoomed_img).squeeze(dim=0)
        s_x = random.randint(0, mask.shape[1]-crop_width)
        s_y = random.randint(0, mask.shape[0]-crop_width)

        if focus==True:
            while torch.min(mask[s_y:s_y+crop_width,s_x:s_x+crop_width]) != 1:
                s_x = random.randint(0, mask.shape[1] - crop_width)
                s_y = random.randint(0, mask.shape[0] - crop_width)
        else:
            while torch.max(mask[s_y:s_y+crop_width,s_x:s_x+crop_width]) != 0:
                s_x = random.randint(0, mask.shape[1] - crop_width)
                s_y = random.randint(0, mask.shape[0] - crop_width)
        loc.append([s_x, s_y])
        if index % 10000 == 0:
            print("{}/100000".format(index))
    print("finished")

    return loc

def visualize_patch(focus_path, crop_width, zoom_levels_A, patch_locations_A, zoom_levels_B, patch_locations_B):
    img = Image.open(focus_path)
    iw, ih = img.size
    img = np.array(img)
    print('visualizing...')
    for index in range(len(patch_locations_A)):
        zoomw = max(crop_width, iw * zoom_levels_A[index][0])
        zoomh = max(crop_width, ih * zoom_levels_A[index][1])
        img = cv2.resize(img, (int(round(zoomw)), int(round(zoomh))), interpolation=cv2.INTER_CUBIC)
        img = cv2.rectangle(img, (patch_locations_A[index][0], patch_locations_A[index][1]), (patch_locations_A[index][0]+crop_width, patch_locations_A[index][1]+crop_width), (255, 0, 0), 3)
        img = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_CUBIC)

    for index in range(len(patch_locations_B)):
        zoomw = max(crop_width, iw * zoom_levels_B[index][0])
        zoomh = max(crop_width, ih * zoom_levels_B[index][1])
        img = cv2.resize(img, (int(round(zoomw)), int(round(zoomh))), interpolation=cv2.INTER_CUBIC)
        img = cv2.rectangle(img, (patch_locations_B[index][0], patch_locations_B[index][1]), (patch_locations_B[index][0]+crop_width, patch_locations_B[index][1]+crop_width), (0, 255, 0), 3)
        img = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_CUBIC)
    print("showing")
    plt.imshow(img)
    plt.show()

def crop_test_area(img, focus):
    ts_img = transforms.Compose([transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ts_focus = transforms.Compose([transforms.ToTensor()])
    input = ts_img(img)
    mask = ts_focus(focus).squeeze(dim=0)
    miny = 10000000000
    maxy = 0
    minx = 10000000000
    maxx = 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 0:
                miny = min(miny, i)
                maxy = max(maxy, i)
                minx = min(minx, j)
                maxx = max(maxx, j)
    cropped = input[:, miny:maxy, minx:maxx]

    return cropped, [miny, maxy, minx, maxx]


class SingleImageDataset(Dataset):

    def __init__(self, opt):

        self.opt = opt
        self.root = opt.dataroot
        self.current_epoch = 0

        self.img_path = os.path.join(opt.dataroot, 'image.jpg')
        self.focus_path = os.path.join(opt.dataroot, 'focus2.jpg')
        self.img_img = Image.open(self.img_path).convert('RGB')
        self.focus_img = Image.open(self.focus_path).convert('L')

        assert self.img_img.size == self.focus_img.size, \
            "The focus image should have the same size with the input image"

        print("Image sizes %s " % (str(self.img_img.size)))


        if self.opt.phase == "train":

            A_zoom = 1 / self.opt.random_scale_max
            zoom_levels_A = np.random.uniform(A_zoom, 1.0, size=(len(self) // opt.batch_size + 1, 1, 2))
            self.zoom_levels_A = np.reshape(np.tile(zoom_levels_A, (1, opt.batch_size, 1)), [-1, 2])

            # 每个batch内行相同，每行两个数,共batch_size*batch>=10000行

            B_zoom = 1 / self.opt.random_scale_max
            zoom_levels_B = np.random.uniform(B_zoom, 1.0, size=(len(self) // opt.batch_size + 1, 1, 2))
            self.zoom_levels_B = np.reshape(np.tile(zoom_levels_B, (1, opt.batch_size, 1)), [-1, 2])

            # 每个patch对应的位置(m,n)
            self.patch_locations_A = random_patch_locations(self.focus_img, opt.crop_size, self.zoom_levels_A, focus=False)
            self.patch_locations_B = random_patch_locations(self.focus_img, opt.crop_size, self.zoom_levels_B, focus=True)

            #visualize_patch(self.focus_path, opt.crop_size, self.zoom_levels_A, self.patch_locations_A, self.zoom_levels_B, self.patch_locations_B)

    def __getitem__(self, index):

        img_img  = self.img_img

        # apply image transformation
        if self.opt.phase == "train":
            param = {'scale_factor': self.zoom_levels_A[index],
                     'patch_location': self.patch_locations_A[index],
                     'flip': random.random() > 0.5}

            transform_A = get_c_transform(self.opt, params=param, method=Image.BILINEAR)
            A = transform_A(img_img)

            param = {'scale_factor': self.zoom_levels_B[index],
                     'patch_location': self.patch_locations_B[index],
                     'flip': random.random() > 0.5}
            transform_B = get_c_transform(self.opt, params=param, method=Image.BILINEAR)
            B = transform_B(img_img)
        else:
            transform = get_c_transform(self.opt, method=Image.BILINEAR)
            cropped, minmax = crop_test_area(img_img, self.focus_img)
            ts = transforms.Compose([transforms.ToTensor()])
            img_tensor = ts(img_img)
            focus_tensor = ts(self.focus_img)
        return {'A': cropped, 'B': cropped, 'AREA': minmax, 'IMG': img_tensor, 'FOCUS': focus_tensor}

    def __len__(self):
        """ Let's pretend the single image contains 100,000 crops for convenience.
        """
        return 100000

