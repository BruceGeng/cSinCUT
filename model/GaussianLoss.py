import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = [[0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.01330373, 0.11098164, 0.22508352, 0.11098164, 0.01330373],
                  [0.00655965, 0.05472157, 0.11098164, 0.05472157, 0.00655965],
                  [0.00078633, 0.00655965, 0.01330373, 0.00655965, 0.00078633]]
        print(torch.FloatTensor(kernel).shape)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        print(kernel.shape)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
    def __call__(self, x):
        re_pad = nn.ReflectionPad2d(2)
        x = re_pad(x.unsqueeze(0))
        x = F.conv2d(x, self.weight, padding=0, groups=self.channels)
        return x.squeeze(0)

path = r"../datasets/grass/LHQTCj-h.jpg"

input_x = cv2.imread(path)
input_x = Variable(torch.from_numpy(input_x.astype(np.float32))).permute(2, 0, 1)
gaussian_conv = GaussianBlurConv()
out_x = gaussian_conv(input_x)
for i in range(10):
    out_x = gaussian_conv(out_x)
print(input_x.shape)
print(out_x.shape)
#out_x = gaussian_conv(out_x)
#out_x = out_x.squeeze(0).permute(1, 2, 0).data.numpy().astype(np.uint8)
out_x = out_x.permute(1, 2, 0).data.numpy().astype(np.uint8)

plt.imshow(out_x)
plt.show()