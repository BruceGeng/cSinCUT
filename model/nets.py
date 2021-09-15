import math
import numpy as np
import random

import torch
from torch import nn
from torch.nn import functional as F


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return F.leaky_relu(input + bias, negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        # print("FusedLeakyReLU: ", input.abs().mean())
        out = fused_leaky_relu(input, self.bias,
                               self.negative_slope,
                               self.scale)
        # print("FusedLeakyReLU: ", out.abs().mean())
        return out


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if len(k.shape) == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, minor, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, minor, in_h, 1, in_w, 1)
    out = F.pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0])
    out = out.view(-1, minor, in_h * up_y, in_w * up_x)

    out = F.pad(
        out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        :,
        max(-pad_y0, 0): out.shape[2] - max(-pad_y1, 0),
        max(-pad_x0, 0): out.shape[3] - max(-pad_x1, 0),
    ]

    # out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    # out = out.permute(0, 2, 3, 1)

    return out[:, :, ::down_y, ::down_x]

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out
class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = math.sqrt(1) / math.sqrt(in_channel * (kernel_size ** 2))

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        # print("Before EqualConv2d: ", input.abs().mean())
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
        # print("After EqualConv2d: ", out.abs().mean(), (self.weight * self.scale).abs().mean())

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], downsample=True, skip_gain=1.0):
        super().__init__()

        self.skip_gain = skip_gain
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample, blur_kernel=blur_kernel)

        if in_channel != out_channel or downsample:
            self.skip = ConvLayer(
                in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False
            )
        else:
            self.skip = nn.Identity()

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out * self.skip_gain + skip) / math.sqrt(self.skip_gain ** 2 + 1.0)

        return out

class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim=None,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        inject_noise=True,
    ):
        super().__init__()

        self.inject_noise = inject_noise
        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style=None, noise=None):
        out = self.conv(input, style)
        if self.inject_noise:
            out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out

class StyleGAN2Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False, opt=None):
        super().__init__()
        assert opt is not None
        self.opt = opt
        channel_multiplier = ngf / 32
        channels = {
            4: min(512, int(round(4096 * channel_multiplier))),
            8: min(512, int(round(2048 * channel_multiplier))),
            16: min(512, int(round(1024 * channel_multiplier))),
            32: min(512, int(round(512 * channel_multiplier))),
            64: int(round(256 * channel_multiplier)),
            128: int(round(128 * channel_multiplier)),
            256: int(round(64 * channel_multiplier)),
            512: int(round(32 * channel_multiplier)),
            1024: int(round(16 * channel_multiplier)),
        }

        blur_kernel = [1, 3, 3, 1]

        cur_res = 2 ** int((np.rint(np.log2(min(opt.load_size, opt.crop_size)))))
        convs = [nn.Identity(),
                 ConvLayer(3, channels[cur_res], 1)]

        num_downsampling = self.opt.stylegan2_G_num_downsampling
        for i in range(num_downsampling):
            in_channel = channels[cur_res]
            out_channel = channels[cur_res // 2]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel, downsample=True))
            cur_res = cur_res // 2

        for i in range(n_blocks // 2):
            n_channel = channels[cur_res]
            convs.append(ResBlock(n_channel, n_channel, downsample=False))

        self.convs = nn.Sequential(*convs)

    def forward(self, input, layers=[], get_features=False):
        feat = input
        feats = []
        if -1 in layers:
            layers.append(len(self.convs) - 1)
        for layer_id, layer in enumerate(self.convs):
            feat = layer(feat)
            # print(layer_id, " features ", feat.abs().mean())
            if layer_id in layers:
                feats.append(feat)

        if get_features:
            return feat, feats
        else:
            return feat


class StyleGAN2Decoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False, opt=None):
        super().__init__()
        assert opt is not None
        self.opt = opt

        blur_kernel = [1, 3, 3, 1]

        channel_multiplier = ngf / 32
        channels = {
            4: min(512, int(round(4096 * channel_multiplier))),
            8: min(512, int(round(2048 * channel_multiplier))),
            16: min(512, int(round(1024 * channel_multiplier))),
            32: min(512, int(round(512 * channel_multiplier))),
            64: int(round(256 * channel_multiplier)),
            128: int(round(128 * channel_multiplier)),
            256: int(round(64 * channel_multiplier)),
            512: int(round(32 * channel_multiplier)),
            1024: int(round(16 * channel_multiplier)),
        }

        num_downsampling = self.opt.stylegan2_G_num_downsampling
        cur_res = 2 ** int((np.rint(np.log2(min(opt.load_size, opt.crop_size))))) // (2 ** num_downsampling)
        convs = []

        for i in range(n_blocks // 2):
            n_channel = channels[cur_res]
            convs.append(ResBlock(n_channel, n_channel, downsample=False))

        for i in range(num_downsampling):
            in_channel = channels[cur_res]
            out_channel = channels[cur_res * 2]
            inject_noise = "small" not in self.opt.netG
            convs.append(
                StyledConv(in_channel, out_channel, 3, upsample=True, blur_kernel=blur_kernel, inject_noise=inject_noise)
            )
            cur_res = cur_res * 2

        convs.append(ConvLayer(channels[cur_res], 3, 1))

        self.convs = nn.Sequential(*convs)

    def forward(self, input):
        return self.convs(input)


class StyleGAN2Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, use_dropout=False, n_blocks=6, padding_type='reflect', no_antialias=False, opt=None):
        super().__init__()
        self.opt = opt
        self.encoder = StyleGAN2Encoder(input_nc, output_nc, ngf, use_dropout, n_blocks, padding_type, no_antialias, opt)
        self.decoder = StyleGAN2Decoder(input_nc, output_nc, ngf, use_dropout, n_blocks, padding_type, no_antialias, opt)

    def forward(self, input, layers=[], encode_only=False):
        feat, feats = self.encoder(input, layers, True)
        if encode_only:
            return feats
        else:
            fake = self.decoder(feat)

            if len(layers) > 0:
                return fake, feats
            else:
                return fake