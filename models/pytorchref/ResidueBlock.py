import torch
import torch.nn as nn
import math

from torch.nn.parameter import Parameter
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from datetime import datetime


class Affine(nn.Module):

    def __init__(self, num_parameters, scale = True, bias = True, scale_init= 1.0):
        super(Affine, self).__init__()
        if scale:
            self.scale = nn.Parameter(torch.ones(num_parameters)*scale_init)
        else:
            self.register_parameter('scale', None)
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_parameters))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        output = input
        if self.scale is not None:
            scale = self.scale.unsqueeze(0)
            while scale.dim() < input.dim():
                scale = scale.unsqueeze(2)
        output = output.mul(scale)

        if self.bias is not None:
            bias = self.bias.unsqueeze(0)
            while bias.dim() < input.dim():
                bias = bias.unsqueeze(2)
        output += bias

        return output

def compute_loss(input):
    input_flat = input.view(input.size(1), input.numel() // input.size(1))
    mean = input_flat.mean(1)
    lstd = (input_flat.pow(2).mean(1) - mean.pow(2)).sqrt().log()
    return mean.pow(2).mean() + lstd.pow(2).mean()

class BatchStatistics(nn.Module):
    def __init__(self, affine = -1):
        super(BatchStatistics, self).__init__()
        self.affine = nn.Sequential() if affine == -1 else Affine(affine)
        self.loss = 0
    
    def clear_loss(self):
        self.loss = 0

    def forward(self, input):
        self.loss = compute_loss(input)
        return self.affine(input)

class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, bias = True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, 1, 1, bias)
        self.weight.data.normal_()
        weight_norm = self.weight.pow(2).sum(3, keepdim = True).sum(2, keepdim = True).add(1e-8).sqrt()
        self.weight.data.div_(weight_norm)
        if bias:
            self.bias.data.zero_()

class ConvTranspose2d(nn.ConvTranspose1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0, bias = True):
        super(ConvTranspose1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, 0, 1, bias, 1)
        norm_scale = math.sqrt(self.stride[0])
        self.weight.data.normal_()
        weight_norm = self.weight.pow(2).sum(3, keepdim = True).sum(2, keepdim = True).add(1e-8).sqrt()
        self.weight.data.div_(weight_norm).mul_(norm_scale)
        if bias:
            self.bias.data.zero_()

class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias = True):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.weight.data.normal_()
        weight_norm = self.weight.pow(2).sum(1, keepdim = True).add(1e-8).sqrt()
        self.weight.data.div_(weight_norm)
        if bias:
            self.bias.data.zero_()

class CPReLU(nn.Module):

    def __init__(self, num_parameters = 1, init = 0.25):
        super(CPReLU, self).__init__()
        self.num_parameters = num_parameters
        self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))

        self.post_mean = (1 - init) / math.sqrt(2 * math.pi)
        post_ex2 = (1 + init ** 2) / 2
        self.post_stdv = math.sqrt(post_ex2 - self.post_mean ** 2)

    def forward(self, input):
        return (F.prelu(input, self.weight) - self.post_mean) / self.post_stdv

class ResidueBlock(nn.Module):

    def __init__(self, in_channels, out_channels, add_kernel, stride, pad = 0, residue_ratio = 0, use_BatchStatistics = True):
        super(ResidueBlock, self).__init__()

        self.residue_ratio = math.sqrt(residue_ratio)
        self.shortcut_ratio = math.sqrt(1 - residue_ratio)
        #self.residue_ratio = Parameter(torch.full((1, out_channels, 1), math.sqrt(residue_ratio)))
        #self.shortcut_ratio = Parameter(torch.full((1, out_channels, 1), math.sqrt(1 - residue_ratio)))

        self.residue = nn.Sequential(
            Conv2d(in_channels, out_channels, stride + add_kernel * 2, stride, pad + add_kernel),
            BatchStatistics(out_channels) if use_BatchStatistics==True else nn.Sequential(),
            CPReLU(out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2, padding = pad) if stride == 2 else nn.Sequential(),
            Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Sequential(),
            BatchStatistics(out_channels) if ((in_channels != out_channels) and (use_BatchStatistics==True)) else nn.Sequential()
        )

    def forward(self, input):
        return self.shortcut(input).mul(self.shortcut_ratio) + self.residue(input).mul(self.residue_ratio)


    
    
    
    
    
