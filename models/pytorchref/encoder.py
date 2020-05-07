#import os
import torch
import torch.nn as nn
#from torch.autograd import Variable
import numpy as np\
from torch.autograd import Variable as tv

from ResidueBlock import ResidueBlock, Linear, CPReLU, BatchStatistics, Affine,Conv1d

import cv2 as cv


##input size b*3*224*224
class Encoder_complex(nn.Module):
    def __init__(self, joint_num=57):
        super(Generator, self).__init__()

        self.cnn = nn.Sequential(
            ResidueBlock(in_channels=3, out_channels=32, add_kernel=1, stride=2, pad = 0, residue_ratio = 0.5, use_BatchStatistics = False), #112
            ResidueBlock(in_channels=3, out_channels=64, add_kernel=1, stride=2, pad = 0, residue_ratio = 0.5, use_BatchStatistics = False), #56      
            ResidueBlock(in_channels=3, out_channels=128, add_kernel=1, stride=2, pad = 0, residue_ratio = 0.5, use_BatchStatistics = False), #28
            ResidueBlock(in_channels=3, out_channels=256, add_kernel=1, stride=2, pad = 0, residue_ratio = 0.5, use_BatchStatistics = False), #14
            ResidueBlock(in_channels=3, out_channels=256, add_kernel=1, stride=2, pad = 0, residue_ratio = 0.5, use_BatchStatistics = False), #7
            )
        self.fn = nn.Sequential(
            Linear(7*7*256, 1024)
            )

    #input b*3*224*224 output b*1024
    def forward(self, input):
        batch = input.shape[0]
        output = self.cnn(input)
        output = self.fn(output.view(batch, -1) )

        return output







        


