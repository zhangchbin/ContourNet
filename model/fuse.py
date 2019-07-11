import torch
import torch.nn as nn
import torch.functional as F
from model.inception import inception

class fuse(nn.Module):
    def __init__(self, input_nc):
        super(fuse, self).__init__()
        self.conv_1_1_1 = nn.Conv2d(
            input_nc * 2, input_nc, kernel_size=1,
            stride=1
        )
        self.inception_module = inception(input_nc)
        self.relu = nn.ReLU()
        self.conv_1_1_2 = nn.Conv2d(input_nc, input_nc, kernel_size=1,
                                  stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pre_out):
        tmp = torch.cat([
            x, pre_out
        ], 1)
        tmp = self.conv_1_1_1(tmp)
        tmp = self.inception_module(tmp)
        tmp = self.relu(tmp)
        tmp = self.sigmoid(self.conv_1_1_2(tmp))
        tmp = torch.mul(tmp, x)
        out = tmp + pre_out
        return out
