import torch
import torch.nn as nn
import torch.functional as F
from model.inception import inception

class together(nn.Module):
    def __init__(self, input_nc):
        super(together, self).__init__()
        self.inception_module = inception(input_nc)


    def forward(self, x, pre_out):
        tmp = torch.cat([
            pre_out, x
        ], 1)
        out = self.inception_module(x)
        return out