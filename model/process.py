import torch
import torch.nn as nn
import torch.functional as F
from model.inception import inception

class process(nn.Module):
    def __init__(self, input_nc):
        super(process, self).__init__()
        self.conv_1_1 = nn.Conv2d(input_nc, input_nc,1,1)
        self.bn = nn.BatchNorm2d(input_nc)
        self.relu = nn.ReLU()
        self.conv_3_3 = nn.Conv2d(input_nc, input_nc,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        self.inception_module = inception(input_nc)
    def forward(self, x):
        tmp = self.conv_1_1(x)
        main_path = self.inception_module(tmp)
        main_path = self.relu(self.bn(main_path))
        main_path = self.conv_3_3(main_path)
        sum = main_path + tmp
        out = self.relu(sum)
        return out