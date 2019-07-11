import torch
import torch.nn as nn
import torch.functional as F


#single-res befor the process Module
class residualBlock(nn.Module):
    def __init__(self, input_nc):
        super(residualBlock, self).__init__()
        self.conv1_1 = nn.Conv2d(input_nc,input_nc,1,stride=1)
        self.conv3_3_1 = nn.Conv2d(input_nc,input_nc,3,stride=1,
                                   padding=1)
        self.bn = nn.BatchNorm2d(input_nc)
        self.conv3_3_2 = nn.Conv2d(input_nc, input_nc,3,stride=1,
                                   padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        tmp = self.conv1_1(x)
        branch_1 = self.conv3_3_1(tmp)
        branch_1 = self.bn(branch_1)
        branch_1 = self.relu(branch_1)
        branch_1 = self.conv3_3_2(branch_1)
        return self.relu(tmp + branch_1)
