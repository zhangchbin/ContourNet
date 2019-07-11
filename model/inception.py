import torch
import torch.nn as nn
import torch.functional as F

#inception-module
# 4 branches ,each branch decrease the input-channel 1/4
# out-channel is equal to input-channel

class inception(nn.Module):
    def __init__(self, input_nc):
        super(inception, self).__init__()
        self.conv_1_1_list = nn.ModuleList([
            nn.Conv2d(input_nc,input_nc//4,1,1) for i in range(4)
        ])
        self.conv_3_3 = nn.Conv2d(input_nc//4, input_nc//4,3,1,
                                  padding=1)
        self.conv_5_5 = nn.Conv2d(input_nc//4, input_nc//4,5,1,
                                  padding=2)
        self.avgpooling = nn.AvgPool2d(3,1,padding=1)
    def forward(self, x):
        branch_1 = self.conv_1_1_list[0](x)
        branch_2 = self.conv_3_3(self.conv_1_1_list[1](x))
        branch_3 = self.conv_5_5(self.conv_1_1_list[2](x))
        branch_4 = self.conv_1_1_list[3](self.avgpooling(x))
        out = torch.cat([
            branch_1, branch_2, branch_3, branch_4
        ], 1)
        return out


