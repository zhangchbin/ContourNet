import torch
import torch.nn as nn
import torch.nn.functional as F

class upsample(nn.Module):
    output_w = 0
    def __init__(self, input_nc, output_nc, output_w):
        super(upsample, self).__init__()
        self.conv = nn.Conv2d(input_nc, output_nc,
                              kernel_size=1, stride=1)
        self.output_w = output_w
    def forward(self, x):
        x = F.upsample_bilinear(x, size=self.output_w)
        x = self.conv(x)
        return x
