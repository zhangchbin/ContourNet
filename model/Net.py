from args import args, flags
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50, resnet101
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.residualBlock import residualBlock
from model.upsample import upsample
from model.inception import inception
from model.together import together
from model.fuse import fuse
from model.process import process

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = resnet50(pretrained = True)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.layer_1_inner_resBlcok_0 = residualBlock(256)
        self.layer_1_inner_resBlcok_1 = residualBlock(256)
        self.layer_1_inner_resBlcok_2 = residualBlock(256)

        self.layer_2_inner_resBlcok_0 = residualBlock(512)
        self.layer_2_inner_resBlcok_1 = residualBlock(512)
        self.layer_2_inner_resBlcok_2 = residualBlock(512)
        self.layer_2_inner_resBlcok_3 = residualBlock(512)

        self.layer_3_inner_resBlock_0 = residualBlock(1024)
        self.layer_3_inner_resBlock_1 = residualBlock(1024)
        self.layer_3_inner_resBlock_2 = residualBlock(1024)
        self.layer_3_inner_resBlock_3 = residualBlock(1024)
        self.layer_3_inner_resBlock_4 = residualBlock(1024)
        self.layer_3_inner_resBlock_5 = residualBlock(1024)

        self.layer_4_inner_resBlock_0 = residualBlock(2048)
        self.layer_4_inner_resBlock_1 = residualBlock(2048)
        self.layer_4_inner_resBlock_2 = residualBlock(2048)

        self.avgpool = nn.AvgPool2d(3,1,padding=1)

        self.layer_1_inner_process_0 = process(256)
        self.layer_1_inner_process_1 = process(256)
        self.layer_1_inner_process_2 = process(256)

        self.layer_2_inner_process_0 = process(512)
        self.layer_2_inner_process_1 = process(512)
        self.layer_2_inner_process_2 = process(512)
        self.layer_2_inner_process_3 = process(512)

        self.layer_3_inner_process_0 = process(1024)
        self.layer_3_inner_process_1 = process(1024)
        self.layer_3_inner_process_2 = process(1024)
        self.layer_3_inner_process_3 = process(1024)
        self.layer_3_inner_process_4 = process(1024)
        self.layer_3_inner_process_5 = process(1024)

        self.layer_4_inner_process_0 = process(2048)
        self.layer_4_inner_process_1 = process(2048)
        self.layer_4_inner_process_2 = process(2048)

        self.layer_1_inner_together_1 = together(256)
        self.layer_1_inner_together_2 = together(256)

        self.layer_2_inner_together_1 = together(512)
        self.layer_2_inner_together_2 = together(512)
        self.layer_2_inner_together_3 = together(512)

        self.layer_3_inner_together_1 = together(1024)
        self.layer_3_inner_together_2 = together(1024)
        self.layer_3_inner_together_3 = together(1024)
        self.layer_3_inner_together_4 = together(1024)
        self.layer_3_inner_together_5 = together(1024)

        self.layer_4_inner_together_1 = together(2048)
        self.layer_4_inner_together_2 = together(2048)

        #left the fuse-block
        self.layer_1_outer_resBlock_1 = residualBlock(256)
        self.layer_1_outer_resBlock_2 = residualBlock(256)
        self.layer_2_outer_resBlock_1 = residualBlock(512)
        self.layer_2_outer_resBlock_2 = residualBlock(512)
        self.layer_3_outer_resBlock_1 = residualBlock(1024)
        self.layer_3_outer_resBlock_2 = residualBlock(1024)
        self.layer_4_outer_resBlock_1 = residualBlock(2048)
        self.layer_4_outer_resBlock_2 = residualBlock(2048)

        self.layer_1_outer_fuse = fuse(256)
        self.layer_2_outer_fuse = fuse(512)
        self.layer_3_outer_fuse = fuse(1024)
        self.layer_4_outer_fuse = fuse(2048)

        self.layer_1_downsample = nn.Conv2d(256, 1, 1, 1)
        self.layer_2_downsample = nn.Conv2d(512, 256, 1, 1)
        self.layer_3_downsample = nn.Conv2d(1024, 512, 1, 1)
        self.layer_4_downsample = nn.Conv2d(2048, 1024, 1, 1)


    def forward(self, input):
        input_h, input_w = input.size()[2], input.size()[3]
        #stage-1
        x = self.model.conv1(input)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        #layer-1
        layer_1_res_0 = self.model.layer1[0](x)
        layer_1_res_1 = self.model.layer1[1](layer_1_res_0)
        layer_1_res_2 = self.model.layer1[2](layer_1_res_1)

        #layer-2
        layer_2_res_0 = self.model.layer2[0](layer_1_res_2)
        layer_2_res_1 = self.model.layer2[1](layer_2_res_0)
        layer_2_res_2 = self.model.layer2[2](layer_2_res_1)
        layer_2_res_3 = self.model.layer2[3](layer_2_res_2)

        #layer-3
        layer_3_res_0 = self.model.layer3[0](layer_2_res_3)
        layer_3_res_1 = self.model.layer3[1](layer_3_res_0)
        layer_3_res_2 = self.model.layer3[2](layer_3_res_1)
        layer_3_res_3 = self.model.layer3[3](layer_3_res_2)
        layer_3_res_4 = self.model.layer3[4](layer_3_res_3)
        layer_3_res_5 = self.model.layer3[5](layer_3_res_4)

        #layer-4
        layer_4_res_0 = self.model.layer4[0](layer_3_res_5)
        layer_4_res_1 = self.model.layer4[1](layer_4_res_0)
        layer_4_res_2 = self.model.layer4[2](layer_4_res_1)

        #avg-pool
        avg_pool = self.avgpool(layer_4_res_2)

        #layer-1-sideout
        layer_1_side_res_0 = self.layer_1_inner_resBlcok_0(layer_1_res_0)
        layer_1_side_res_1 = self.layer_1_inner_resBlcok_1(layer_1_res_1)
        layer_1_side_res_2 = self.layer_1_inner_resBlcok_2(layer_1_res_2)
        layer_1_side_res_0 = self.layer_1_inner_process_0(layer_1_side_res_0)
        layer_1_side_res_1 = self.layer_1_inner_process_1(layer_1_side_res_1)
        layer_1_side_res_2 = self.layer_1_inner_process_2(layer_1_side_res_2)
        layer_1_side_tb_1 = self.layer_1_inner_together_1(layer_1_side_res_1, layer_1_side_res_0)
        layer_1_side_tb_2 = self.layer_1_inner_together_2(layer_1_side_res_2, layer_1_side_tb_1)
        layer_1_side_out = layer_1_side_tb_2

        #layer-2-sideout
        layer_2_side_res_0 = self.layer_2_inner_resBlcok_0(layer_2_res_0)
        layer_2_side_res_1 = self.layer_2_inner_resBlcok_1(layer_2_res_1)
        layer_2_side_res_2 = self.layer_2_inner_resBlcok_2(layer_2_res_2)
        layer_2_side_res_3 = self.layer_2_inner_resBlcok_3(layer_2_res_3)
        layer_2_side_res_0 = self.layer_2_inner_process_0(layer_2_side_res_0)
        layer_2_side_res_1 = self.layer_2_inner_process_1(layer_2_side_res_1)
        layer_2_side_res_2 = self.layer_2_inner_process_2(layer_2_side_res_2)
        layer_2_side_res_3 = self.layer_2_inner_process_3(layer_2_side_res_3)
        layer_2_side_tb_1 = self.layer_2_inner_together_1(layer_2_side_res_1, layer_2_side_res_0)
        layer_2_side_tb_2 = self.layer_2_inner_together_2(layer_2_side_res_2, layer_2_side_tb_1)
        layer_2_side_tb_3 = self.layer_2_inner_together_3(layer_2_side_res_3, layer_2_side_tb_2)
        layer_2_side_out = layer_2_side_tb_3

        #layer-3-sideout
        layer_3_side_res_0 = self.layer_3_inner_resBlock_0(layer_3_res_0)
        layer_3_side_res_1 = self.layer_3_inner_resBlock_1(layer_3_res_1)
        layer_3_side_res_2 = self.layer_3_inner_resBlock_2(layer_3_res_2)
        layer_3_side_res_3 = self.layer_3_inner_resBlock_3(layer_3_res_3)
        layer_3_side_res_4 = self.layer_3_inner_resBlock_4(layer_3_res_4)
        layer_3_side_res_5 = self.layer_3_inner_resBlock_5(layer_3_res_5)
        layer_3_side_res_0 = self.layer_3_inner_process_0(layer_3_side_res_0)
        layer_3_side_res_1 = self.layer_3_inner_process_1(layer_3_side_res_1)
        layer_3_side_res_2 = self.layer_3_inner_process_2(layer_3_side_res_2)
        layer_3_side_res_3 = self.layer_3_inner_process_3(layer_3_side_res_3)
        layer_3_side_res_4 = self.layer_3_inner_process_4(layer_3_side_res_4)
        layer_3_side_res_5 = self.layer_3_inner_process_5(layer_3_side_res_5)
        layer_3_side_tb_1 = self.layer_3_inner_together_1(layer_3_side_res_1, layer_3_side_res_0)
        layer_3_side_tb_2 = self.layer_3_inner_together_2(layer_3_side_res_2, layer_3_side_tb_1)
        layer_3_side_tb_3 = self.layer_3_inner_together_3(layer_3_side_res_3, layer_3_side_tb_2)
        layer_3_side_tb_4 = self.layer_3_inner_together_4(layer_3_side_res_4, layer_3_side_tb_3)
        layer_3_side_tb_5 = self.layer_3_inner_together_5(layer_3_side_res_5, layer_3_side_tb_4)
        layer_3_side_out = layer_3_side_tb_5

        #layer-4-sideout
        layer_4_side_res_0 = self.layer_4_inner_resBlock_0(layer_4_res_0)
        layer_4_side_res_1 = self.layer_4_inner_resBlock_1(layer_4_res_1)
        layer_4_side_res_2 = self.layer_4_inner_resBlock_2(layer_4_res_2)
        layer_4_side_res_0 = self.layer_4_inner_process_0(layer_4_side_res_0)
        layer_4_side_res_1 = self.layer_4_inner_process_1(layer_4_side_res_1)
        layer_4_side_res_2 = self.layer_4_inner_process_2(layer_4_side_res_2)
        layer_4_side_tb_1 = self.layer_4_inner_together_1(layer_4_side_res_1, layer_4_side_res_0)
        layer_4_side_tb_2 = self.layer_4_inner_together_2(layer_4_side_res_2, layer_4_side_tb_1)
        layer_4_side_out = layer_4_side_tb_2

        layer_1_side_out = self.layer_1_outer_resBlock_1(layer_1_side_out)
        layer_2_side_out = self.layer_2_outer_resBlock_1(layer_2_side_out)
        layer_3_side_out = self.layer_3_outer_resBlock_1(layer_3_side_out)
        layer_4_side_out = self.layer_4_outer_resBlock_1(layer_4_side_out)


        layer_4_side_out_fuse = self.layer_4_outer_fuse(layer_4_side_out,
                                                        avg_pool)
        layer_4_side_out_fuse = self.layer_4_outer_resBlock_2(layer_4_side_out_fuse)
        layer_4_side_out_fuse = self.layer_4_downsample(layer_4_side_out_fuse)
        layer_4_side_upsample = \
            F.interpolate(layer_4_side_out_fuse,
                          size=(layer_3_side_out.size()[2],layer_3_side_out.size()[3]),
                          mode='bilinear', align_corners=True)
        layer_3_side_out_fuse = self.layer_3_outer_fuse(layer_3_side_out,
                                                        layer_4_side_upsample)
        layer_3_side_out_fuse = self.layer_3_outer_resBlock_2(layer_3_side_out_fuse)
        layer_3_side_out_fuse = self.layer_3_downsample(layer_3_side_out_fuse)
        layer_3_side_upsample = \
            F.interpolate(layer_3_side_out_fuse,
                          (layer_2_side_out.size()[2],layer_2_side_out.size()[3]),
                          mode='bilinear', align_corners=True)

        layer_2_side_out_fuse = self.layer_2_outer_fuse(layer_2_side_out,
                                                        layer_3_side_upsample)
        layer_2_side_out_fuse = self.layer_2_outer_resBlock_2(layer_2_side_out_fuse)
        layer_2_side_out_fuse = self.layer_2_downsample(layer_2_side_out_fuse)

        layer_2_side_upsample = \
            F.interpolate(layer_2_side_out_fuse,
                          (layer_1_side_out.size()[2], layer_1_side_out.size()[3]),
                          mode='bilinear', align_corners=True)

        layer_1_side_out_fuse = self.layer_1_outer_fuse(layer_1_side_out,
                                                        layer_2_side_upsample)
        layer_1_side_out_fuse = self.layer_1_outer_resBlock_2(layer_1_side_out_fuse)
        layer_1_side_out_fuse = self.layer_1_downsample(layer_1_side_out_fuse)
        layer_1_side_upsample = \
            F.interpolate(layer_1_side_out_fuse,
                          (input.size()[2], input.size()[3]),
                          mode='bilinear', align_corners=True)
        out = F.sigmoid(layer_1_side_upsample)
        return out, layer_1_side_out, layer_2_side_out, layer_3_side_out, layer_4_side_out


