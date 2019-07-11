import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as tvf
from PIL import Image
from args import args, flags
import math

class getGT():
    #file_root: train.txt, eq
    def __init__(self, file_root, base_root_mask):
        self.file_root = file_root
        self.mask_path = []
        self.scale = [0.75, 1.0, 1.25, 1.5, 1.75] #data_augmentation

        with open(file_root, 'r') as f:
            for line in f:
                pic_root_mask = base_root_mask + '/' + line.split('\n')[0] + '.png'
                self.mask_path.append(pic_root_mask)

    def getitem(self):
        for item in range(len(self.mask_path)):
            mask_origin = Image.open(self.mask_path[item])
            mask_origin = np.array(mask_origin)
            mask_origin[mask_origin == 255] = 255
            mask_origin[mask_origin != 255] = 0
            mask_origin = Image.fromarray(mask_origin)
            name = self.mask_path[item].split('/')[-1]
            new_path = '/home/zhangcb/Desktop/VOCtrainval/VOCdevkit/VOC2012/SegmentationObjectContour_gt/'+name
            mask_origin.save(new_path)


getgt = getGT('/home/zhangcb/Desktop/resnet50/data/val.txt', flags.base_root_mask)
getgt.getitem()






