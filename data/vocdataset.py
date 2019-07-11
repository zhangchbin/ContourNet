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

class dataset(data.Dataset):
    #file_root: train.txt, eq
    #mode = 'train', 'val', 'predict'
    #base_root_img or base_root_mask: imags' root
    def __init__(self, file_root, base_root_img, base_root_mask, mode):
        self.file_root = file_root
        self.img_path = []
        self.mask_path = []
        self.mode = mode
        self.scale = [0.75, 1.0, 1.25, 1.5, 1.75] #data_augmentation

        with open(file_root, 'r') as f:
            for line in f:
                pic_root_img = base_root_img + '/' +line.split('\n')[0] + '.jpg'
                self.img_path.append(pic_root_img)
                pic_root_mask = base_root_mask + '/' + line.split('\n')[0] + '.png'
                self.mask_path.append(pic_root_mask)

    def transform(self, image_origin, mask_origin):
        image_res, mask_res = None, None

        totensor_op = transforms.ToTensor()
        color_op = transforms.ColorJitter(0.1, 0.1, 0.1)
        resize_op = transforms.Resize((224, 224))
        image_origin = color_op(image_origin)

        if flags.mode == 'val' or flags.mode == 'predict':
            image_res = totensor_op(image_origin)
            mask_res = totensor_op(mask_origin)
        elif flags.mode == 'train':
            if flags.data_augmentation == 'multiscale':
                pass
            elif flags.data_augmentation == 'fivecrop':
                pass

            elif flags.data_augmentation == 'randomcrop':
                if image_origin.size[0] < 224 or image_origin.size[1] < 224:
                    #padding-val:
                    val = int(np.array(image_origin).sum() / image_origin.size[0] / image_origin.size[1])
                    padding_width = 224-min(image_origin.size[0],image_origin.size[1])
                    padding_op = transforms.Pad(padding_width,fill=val)
                    image_origin = padding_op(image_origin)
                    padding_op = transforms.Pad(padding_width, fill=0)
                    mask_origin = padding_op(mask_origin)
                i, j, h, w = transforms.RandomCrop.get_params(
                    image_origin, output_size=(224, 224)
                )
                image_res = totensor_op(tvf.crop(image_origin, i, j, h, w))
                mask_res = totensor_op(tvf.crop(mask_origin, i, j, h, w))

            elif flags.data_augmentation == 'resize':
                image_res = totensor_op(resize_op(image_origin))
                mask_res = totensor_op(resize_op(mask_origin))

        return image_res, mask_res

    def __getitem__(self, item):
        image_origin = Image.open(self.img_path[item])
        mask_origin = Image.open(self.mask_path[item])

        '''image_origin = np.array(image_origin)
        mask_origin = np.array(mask_origin)
        mask_origin[mask_origin == 255] = 1
        mask_origin[mask_origin != 255] = 0
        image_origin = Image.fromarray(image_origin)
        mask_origin = Image.fromarray(mask_origin)
        '''
        image_res, mask_res = self.transform(image_origin, mask_origin)
        mask_res[mask_res >= 0.98] = 1
        mask_res[mask_res < 0.98] = 0

        return image_res, mask_res, self.img_path[item]

    def __len__(self):
        return len(self.img_path)






