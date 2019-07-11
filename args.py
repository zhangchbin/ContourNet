import argparse
import sys

class Args():
    def __init__(self):
        self.parse = argparse.ArgumentParser()
        self.parse.add_argument('--device', type=str, default='cuda:0', help='device,default is cuda:0')
        self.parse.add_argument('--learning_rate', type=float, default=1e-4, help='learning_rate')
        self.parse.add_argument('--epoches', type=int, default=100, help='epoches')
        self.parse.add_argument('--batch_size',
                                type=int,
                                default=64,
                                help='batch_size')
        self.parse.add_argument('--file_root', type=str,
                                default='/home/zhangcb/Desktop/resnet50v2/data/train.txt',
                                help='train.txt')
        self.parse.add_argument('--base_root_img', type=str,
                                default='/home/zhangcb/Desktop/VOCpreprocessed/PASCALContourData/JPEGImages',
                                help='the parent root of image')
        self.parse.add_argument('--base_root_mask', type=str,
                                default='/home/zhangcb/Desktop/VOCpreprocessed/PASCALContourData/groundTruth',
                                help='the parent root of mask')
        self.parse.add_argument('--mode', type=str, default='train',
                                help='train or val or predict')
        self.parse.add_argument('--data_augmentation',
                                type=str,
                                default='randomcrop',
                                help='multiscale or randomcrop or fivecrop or resize')
        self.parse.add_argument('--randomcrop_size',
                                type=int,
                                default=224,
                                help='when use the dataaugmentation'
                                     'of randomcrop')
        self.parse.add_argument('--checkpoint_name',
                                type=str,
                                default='final',
                                help='pretrained_model_name')

args = Args()
flags, _ = args.parse.parse_known_args(sys.argv[1:])
