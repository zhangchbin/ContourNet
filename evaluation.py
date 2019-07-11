import argparse
import sys
from PIL import Image
import numpy as np
import torch
import math

parser = argparse.ArgumentParser()
parser.add_argument('--gtdir', type=str,
                    default='/home/zhangcb/Desktop/VOCtrainval/VOCdevkit/VOC2012/SegmentationObjectContour_gt',
                    help='groundTruthDir')
parser.add_argument('--resultdir', type=str,
                    default='/home/zhangcb/Desktop/resnet50/result/04m_09d_21h/predict',
                    help='resultDir')
parser.add_argument('--txt', type=str,
                    default='/home/zhangcb/Desktop/resnet50/data/val.txt',
                    help='val.txt')
args, _ = parser.parse_known_args(sys.argv[1:])
file_root = args.txt
base_root_result = args.resultdir
base_root_mask = args.gtdir

def compute_fusion_matrix(match_result, gt, threshold):
    result = torch.tensor(np.array(match_result)).to('cuda:0')
    gt = torch.tensor(np.array(gt)).to('cuda:0')
    result[result < threshold] = 0
    result[result >= threshold] = 1

    gt[gt != 255] = 0
    gt[gt == 255] = 1
    tmp1 = result - gt
    tmp2 = result + gt
    TP = (tmp2 == 2).sum()
    TN = (tmp2 == 0).sum()
    FP = (tmp1 == 1).sum()
    FN = (tmp1 == -1).sum()
    return TP.item(), FP.item(), TN.item(), FN.item()



def main():
    precision = np.zeros(256)
    recall = np.zeros(256)

    result_path = []
    mask_path = []
    with open(file_root, 'r') as f:
        for line in f:
            pic_root_img = base_root_result + '/' + line.split('\n')[0] + '.jpg'
            result_path.append(pic_root_img)
            pic_root_mask = base_root_mask + '/' + line.split('\n')[0] + '.png'
            mask_path.append(pic_root_mask)
    for i in range(len(mask_path)):
        #result_path[i] = base_root_result + '/' + '2007_000129.jpg'
        #mask_path[i] = base_root_mask + '/' + '2007_000129.png'
        result = np.array(Image.open(result_path[i]))
        mask = np.array(Image.open(mask_path[i]))
        #result = NMS(result)

        for threshold in range(256):
            TP, FP, TN, FN = compute_fusion_matrix(result, mask, threshold)
            if TP + FP == 0:
                p = 0
            else:
                p = TP / (TP + FP)
            if TP + FN == 0:
                r = 0
            else:
                r = TP / (TP + FN)
            precision[threshold] += p
            recall[threshold] += r
    precision /= len(mask_path)
    recall /= len(mask_path)
    return precision, recall


# 计算梯度幅值
def gradients(new_gray):
    """
    :type: image which after smooth
    :rtype:
        dx: gradient in the x direction
        dy: gradient in the y direction
        M: gradient magnitude
        theta: gradient direction
    """

    W, H = new_gray.shape
    dx = np.zeros([W - 1, H - 1])
    dy = np.zeros([W - 1, H - 1])
    M = np.zeros([W - 1, H - 1])
    theta = np.zeros([W - 1, H - 1])

    for i in range(W - 1):
        for j in range(H - 1):
            dx[i, j] = new_gray[i + 1, j] - new_gray[i, j]
            dy[i, j] = new_gray[i, j + 1] - new_gray[i, j]
            # 图像梯度幅值作为图像强度值
            M[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))
            # 计算  θ - artan(dx/dy)
            theta[i, j] = math.atan(dx[i, j] / (dy[i, j] + 0.000000001))

    return dx, dy, M, theta

def NMS(new_gary):
    dx, dy, M, theta = gradients(new_gary)
    d = np.copy(M)
    W, H = M.shape
    NMS = np.copy(d)
    NMS[0, :] = NMS[W - 1, :] = NMS[:, 0] = NMS[:, H - 1] = 0

    for i in range(1, W - 1):
        for j in range(1, H - 1):

            # 如果当前梯度为0，该点就不是边缘点
            if M[i, j] == 0:
                NMS[i, j] = 0

            else:
                gradX = dx[i, j]  # 当前点 x 方向导数
                gradY = dy[i, j]  # 当前点 y 方向导数
                gradTemp = d[i, j]  # 当前梯度点

                # 如果 y 方向梯度值比较大，说明导数方向趋向于 y 分量
                if np.abs(gradY) > np.abs(gradX):
                    weight = np.abs(gradX) / np.abs(gradY)  # 权重
                    grad2 = d[i - 1, j]
                    grad4 = d[i + 1, j]

                    # 如果 x, y 方向导数符号一致
                    # 像素点位置关系
                    # g1 g2
                    #    c
                    #    g4 g3
                    if gradX * gradY > 0:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]

                    # 如果 x，y 方向导数符号相反
                    # 像素点位置关系
                    #    g2 g1
                    #    c
                    # g3 g4
                    else:
                        grad1 = d[i - 1, j + 1]
                        grad3 = d[i + 1, j - 1]

                # 如果 x 方向梯度值比较大
                else:
                    weight = np.abs(gradY) / np.abs(gradX)
                    grad2 = d[i, j - 1]
                    grad4 = d[i, j + 1]

                    # 如果 x, y 方向导数符号一致
                    # 像素点位置关系
                    #      g3
                    # g2 c g4
                    # g1
                    if gradX * gradY > 0:

                        grad1 = d[i + 1, j - 1]
                        grad3 = d[i - 1, j + 1]

                    # 如果 x，y 方向导数符号相反
                    # 像素点位置关系
                    # g1
                    # g2 c g4
                    #      g3
                    else:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]

                # 利用 grad1-grad4 对梯度进行插值
                gradTemp1 = weight * grad1 + (1 - weight) * grad2
                gradTemp2 = weight * grad3 + (1 - weight) * grad4

                # 当前像素的梯度是局部的最大值，可能是边缘点
                if gradTemp >= gradTemp1 and gradTemp >= gradTemp2:
                    NMS[i, j] = gradTemp

                else:
                    # 不可能是边缘点
                    NMS[i, j] = 0

    return NMS

if __name__ == '__main__':
    precision, recall = main()
    precision, recall = np.array(precision), np.array(recall)
    np.save('/home/zhangcb/Desktop/resnet50/util/precision.npy', precision)
    np.save('/home/zhangcb/Desktop/resnet50/util/recall.npy', recall)
    Fscore = np.true_divide(2 * np.multiply(precision, recall), precision + recall)
    arg_threshold = np.argmax(Fscore)
    Fscore = Fscore.max()
    print('max F-score = ', Fscore, 'threshold = ', arg_threshold)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.ylim(0, 1.1)
    plt.xlim(0, 1.1)
    plt.ylabel('precision')
    plt.xlabel('recall')
    plt.plot(recall, precision)
    plt.show()
    #print(precision, recall)