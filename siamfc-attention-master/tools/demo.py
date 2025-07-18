from __future__ import absolute_import

import cv2
import os
import glob
import numpy as np

from siamfc import TrackerSiamFC


if __name__ == '__main__':
    seq_dir = 'E:\cxy\siamfc-pytorch-master\data\OTB100\CarScale/'
    img_files = sorted(glob.glob(os.path.join(seq_dir, 'img\*.jpg')))
    anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt', delimiter=',')
    #anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt', delimiter=None)

    net_path = r'E:\cxy\siamfc-pytorch-master\pretrained_weight_all\siamfc_alexnet_e50.pth'
    tracker = TrackerSiamFC(net_path=net_path)
    tracker.track(img_files, anno[0], visualize=True)

    # output_dir = 'E:/cxy/siamfc-pytorch-master/output_images'
    # os.makedirs(output_dir, exist_ok=True)
    #
    # # 跟踪每一帧
    # for i, img_file in enumerate(img_files):
    #     img = cv2.imread(img_file)
    #
    #     # 获取每一帧的目标框
    #     box = anno[i]
    #
    #     # 在图像上绘制边界框
    #     x, y, w, h = box
    #     img = cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
    #
    #     # 保存处理后的图像
    #     output_img_path = os.path.join(output_dir, f'frame_{i + 1}.jpg')
    #     cv2.imwrite(output_img_path, img)
    #
    # print(f"Images have been saved to {output_dir}")
