import torch
import os
import numpy as np
import xml.etree.ElementTree as ET
import shutil
import cv2 as cv


def GetFileList(dir, fileList):
    newDir = dir
    if os.path.isfile(dir):
        last = os.path.splitext(dir)[1]
        if last == '.jpg' or last == '.xml' or last == '.txt':
            fileList.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码
            # if s == "xxx":
            # continue
            newDir = os.path.join(dir, s)
            GetFileList(newDir, fileList)
    return fileList


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = w * (x[0] - x[2] / 2) + padw  # top left x
    y[1] = h * (x[1] - x[3] / 2) + padh  # top left y
    y[2] = w * (x[0] + x[2] / 2) + padw  # bottom right x
    y[3] = h * (x[1] + x[3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(size, box):  # 从xml改过来的，改几个位置就行
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


fileDir = "/home/shuai/HCX/yolo5/mydata/images/"  # 原图片路径
label_path = "/home/shuai/HCX/yolo5/mydata/labels/"  # 原label的路径
list1 = GetFileList(fileDir, [])

image_save_path_head = "/home/shuai/HCX/yolo5/mydata1/images/"  # 分割后有标注图片储存路径
image_save_path_tail = ".jpg"
label_save_path_head = "/home/shuai/HCX/yolo5/mydata1/labels/"  # 标签储存路径
label_save_path_tail = ".txt"
for i in list1:
    img = cv.imread(i)
    shape = img.shape
    seq = 1
    basename = os.path.basename(i)
    basename = os.path.splitext(basename)[0]
    labelname = label_path + basename + '.txt'  # 找到对应图片的label
    # for i in range(2):
    #     for j in range(2):
    #         new_name=basename+'_'+str(seq)
    #         img_roi = img[(i * 1024):((i + 1) * 1024), (j * 1224):((j + 1) * 1224)] #imread 格式是（h，w，c）
    #         image_save_path = "%s%s%s" % (image_save_path_head, new_name, image_save_path_tail)  ##·将整数和字符串连接在一起
    #         cv.imwrite(image_save_path, img_roi)
    #         seq = seq + 1
    pos = []
    with open(labelname, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if lines == '\n':
                lines = None
            if not lines:
                break

            p_tmp = [float(i) for i in lines.split()]
            pos.append(p_tmp)
        # label=pos.copy()  #保留xywh坐标lable
        # pos = np.array(pos)
        # if pos.size:
        #     pos[:, 1:]=xywhn2xyxy(pos[:,1:],shape[1],shape[0])#转成xyxy

        pos = np.array(pos)
        if pos.size:
            pos[:, 3:] = pos[:, 3:] * 2
        for k in pos:  # xywh ,遍历label
            k = np.array(k)

            if k[1] <= 0.5 and k[2] <= 0.5:  # 左上
                img_roi = img[0:shape[0] // 2, 0: shape[1] // 2]
                image_save_path = "%s%s%s" % (image_save_path_head, basename + '_1', image_save_path_tail)
                cv.imwrite(image_save_path, img_roi)
                label_save_path = "%s%s%s" % (label_save_path_head, basename + '_1', label_save_path_tail)
                # padw=0
                # padh=0
                # k[1:] = xywhn2xyxy(k[1:], shape[1], shape[0], padw, padh)

                # np.clip(k[1:], 0, max(shape[0], shape[1]), k[1:])
                # k[1:]=xyxy2xywhn((shape[1],shape[0]),k[1:])
                k[1] = k[1] * 2
                k[2] = k[2] * 2

                f = open(label_save_path, 'a')
                f.write(str(k[0]) + " " + " ".join([str(a) for a in k[1:]]) + '\n')
                f.close()
            elif k[1] >= 0.5 and k[2] <= 0.5:  # 右上
                img_roi = img[0:shape[0] // 2, shape[1] // 2:shape[1]]
                image_save_path = "%s%s%s" % (image_save_path_head, basename + '_2', image_save_path_tail)
                cv.imwrite(image_save_path, img_roi)
                label_save_path = "%s%s%s" % (label_save_path_head, basename + '_2', label_save_path_tail)
                k[1] = 2 * k[1] - 1
                k[2] = k[2] * 2
                # padw=-shape[1]//2
                # padh=0
                # k[1:]=xywhn2xyxy(k[1:],shape[1],shape[0],padw,padh)

                # np.clip(k[1:], 0, max(shape[0],shape[1]), k[1:])
                # k[1:] = xyxy2xywhn((shape[1], shape[0]), k[1:])
                f = open(label_save_path, 'a')
                f.write(str(k[0]) + " " + " ".join([str(a) for a in k[1:]]) + '\n')
                f.close()
            elif k[1] <= 0.5 and k[2] >= 0.5:  # 左下
                img_roi = img[shape[0] // 2:shape[0], 0:shape[1] // 2]
                image_save_path = "%s%s%s" % (image_save_path_head, basename + '_3', image_save_path_tail)
                cv.imwrite(image_save_path, img_roi)
                label_save_path = "%s%s%s" % (label_save_path_head, basename + '_3', label_save_path_tail)
                k[1] = k[1] * 2
                k[2] = 2 * k[2] - 1
                # padw=0
                # padh=shape[0]//2
                # k[1:] = xywhn2xyxy(k[1:], shape[1], shape[0], padw, padh)

                # np.clip(k[1:], 0, max(shape[0], shape[1]), k[1:])
                # k[1:] = xyxy2xywhn((shape[1], shape[0]), k[1:])
                f = open(label_save_path, 'a')
                f.write(str(k[0]) + " " + " ".join([str(a) for a in k[1:]]) + '\n')
                f.close()
            elif k[1] >= 0.5 and k[2] >= 0.5:
                img_roi = img[shape[0] // 2:shape[0], shape[1] // 2:shape[1]]
                image_save_path = "%s%s%s" % (image_save_path_head, basename + '_4', image_save_path_tail)
                cv.imwrite(image_save_path, img_roi)
                label_save_path = "%s%s%s" % (label_save_path_head, basename + '_4', label_save_path_tail)
                k[1] = 2 * k[1] - 1
                k[2] = 2 * k[2] - 1
                # padw=-shape[1]//2
                # padh=-shape[0]//2
                # k[1:] = xywhn2xyxy(k[1:], shape[1], shape[0], padw, padh)

                # np.clip(k[1:], 0, max(shape[0], shape[1]), k[1:])
                # k[1:] = xyxy2xywhn((shape[1], shape[0]), k[1:])
                f = open(label_save_path, 'a')
                f.write(str(k[0]) + " " + " ".join([str(a) for a in k[1:]]) + '\n')
                f.close()