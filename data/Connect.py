import os
 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
 
import cv2
from PIL import ImageDraw, Image, ImageFont
 
from pyecharts.charts import Bar, Pie, Grid
from pyecharts import options as opts
 
import torch
from torchvision.transforms import transforms
 
from Crop import toYolo, save_txt, computeSlicePosition, slidingWindowCrop
 
 
# 取顔色
def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
 
    return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)
 
 
# 对子图预测的重叠部分进行非极大值抑制
def NMS(boxes, scores, iou_thres, GIoU=False, DIoU=False, CIoU=False):
    """
    :param boxes:  (Tensor[N, 4])): are expected to be in ``(x1, y1, x2, y2)
    :param scores: (Tensor[N]): scores for each one of the boxes
    :param iou_thres: discards all overlapping boxes with IoU > iou_threshold
    :return:keep (Tensor): int64 tensor with the indices
            of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    """
    # 按conf从大到小排序
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []
    repeat = []
    while B.numel() > 0:
        # 取出置信度最高的
        index = B[0]
        keep.append(index.tolist())
        if B.numel() == 1: break
        # 计算iou,根据需求可选择GIOU,DIOU,CIOU
        iou = bbox_iou(boxes[index, :], boxes[B[1:], :], GIoU=GIoU, DIoU=DIoU, CIoU=CIoU)
        # 找到符合阈值的下标
        inds = torch.nonzero(iou <= iou_thres).reshape(-1)
        no_inds = torch.nonzero(iou > iou_thres).reshape(-1)
        for i in no_inds:
            repeat.append((B[0], B[i + 1]))
        B = B[inds + 1]
    # repeat存储每个匹配对，即保存的与删除的对，用于存在两个子图框都不为完整时（由于原图框较大而重叠率较小时可能会导致）融合得到完整框
    return keep, repeat
 
 
# 计算iou
def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
 
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
 
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
 
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
 
    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / np.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU
 
 
# 将子图坐标信息转换到原图上的坐标信息
def transAnchor(boxes, subimage_position, nrow_ncol, Width_Height, windowSize, stepLength):
    img_row, img_col = subimage_position  # 位置索引
    nrow, ncol = nrow_ncol
    sliceWidth, sliceHeight = windowSize
    Width, Height = Width_Height
    dx, dy = stepLength
 
    if img_col == ncol - 1:  # 如果为最后一列
        boxes.iloc[:, [1, 3]] += (Width - sliceWidth)
    else:
        boxes.iloc[:, [1, 3]] += img_col * dx
 
    if img_row == nrow - 1:  # 如果为最后一行
        boxes.iloc[:, [2, 4]] += (Height - sliceHeight)
    else:
        boxes.iloc[:, [2, 4]] += img_row * dy
 
    return boxes
 
 
# 将一个框绘制在原图上
def plot_one_box_PIL(box, img, color=None, label=None, line_thickness=None):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
    draw.rectangle(box[:4], width=line_thickness, outline=tuple(color))  # plot
    confidence = box[4] if len(box) == 5 else None
    label = label + ' ' + str(confidence) if confidence else label
    if label:
        fontsize = 60
        # font = ImageFont.truetype("font/simsun.ttc", fontsize, encoding="utf-8")
        # txt_width, txt_height = font.getsize(label)
        # draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=tuple(color))
        # draw.text((box[0], box[1] - txt_height + 1), label, fill=(255, 255, 255), font=font)
    return np.asarray(img)
 
 
# 将所有目标框绘制在原图上,name:结果展示窗口的名字，label_names:每个标签代表的含义，为字典存储
def plot_boxes(img, labels, name='image', label_names=None, show=False, wait=100):
    colors = color_list()
    for cls, *box in labels:
        color = colors[int(cls) % len(colors)]
        img = plot_one_box_PIL(box, img, color=color, label=label_names[int(cls)] if label_names else None)
    if show:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 1368, 912)
        cv2.moveWindow(name, 0, 0)
        cv2.imshow(name, img)
        # 一定要加这一句，否则图片会一闪而过
        cv2.waitKey(wait)
 
    return img
 
 
# 获得原图目标框数据
def getAnchor(image, image_name, txt_dir_path, GIoU=False, DIoU=False, CIoU=False, sliceWidth=2200, sliceHeight=1900,
              overlap=0.5, iou_thres=0.35):
    dx, dy = int(sliceWidth * (1 - overlap)), int(sliceHeight * (1 - overlap))
    Width, Height = image.shape[1], image.shape[0]
    txt_data = pd.DataFrame()
    # 计算切图有几行几列
    sub_img_row_nums = len(computeSlicePosition(Height, sliceHeight, overlap))
    sub_img_col_nums = len(computeSlicePosition(Width, sliceWidth, overlap))
 
    for row in range(0, sub_img_row_nums):
        for col in range(0, sub_img_col_nums):
            sub_path = os.path.join(txt_dir_path, image_name.split('.')[0] + f'_{row}' + f'_{col}' + '.txt')
            try:
                sub_txt = pd.read_table(sub_path, header=None, sep=' ')
                sub_txt.loc[:, [1, 2]], sub_txt.loc[:, [3, 4]] = (sub_txt.loc[:, [1, 2]].values - (
                        sub_txt.loc[:, [3, 4]] / 2).values) * np.array([sliceWidth, sliceHeight]), \
                                                                 (sub_txt.loc[:, [1, 2]].values + (sub_txt.loc[:, [3,
                                                                                                                   4]] / 2).values) * np.array(
                                                                     [sliceWidth, sliceHeight])
 
                # 将相对于子图上的坐标转换为相对于原图上的坐标
                sub_txt = transAnchor(sub_txt, (row, col), (sub_img_row_nums, sub_img_col_nums),
                                      (Width, Height), (sliceWidth, sliceHeight), (dx, dy))  # 子图位置、子图序号、分割图的步幅
 
                txt_data = pd.concat((txt_data, sub_txt))
            except:
                pass
 
    txt_data = torch.from_numpy(txt_data.values)
    # 若没有置信度则以框的面积作为score
    if len(txt_data) > 0:
        if len(txt_data[0]) == 6:
            score = txt_data[:, -1]
        else:
            score = (txt_data[:, 3] - txt_data[:, 1]) * (txt_data[:, 4] - txt_data[:, 2])
    else:
        score = torch.tensor([])
    keep, repeat_boxes = NMS(txt_data[:, 1:5] if len(txt_data) > 0 else torch.tensor([]), score, \
                             iou_thres=iou_thres, GIoU=GIoU, DIoU=DIoU, CIoU=CIoU)
    # 根据留下的框以及去除的相似框的两个框坐标更新留下的框坐标
    for keep_box, delete_box in repeat_boxes:
        x1min, y1min, x1max, y1max = txt_data[keep_box][1:5]
        x2min, y2min, x2max, y2max = txt_data[delete_box][1:5]
        score1 = txt_data[keep_box][5]
        score2 = txt_data[delete_box][5]
        x_pos = np.sort([x1min, x1max, x2min, x2max])
        y_pos = np.sort([y1min, y1max, y2min, y2max])
        txt_data[keep_box][1:6] = torch.tensor([x_pos[0], y_pos[0], x_pos[3], y_pos[3], max(score1, score2)])
    #     print(image_name+f' 抑制重复框数量：{len(txt_data)-len(keep)}')
    #     print(txt_data[keep].numpy())
 
    return txt_data[keep].numpy()
 
 
# 保存单张结合图
def save_connect_image(save_image_path, image, labels, resize=(640, 640), only_save_have_box=False):
    if only_save_have_box:
        if len(labels) != 0:
            transforms.ToPILImage()(image[:, :, [2, 1, 0]]).resize(resize).save(save_image_path)
 
    else:
        transforms.ToPILImage()(image[:, :, [2, 1, 0]]).resize(resize).save(save_image_path)
    if not only_save_have_box or len(labels) != 0:
        print('save:', save_image_path)
 
 
# 保存单个结合图的目标框数据txt文件
def save_connect_txt(save_txt_path, image, labels, coordinates='x1y1x2y2'):
    if coordinates in ['yolo', 'x1y1x2y2']:
        pass
    else:
        raise Exception('coordinates参数需指定yolo或x1y1x2y2之一')
    #  防止原来存在txt导致写入出错
 
    if os.path.exists(save_txt_path):
        os.remove(save_txt_path)
    for label in labels:
        #         print('label',label)
        if len(label) == 6:
            data = label[0:5].tolist()
            data.append(label[5])
        else:
            data = label[0:5].tolist()
        data[:5] = toYolo(data, image.shape[1], image.shape[0]) if coordinates == 'yolo' else data[:5]
 
        save_txt(save_txt_path, data, )
    if len(labels) != 0:
        print('save:', save_txt_path)
 
 
# 连接单张图并保存
def saveConnect(image_path, image_name, windowSize, rowcol, overlap, iou_thres, label_names, test_labels_path,
                window_name,
                wait, save_imagedir_path, save_labeldir_path, resize, show, coordinates, only_save_have_box):
    print(image_path)
    image = cv2.imread(image_path)
    Width, Height = image.shape[1], image.shape[0]
    if rowcol:
        sliceWidth, sliceHeight = slidingWindowCrop.ranksGetWindowSize(rowcol, (Width, Height), overlap)
    else:
        sliceWidth, sliceHeight = windowSize
    labels = getAnchor(image, image_name, test_labels_path, sliceWidth=sliceWidth, sliceHeight=sliceHeight,
                       overlap=overlap, iou_thres=iou_thres)
    labels[:6] = np.round(labels[:6])
    image = plot_boxes(image, labels, name=window_name, label_names=label_names, show=show, wait=wait)
 
    if save_imagedir_path:
        filename = os.path.join(save_imagedir_path, image_name)
        resize = (Width, Height) if not resize else resize
        save_connect_image(filename, image, labels, resize, only_save_have_box)
 
    if save_labeldir_path:
        filename = os.path.join(save_labeldir_path, image_name.split('.')[0] + '.txt')
        save_connect_txt(filename, image, labels, coordinates)
 
    return labels
 
 
# 保存连接图，可对单张图片或文件夹中所有图片进行连接并展示
def connectImage(test_data_path, test_labels_path, save_imagedir_path=None, save_labeldir_path=None,
                 windowSize=(2200, 1900), overlap=0.5, iou_thres=0.3, coordinates='x1y1x2y2',
                 resize=None, label_names=None, only_save_have_box=True, show=True, rowcol=None):
    if os.path.isdir(test_data_path):
        test_images_name = os.listdir(test_data_path)
        for image_name in test_images_name:
            image_path = os.path.join(test_data_path, image_name)
            saveConnect(image_path, image_name, windowSize, rowcol, overlap, iou_thres, label_names, test_labels_path,
                        'image', 50,
                        save_imagedir_path, save_labeldir_path, resize, show, coordinates, only_save_have_box)
    else:
        #         print(image)
        image_name = test_data_path.split('/')[-1]
        labels = saveConnect(test_data_path, image_name, windowSize, rowcol, overlap, iou_thres, label_names,
                             test_labels_path, image_name, 0,
                             save_imagedir_path, save_labeldir_path, resize, show, coordinates, only_save_have_box)
 
    cv2.destroyAllWindows()
    # 如果labels不为空且为检测一张图片时，返回单张图的目标框数据
    if not os.path.isdir(test_data_path):
        return labels
 
 
class Connect():
    def __init__(self, overlap, iou_thres, windowSize=None, rowcol=None, label_names=None):
 
        if not ((windowSize or rowcol) and not (windowSize and rowcol)):
            raise Exception('windowSize and rowcol must Only one can be defined')
        self.windowSize = windowSize  # (width, height)
        self.rowcol = rowcol
        self.overlap = overlap
        self.iou_thres = iou_thres
        self.label_names = label_names
 
    def showConnectImage(self, imagePath, txtDirPath, notebook=False, title=None, subtitle=None):
        image = cv2.imread(imagePath)
        image_name = imagePath.split('/')[-1]
        Width, Height = image.shape[1], image.shape[0]
 
        if self.rowcol:
            windowSize = slidingWindowCrop.ranksGetWindowSize(self.rowcol, (Width, Height), self.overlap)
        else:
            windowSize = self.windowSize
        labels = getAnchor(image, image_name, txtDirPath, sliceWidth=windowSize[0], sliceHeight=windowSize[1],
                           overlap=self.overlap, iou_thres=self.iou_thres, )
 
        plot_boxes(image, labels, name=image_name, label_names=self.label_names, show=True, wait=0)
 
        return visualAnalysis(labels[:, 0].tolist(), label_names=self.label_names, notebook=notebook, title=title,
                              subtitle=subtitle)
 
    def saveConnectImageAndTxt(self, imagePath, txtDirPath, savePath, coordinates='yolo',
                               resize=None, show=False, only_save_have_box=False):
        if savePath:
            # 如果不存在文件夾则创建
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            imagesfile = os.path.join(savePath, 'images')
            labelsfile = os.path.join(savePath, 'labels')
 
            # 如果文件夹不存在则创建
            if not os.path.exists(labelsfile):
                os.makedirs(labelsfile)
            if not os.path.exists(imagesfile):
                os.makedirs(imagesfile)
 
        connectImage(imagePath, txtDirPath, save_imagedir_path=imagesfile, save_labeldir_path=labelsfile,
                     windowSize=self.windowSize, overlap=self.overlap, iou_thres=self.iou_thres, resize=resize,
                     show=show, label_names=self.label_names, only_save_have_box=only_save_have_box,
                     coordinates=coordinates, rowcol=self.rowcol)
 
    def saveConnectImage(self, imagePath, txtDirPath, saveImagePath,
                         resize=None, show=False, only_save_have_box=False):
        # 如果文件夹不存在则创建
        if not os.path.exists(saveImagePath):
            os.makedirs(saveImagePath)
 
        connectImage(imagePath, txtDirPath, save_imagedir_path=saveImagePath,
                     windowSize=self.windowSize, overlap=self.overlap, iou_thres=self.iou_thres,
                     resize=resize, show=show, label_names=self.label_names,
                     only_save_have_box=only_save_have_box, rowcol=self.rowcol)
 
    def saveConnectTxt(self, imagePath, txtDirPath, saveTxtPath, coordinates='yolo', show=False, ):
        # 如果文件夹不存在则创建
        if not os.path.exists(saveTxtPath):
            os.makedirs(saveTxtPath)
 
        connectImage(imagePath, txtDirPath, save_labeldir_path=saveTxtPath,
                     windowSize=self.windowSize, overlap=self.overlap, iou_thres=self.iou_thres,
                     show=show, label_names=self.label_names, coordinates=coordinates, rowcol=self.rowcol)
 
 
# 可视化分析
def visualAnalysis(labels, label_names=None, title=None, subtitle=None, notebook=False):
    """传入图片名字（作为标题）、标签"""
    data = pd.DataFrame(labels)
    # 训练标签转换为虫子名称
    data.loc[:, 0] = list(map(lambda x: label_names[x] if label_names else x, data.loc[:, 0]))
    # 统计数量
    data = data.groupby(data.loc[:, 0]).size()
    data_x, data_y = data.index.tolist(), data.values.tolist()
 
    bar = Bar()
    bar.add_xaxis(data_x)
    bar.add_yaxis('数量', data_y, category_gap="70%", )
    bar.set_global_opts(title_opts=opts.TitleOpts(title=title, subtitle=subtitle),
                        legend_opts=opts.LegendOpts(is_show=False, pos_bottom=0, pos_left=0))
    bar.set_series_opts(label_opts=opts.LabelOpts(position='top'))  # 水平直方图时position指定right
 
    pie = Pie()
    pie.add('数量', [x for x in zip(data_x, data_y)], radius=['30%', '48%'], rosetype='radius', center=['72%', '58%'])
    # rosetype可选area,radius,None
    pie.set_global_opts(title_opts=opts.TitleOpts(title=title),
                        legend_opts=opts.LegendOpts(pos_top='15%', pos_left='50%'))
    # formatter中 a表示data_pair,b表示类别名，c表示类别数量,d表示百分数
    pie.set_series_opts(label_opts=opts.LabelOpts(formatter='{b}:{d}%\n数量:{c}', position='outside'))
 
    grid = Grid(init_opts=opts.InitOpts(width='900px', height='550px'))
    grid.add(bar, grid_opts=opts.GridOpts(pos_left='0%', pos_right='57%', pos_top='20%', pos_bottom='20%'))
    grid.add(pie, grid_opts=opts.GridOpts(pos_left='75%', pos_right='70%', ))
    if notebook:
        return grid.render_notebook()
    else:
        return grid.render()
# 以下为作者测试所用，可复制修改参数为自己的文件路径等
if __name__ == '__main__':
    bug = {0: '大螟', 1: '二化螟', 2: '稻纵卷叶螟', 3: '白背飞虱', 4: '褐飞虱属', 5: '地老虎', 6: '蝼蛄', 7: '粘虫',
           8: '草地螟', 9: '甜菜夜蛾', 10: '黄足猎蝽', 11: '八点灰灯蛾', 12: '棉铃虫', 13: '二点委夜蛾', 14: '甘蓝夜蛾',
           15: '蟋蟀', 16: '黄毒蛾', 17: '稻螟蛉', 18: '紫条尺蛾', 19: '水螟蛾', 20: '线委夜蛾', 21: '甜菜白带野螟', 22: '歧角螟',
           23: '瓜绢野螟', 24: '豆野螟', 25: '石蛾', 26: '大黑鳃金龟', 27: '干纹冬夜蛾'}
    connect = Connect(rowcol=(4, 4), overlap=0.5, iou_thres=0.35, label_names=bug)
    connect = Connect(windowSize=(2200, 1900), overlap=0.5, iou_thres=0.35, label_names=bug)
    render_html = connect.showConnectImage('images/00295.png', 'la', )
    print('浏览器打开：', render_html)
 
    connect.saveConnectImageAndTxt('images', 'la', 'Result', show=True, coordinates='yolo', resize=None,
                             only_save_have_box=False)
