import os
 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
 
import random
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
plt.rcParams['font.family'] = 'SimHei'  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
 
from PIL import Image
 
from torch.utils.data.dataset import Dataset
 
from torchvision.transforms import transforms
 
 
def exist_objs(list_1, list_2, new_box_iou_limit=0.35):
    '''
    list_1:当前slice的图像
    list_2:原图中的所有目标
    return:原图中位于当前slicze中的目标集合
    '''
    return_objs = []
    # 当该原图无目标框时返回空列表
    if len(list_2) == 0:
        return return_objs
 
    # 判断新框与旧框的iou是否满足限制条件，若满足则将新框保留作为子图的目标框
    def judge_iou_limit():
        new_box_area = (xmax_new - xmin_new) * (ymax_new - ymin_new)
        if new_box_area / (new_box_area + box_area) >= new_box_iou_limit:
            return_objs.append([category, xmin_new, ymin_new, xmax_new, ymax_new])
 
    s_xmin, s_ymin, s_xmax, s_ymax = list_1[0], list_1[1], list_1[2], list_1[3]
 
    for object_box in list_2:
        category, xmin, ymin, xmax, ymax = object_box[0], object_box[1], object_box[2], object_box[3], object_box[4]
        box_area = (xmax - xmin) * (ymax - ymin)
        # 1
        if s_xmin <= xmin < s_xmax and s_ymin <= ymin < s_ymax:  # 目标点的左上角在切图区域中
            if s_xmin < xmax <= s_xmax and s_ymin < ymax <= s_ymax:  # 目标点的右下角在切图区域中
                xmin_new = xmin - s_xmin
                ymin_new = ymin - s_ymin
                xmax_new = xmin_new + (xmax - xmin)
                ymax_new = ymin_new + (ymax - ymin)
                judge_iou_limit()
 
        if s_xmin <= xmin < s_xmax and ymin < s_ymin:  # 目标点的左上角在切图区域上方
            # 2
            if s_xmin < xmax <= s_xmax and s_ymin < ymax <= s_ymax:  # 目标点的右下角在切图区域中
                xmin_new = xmin - s_xmin
                ymin_new = 0
                xmax_new = xmax - s_xmin
                ymax_new = ymax - s_ymin
                judge_iou_limit()
 
            # 3
            if xmax > s_xmax and s_ymin < ymax <= s_ymax:  # 目标点的右下角在切图区域右方
                xmin_new = xmin - s_xmin
                ymin_new = 0
                xmax_new = s_xmax - s_xmin
                ymax_new = ymax - s_ymin
                judge_iou_limit()
 
        if s_ymin < ymin <= s_ymax and xmin < s_xmin:  # 目标点的左上角在切图区域左方
            # 4
            if s_xmin < xmax <= s_xmax and s_ymin < ymax <= s_ymax:  # 目标点的右下角在切图区域中
                xmin_new = 0
                ymin_new = ymin - s_ymin
                xmax_new = xmax - s_xmin
                ymax_new = ymax - s_ymin
                judge_iou_limit()
 
            # 5
            if s_xmin < xmax < s_xmax and ymax >= s_ymax:  # 目标点的右下角在切图区域下方
                xmin_new = 0
                ymin_new = ymin - s_ymin
                xmax_new = xmax - s_xmin
                ymax_new = s_ymax - s_ymin
                judge_iou_limit()
 
        # 6
        if s_xmin >= xmin and ymin <= s_ymin:  # 目标点的左上角在切图区域左上方
            if s_xmin < xmax <= s_xmax and s_ymin < ymax <= s_ymax:  # 目标点的右下角在切图区域中
                xmin_new = 0
                ymin_new = 0
                xmax_new = xmax - s_xmin
                ymax_new = ymax - s_ymin
                judge_iou_limit()
 
        # 7
        if s_xmin <= xmin < s_xmax and s_ymin <= ymin < s_ymax:  # 目标点的左上角在切图区域中
            if ymax >= s_ymax and xmax >= s_xmax:  # 目标点的右下角在切图区域右下方
                xmin_new = xmin - s_xmin
                ymin_new = ymin - s_ymin
                xmax_new = s_xmax - s_xmin
                ymax_new = s_ymax - s_ymin
                judge_iou_limit()
 
            # 8
            if s_xmin < xmax < s_xmax and ymax >= s_ymax:  # 目标点的右下角在切图区域下方
                xmin_new = xmin - s_xmin
                ymin_new = ymin - s_ymin
                xmax_new = xmax - s_xmin
                ymax_new = s_ymax - s_ymin
                judge_iou_limit()
 
            # 9
            if xmax > s_xmax and s_ymin < ymax <= s_ymax:  # 目标点的右下角在切图区域右方
                xmin_new = xmin - s_xmin
                ymin_new = ymin - s_ymin
                xmax_new = s_xmax - s_xmin
                ymax_new = ymax - s_ymin
                judge_iou_limit()
 
    return return_objs
 
 
# 通过子图宽高以及重叠率来计算得到行/列切分位置列表
def computeSlicePosition(WidthOrHeight, sliceWidthOrHeight, overlap):
    # 计算步长
    dx_or_dy = int(sliceWidthOrHeight * (1 - overlap))
    sp = np.array(range(0, WidthOrHeight, dx_or_dy))
    # 获取最终切点位置：当切点位置加上子图宽/高大于原图宽高时
    end_index = list(sp + sliceWidthOrHeight >= WidthOrHeight).index(True) + 1
 
    return sp[:end_index].tolist()
 
 
def slice_imag(image, sliceWidth=2200, sliceHeight=1900, image_name=None, object_list=[],
               overlap=0.5, new_box_iou_limit=0.4, figsize=(10, 8), imshow=False, label_names=None):
    """
        object_list:原图labels
        overlap:分割子图间的重叠部分（长、宽）
        new_box_area_limit:子图上的目标框面积限制（小于原图目标框面积指定比例则去除该框）
    """
    #     print('name:',image_name,'width:',image.shape[2],'height:',image.shape[1])
 
    n_imgs = 0  # 表示第几张子图
    slice_images = []  # 存储切分后的子图
    exiset_obj_lists = []  # 存储每个子图的目标框
 
    # 存储每行/每列的切分位置
    rangeHeight = computeSlicePosition(image.shape[1], sliceHeight, overlap)
    rangeWidth = computeSlicePosition(image.shape[2], sliceWidth, overlap)
 
    #     print(rangeHeight,rangeWidth)
 
    if imshow:
        nrow = len(rangeHeight)
        ncol = len(rangeWidth)
        print('行列数为：', nrow, ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
        axes = axes.flatten()
 
    for y0 in rangeHeight:
        for x0 in rangeWidth:
            n_imgs += 1
            if y0 + sliceHeight >= image.shape[1]:
                y = image.shape[1] - sliceHeight
            else:
                y = y0
            if x0 + sliceWidth >= image.shape[2]:
                x = image.shape[2] - sliceWidth
            else:
                x = x0
            slice_xmax = x + sliceWidth
            slice_ymax = y + sliceHeight
 
            sub_image = image[:, y:slice_ymax, x:slice_xmax]
            slice_images.append(sub_image)
 
            # 得到每个分割子图上的目标位置信息
            exiset_obj_list = exist_objs([x, y, slice_xmax, slice_ymax], object_list,
                                         new_box_iou_limit)
            #             print(exiset_obj_list)
            exiset_obj_lists.append(exiset_obj_list)
 
            if imshow:
                # 展示分割后的子图
                axes[n_imgs - 1].imshow(sub_image.permute((1, 2, 0)).numpy())
                axes[n_imgs - 1].axes.get_xaxis().set_visible(False)
                axes[n_imgs - 1].axes.get_yaxis().set_visible(False)
                # 在新的子图上展示目标框
                for category, *position in exiset_obj_list:
                    axes[n_imgs - 1].add_patch(bbox_to_rect(position, color='red'))
                    if label_names:
                        axes[n_imgs - 1].text(position[0], position[1], label_names[category], color='blue')
    if imshow:
        fig.show()
    # 返回切割后的子图，以及子图的目标框
    return slice_images, exiset_obj_lists
 
 
# 将（左上X,左上Y,右下X,右下Y）格式转换成matplotlib格式：
# ((左上X,左上Y),宽，高)
def bbox_to_rect(bbox, color):
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1], fill=False, edgecolor=color,
        linewidth=2, )
 
 
# 保存类别以及坐标信息
def save_txt(path, position_list, mode='a+'):
    with open(path, 'a+') as f:
        for i in range(len(position_list)):
            f.write(str(position_list[i]))
            if i == len(position_list) - 1:
                f.write('\n')
            else:
                f.write(' ')
 
 
# 数据集类
class MyDataset(Dataset):
    def __init__(self, images_path, transform=None):
        self.transform = transforms.Compose([
            transforms.ToTensor()  # 这里仅以最基本的为例
        ]) if not transform else transform
        self.image_path = images_path if os.path.isdir(images_path) else os.path.abspath(os.path.dirname(images_path))
        self.image_names = os.listdir(self.image_path) if os.path.isdir(images_path) else [images_path.split('/')[-1]]
 
    def __len__(self):
        return len(self.image_names)
 
    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(os.path.join(self.image_path, image_name)).convert('RGB')  # 读取到的是RGB， C, H, W
        #         print(image_name)
        image = self.transform(image)
        return image
 
    def get_name(self, index):
        image_name = self.image_names[index]
        return image_name
 
 
# 将x1y1x2y2格式转换为yolo格式
def toYolo(box, imageWidth, imageHeight):
    center_x = (box[1] + box[3]) / 2 / imageWidth
    center_y = (box[2] + box[4]) / 2 / imageHeight
    width = (box[3] - box[1]) / imageWidth
    height = (box[4] - box[2]) / imageHeight
    return box[0], center_x, center_y, width, height
 
 
# 切图主类
class Crop():
    def __init__(self, ):
        self.dataSet = None
        self.labelPath = ''
        self.label_names = None
 
    def inputImage(self, imagePath):
        self.dataSet = MyDataset(imagePath)
 
    def inputLabel(self, labelPath, label_names=None, coordinates='x1y1x2y2'):
        self.labelPath = labelPath
        self.label_names = label_names
        if coordinates not in ['yolo', 'x1y1x2y2']:
            raise Exception('coordinates参数需指定yolo或x1y1x2y2之一')
        self.inputlabel_coordinates = coordinates
 
    def getLabel(self, index, ):
        if self.labelPath == '':
            print('未定义标签地址，若需使用标签请使用inputLabel方法传入标签地址')
            return []
        else:
            txtPath = os.path.join(self.labelPath, self.dataSet.get_name(index).split('.')[0] + '.txt')
            try:
                object_list = (pd.read_table(txtPath, header=None, sep=' ')).values
                if self.inputlabel_coordinates == 'yolo':
                    Height, Width = self.dataSet[index].shape[1:3]
                    # 转换为x1y1x2y2
                    object_list[:, 1], object_list[:, 2], object_list[:, 3], object_list[:, 4] = \
                        ((object_list[:, 1] - object_list[:, 3] / 2) * Width).astype(int), \
                        ((object_list[:, 2] - object_list[:, 4] / 2) * Height).astype(int), \
                        ((object_list[:, 1] + object_list[:, 3] / 2) * Width).astype(int), \
                        ((object_list[:, 2] + object_list[:, 4] / 2) * Height).astype(int)
 
            except:
                # 若读取报错（表示文件为空），则指定列表为空。
                object_list = []
            return object_list
 
    # 展示图片
    def showImage(self, index, figsize=(10, 8)):
        plt.figure(figsize=figsize)
        plt.title(self.dataSet.get_name(index))
        if self.labelPath != '':
            labels = self.getLabel(index)
            fig = plt.imshow(self.dataSet[index].permute((1, 2, 0)))
            for cls, *box in labels:
                fig.axes.add_patch(bbox_to_rect(box, color='red'))
                # 注释虫子名称
                plt.text(box[0], box[1], self.label_names[cls] if self.label_names else None, color='blue')
        else:
            plt.imshow(self.dataSet[index].permute((1, 2, 0)))
        plt.show()
 
 
class slidingWindowCrop(Crop):
    def __init__(self, windowSize=None, rowcol=None):
        if not ((windowSize or rowcol) and not (windowSize and rowcol)):
            raise Exception('windowSize and rowcol must Only one can be defined')
 
        self.windowSize = windowSize  # （Width, Height）
        self.rowcol = rowcol  # (row, col)
        self.labelPath = ''
        self.dataSet = None
        self.label_names = None
 
    def showSliceImage(self, index, overlap, new_box_iou_limit=0.35, figsize=(10, 8)):
        object_list = self.getLabel(index)  # 获取该原图上的目标框数据集合
        Width, Height = self.dataSet[index].shape[:0:-1]  # 获取原图片的宽高
        if self.rowcol:
            # 通过行列数计算得到滑动窗口长宽
            windowSize = self.ranksGetWindowSize(self.rowcol, (Width, Height), overlap)
        else:
            windowSize = self.windowSize
        print(f'{self.dataSet.get_name(index)}子图宽高为：', windowSize[0], windowSize[1])
        slice_imag(self.dataSet[index], sliceWidth=windowSize[0], sliceHeight=windowSize[1], object_list=object_list,
                   overlap=overlap, new_box_iou_limit=new_box_iou_limit, imshow=True, figsize=figsize,
                   label_names=self.label_names)
 
    # 当为'rowcol'定义时，通过行列数以及overlap来确定窗口的大小
    @staticmethod
    def ranksGetWindowSize(nrow_ncol, Width_Height, overlap):
        nrow, ncol = nrow_ncol
        Width, Height = Width_Height
 
        # 由于通过行列数以及overla计算得到的子图长宽存在小数，取整时会导致行列数变化，通过更新高宽来保持行列数
        # 且若刚好为整数又会因切图时的程序会导致少一行/一列，因此需减少高/宽来保持行列数不变
        def contral_rowcol(RowOrCol, WidthOrHeight, sliceWidthOrHeight, overlap):
            # 通过以下公式若大于col+1或row+1，则说明会导致多一行/一列,则通过增加长/宽来避免
            while len(computeSlicePosition(WidthOrHeight, sliceWidthOrHeight, overlap)) > RowOrCol:
                sliceWidthOrHeight += 1
            # 通过以下公式若小于col或row，则说明会导致少一行/一列,则通过减小长/宽来避免
            while len(computeSlicePosition(WidthOrHeight, sliceWidthOrHeight, overlap)) < RowOrCol:
                sliceWidthOrHeight -= 1
            #             print(len(computeSlicePosition(WidthOrHeight,sliceWidthOrHeight,overlap)))
            return sliceWidthOrHeight
 
        # 通过公式计算满足指定overlap以及指定行列数时的长宽值
        sliceWidth = np.int(Width / (ncol * (1 - overlap) + overlap))
        sliceHeight = np.int(Height / (nrow * (1 - overlap) + overlap))
 
        # 更新长宽值以保证切分时的行列数不变
        # 当计算得到的值等于原图宽/高时说明指定行/列为1，因此不需要重新更新（若更新则程序会使导致多一行/列）
        if Width != sliceWidth:
            sliceWidth = contral_rowcol(ncol, Width, sliceWidth, overlap)
        if Height != sliceHeight:
            sliceHeight = contral_rowcol(nrow, Height, sliceHeight, overlap)
        return int(sliceWidth), int(sliceHeight)
 
    def __repeatMethod__(self, index, overlap=0.5, new_box_iou_limit=0.35):
        if overlap >= 1 or overlap < 0:
            raise Exception("overlap must >=0 and <1")
 
        image_name = self.dataSet.get_name(index)  # 指定图片的名字
        object_list = self.getLabel(index)  # 指定图片的目标框
        Width, Height = self.dataSet[index].shape[:0:-1]  # 获取原图片的宽高
 
        if self.rowcol:
            # 通过行列数计算得到滑动窗口长宽
            windowSize = self.ranksGetWindowSize(self.rowcol, (Width, Height), overlap)
        else:
            windowSize = self.windowSize
        print(f'{self.dataSet.get_name(index)}子图宽高为：', windowSize[0], windowSize[1])
 
        sliceWidth, sliceHeight = windowSize
 
        # 获取切分子图以及子图目标框
        slice_images, exiset_obj_lists = slice_imag(self.dataSet[index], sliceWidth=sliceWidth, sliceHeight=sliceHeight,
                                                    object_list=object_list, overlap=overlap,
                                                    new_box_iou_limit=new_box_iou_limit, )
 
        ncol = len(computeSlicePosition(Width, sliceWidth, overlap))  # 一行有几个子图
        nrow = len(computeSlicePosition(Height, sliceHeight, overlap))  # 有几行
        print('行列数为：', nrow, ncol)
 
        return image_name, slice_images, windowSize, exiset_obj_lists, nrow, ncol
 
    def saveSubImage(self, index, imgs_save_path, overlap=0.5, resize=None, new_box_iou_limit=0.35):
        """通过索引保存子图"""
        # 如果不存在文件夾则创建
        if not os.path.exists(imgs_save_path):
            os.makedirs(imgs_save_path)
 
        image_name, slice_images, windowSize, exiset_obj_lists, nrow, ncol = self.__repeatMethod__(index, overlap,
                                                                                                   new_box_iou_limit)
 
        # 图片resize尺寸，若为none则尺寸不变
        resize = (windowSize[0], windowSize[1]) if not resize else resize
 
        n_save_imgs = 0
        for num, sub_image, in enumerate(slice_images):
            n_save_imgs += 1
            # 子图位置编号
            sub_row = (num) // ncol
            sub_col = (num) % ncol
            path = os.path.join(imgs_save_path, image_name.split('.')[0] + f'_{sub_row}' + f'_{sub_col}.png')
            print('save:', path)
            # 保存图片到指定路径并将图片resize为（640,640）
            transforms.ToPILImage()(sub_image).resize(resize).save(path)
        return n_save_imgs
 
    def saveSubImageAndTxt(self, index, imgs_save_path, labels_save_path, overlap=0.5,
                           resize=None, new_box_iou_limit=0.35, coordinates='yolo'):
        if coordinates in ['yolo', 'x1y1x2y2']:
            pass
        else:
            raise Exception('coordinates参数需指定yolo或x1y1x2y2之一')
 
        # 如果不存在文件夾则创建
        if not os.path.exists(imgs_save_path):
            os.makedirs(imgs_save_path)
 
        if not os.path.exists(labels_save_path):
            os.makedirs(labels_save_path)
 
        image_name, slice_images, windowSize, exiset_obj_lists, nrow, ncol = self.__repeatMethod__(index, overlap,
                                                                                                   new_box_iou_limit)
 
        # 图片resize尺寸，若为none则尺寸不变
        resize = (windowSize[0], windowSize[1]) if not resize else resize
 
        n_save_imgs = 0
        for num, (sub_image, exiset_obj_list) in enumerate(zip(slice_images, exiset_obj_lists)):
 
            # 子图位置编号
            sub_row = (num) // ncol
            sub_col = (num) % ncol
            if exiset_obj_list:
                n_save_imgs += 1
                path_image = os.path.join(imgs_save_path, image_name.split('.')[0] + f'_{sub_row}' + f'_{sub_col}.png')
 
                # 保存图片到指定路径并将图片resize为
                transforms.ToPILImage()(sub_image).resize(resize).save(path_image)
                # 保存子图相对应labels的txt文件到指定路径
                path_label = os.path.join(labels_save_path,
                                          image_name.split('.')[0] + f'_{sub_row}' + f'_{sub_col}.txt')
                print('save:', path_image, '  ', path_label)
                # 如果已存在该子图名称文件，可能会重复写入，因此移除来重新写入
                if os.path.exists(path_label):
                    os.remove(path_label)
                for box in exiset_obj_list:
                    save_txt(path_label, toYolo(box, windowSize[0], windowSize[1]) if coordinates == 'yolo' else box)
        return n_save_imgs
 
    def saveSubTxt(self, index, labels_save_path, overlap=0.5, new_box_iou_limit=0.35, coordinates='yolo'):
        if coordinates in ['yolo', 'x1y1x2y2']:
            pass
        else:
            raise Exception('coordinates参数需指定yolo或x1y1x2y2之一')
 
        if not os.path.exists(labels_save_path):
            os.makedirs(labels_save_path)
 
        image_name, slice_images, windowSize, exiset_obj_lists, nrow, ncol = self.__repeatMethod__(index, overlap,
                                                                                                   new_box_iou_limit)
 
        n_save_txts = 0
        for num, (sub_image, exiset_obj_list) in enumerate(zip(slice_images, exiset_obj_lists)):
            # 子图位置编号
            sub_row = (num) // ncol
            sub_col = (num) % ncol
            if exiset_obj_list:
                n_save_txts += 1
 
                # 保存子图相对应labels的txt文件到指定路径
                path_label = os.path.join(labels_save_path,
                                          image_name.split('.')[0] + f'_{sub_row}' + f'_{sub_col}.txt')
                print('save:', path_label)
                # 如果已存在该子图名称文件，可能会重复写入，因此移除来重新写入
                if os.path.exists(path_label):
                    os.remove(path_label)
                for box in exiset_obj_list:
                    save_txt(path_label, toYolo(box, windowSize[0], windowSize[1]) if coordinates == 'yolo' else box)
        return n_save_txts
 
 
def randomCropPosition(labels, Width, Height, subWidth, subHeight):
    label, xmin, ymin, xmax, ymax = labels
 
    width_range_min = -(subWidth - (xmax - xmin)) if xmin - (subWidth - (xmax - xmin)) > 0 else -xmin
    width_range_max = 0 if xmin + subWidth < Width else width_range_min + (Width - xmax)
 
    height_range_min = -(subHeight - (ymax - ymin)) if ymin - (subHeight - (ymax - ymin)) > 0 else -ymin
    height_range_max = 0 if ymin + subHeight < Height else height_range_min + (Height - ymax)
 
    try:
        subwidth_deviation = random.randint(int(width_range_min), int(width_range_max))
        subheight_deviation = random.randint(int(height_range_min), int(height_range_max))
    except:
        # 出现异常：子图宽高小于目标框宽高
        return None
    sub_xmin, sub_ymin = xmin + subwidth_deviation, ymin + subheight_deviation
 
    sub_xmax, sub_ymax = sub_xmin + subWidth, sub_ymin + subHeight
    return sub_xmin, sub_ymin, sub_xmax, sub_ymax
 
 
def randomCrop(image, image_name, labels=[], subWidth=1000, subHeight=1000, new_box_iou_limit=0.3,
               imshow=True, label_names=None, figsize=(10, 8)):
    if len(labels) == 0:
        print('无目标框，不进行切分')
        return [], []
    if imshow:
        nrow = int(np.sqrt(len(labels)))
        ncol = int(np.ceil(len(labels) / nrow))
 
        _, axes = plt.subplots(nrow, ncol, figsize=figsize)
        axes = axes.flatten()
    images, exiset_obj_lists = [], []
    for num, label in enumerate(labels):
        try:
            sub_xmin, sub_ymin, sub_xmax, sub_ymax = map(int, randomCropPosition(label, image.shape[2], image.shape[1],
                                                                                 subWidth, subHeight))
        except TypeError:
            print(f'{image_name}:the subimage\'s Width/Height must > box\'s Width/Height')
            return [], []
        exiset_obj_list = exist_objs([sub_xmin, sub_ymin, sub_xmax, sub_ymax], labels,
                                     new_box_iou_limit)
        sub_image = image[:, sub_ymin:sub_ymax, sub_xmin:sub_xmax]
        #         print(exiset_obj_list)
        images.append(sub_image)
        exiset_obj_lists.append(exiset_obj_list)
 
        if imshow:
            # 展示分割后的子图
            axes[num].imshow(sub_image.permute((1, 2, 0)).numpy())
            #             axes[num].axes.get_xaxis().set_visible(False)
            #             axes[num].axes.get_yaxis().set_visible(False)
            # 在新的子图上展示目标框
            for category, *position in exiset_obj_list:
                axes[num].add_patch(bbox_to_rect(position, color='red'))
                if label_names:
                    axes[num].text(position[0], position[1], label_names[category], color='blue')
    plt.show()
    return images, exiset_obj_lists
 
 
class randomCenterCrop(Crop):
    def __init__(self, windowSize):
 
        self.windowSize = windowSize  # （Width, Height）
        self.dataSet = None
        self.labelPath = ''
 
    def showCopImage(self, index, new_box_iou_limit=0.35,
                     figsize=(10, 8), ):
        image = self.dataSet[index]
        image_name = self.dataSet.get_name(index)
        labels = self.getLabel(index)
 
        images, exiset_obj_lists = randomCrop(image, image_name, labels, self.windowSize[0], self.windowSize[1],
                                              new_box_iou_limit=new_box_iou_limit,
                                              label_names=self.label_names, figsize=figsize, imshow=True)
 
    def saveSubImageAndTxt(self, index, imgs_save_path, labels_save_path, coordinates='yolo',
                           resize=None, new_box_iou_limit=0.35, ):
 
        if coordinates in ['yolo', 'x1y1x2y2']:
            pass
        else:
            raise Exception('coordinates参数需指定yolo或x1y1x2y2之一')
 
        # 如果不存在文件夾则创建
        if not os.path.exists(imgs_save_path):
            os.makedirs(imgs_save_path)
 
        if not os.path.exists(labels_save_path):
            os.makedirs(labels_save_path)
 
        image = self.dataSet[index]
        image_name = self.dataSet.get_name(index)
        labels = self.getLabel(index)
 
        images, exiset_obj_lists = randomCrop(image, image_name, labels, self.windowSize[0], self.windowSize[1],
                                              new_box_iou_limit=new_box_iou_limit, imshow=False)
        image_name = self.dataSet.get_name(index)
        # 图片resize尺寸，若为none则尺寸不变
        resize = (self.windowSize[0], self.windowSize[1]) if not resize else resize
 
        n_save_imgs = 0
        for num, (sub_image, exiset_obj_list) in enumerate(zip(images, exiset_obj_lists)):
            if exiset_obj_list:
                n_save_imgs += 1
                # 在多次切图时，保存的子图名称序号依次增加
                num_ = n_save_imgs
                while True:
                    path_image = os.path.join(imgs_save_path, image_name.split('.')[0] + f'_{num_ - 1}.png')
                    if not os.path.exists(path_image):
                        break
                    else:
                        num_ += 1
 
                # 保存图片到指定路径并将图片resize为
                transforms.ToPILImage()(sub_image).resize(resize).save(path_image)
 
                # 保存子图相对应labels的txt文件到指定路径
                path_label = os.path.join(labels_save_path, image_name.split('.')[0] + f'_{num_ - 1}.txt')
                print('save:', path_image, '  ', path_label)
                # 如果已存在该子图名称文件，可能会重复写入，因此移除来重新写入
                if os.path.exists(path_label):
                    os.remove(path_label)
                for box in exiset_obj_list:
                    save_txt(path_label,
                             toYolo(box, self.windowSize[0], self.windowSize[1]) if coordinates == 'yolo' else box)
        return n_save_imgs
