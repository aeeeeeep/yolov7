from Crop import slidingWindowCrop as SW
from Crop import randomCenterCrop as RC

bug = {0: '0', 1:'1', 2:'2', 3:'3',4:'4',5:'5',6:'6', 7:'7'}
sw = SW(rowcol = (1,1))

sw.inputImage('./train/imagesbk')
sw.inputLabel('./train/labelsbk', label_names = bug, coordinates='yolo')
nums=0
for i in range(len(sw.dataSet)):
    nums += 1
    sw.saveSubImageAndTxt(i, './train/images', './train/labels', overlap=0.5, resize=(1440,1024), new_box_iou_limit=0.35, coordinates='yolo')
print(nums)
# rc = RC(windowSize = (1440, 1024))
# rc.inputImage('./train/images')
# rc.inputLabel('./train/labels',label_names = bug, coordinates='yolo')
# nums=0
# for i in range(len(rc.dataSet)):
#     nums += 1
#     rc.saveSubImageAndTxt(i, './train/splitimages', './train/splitlabels', resize=None)
# print(nums)
