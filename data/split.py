from Crop import slidingWindowCrop as SW
from Crop import randomCenterCrop as RC

bug = {0: '0', 1:'1', 2:'2', 3:'3',4:'4',5:'5',6:'6', 7:'7'}
sw = SW(rowcol = (2,2))

sw.inputImage('./train/images')
sw.inputLabel('./train/labels', label_names = bug, coordinates='yolo')
# sw.saveSubImage(-2, './train/splitimages', overlap=0.5, resize=None, new_box_iou_limit=0.35)
# sw.saveSubTxt(-2, './train/splitlabels', overlap=0.5)
nums=0
for i in range(len(sw.dataSet)):
    nums += 1
    sw.saveSubImageAndTxt(i, './train/splitimages', './train/splitlabels', overlap=0.3, resize=None, new_box_iou_limit=0.2, coordinates='yolo')
print(nums)
