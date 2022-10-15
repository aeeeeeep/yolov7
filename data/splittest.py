from Crop import slidingWindowCrop as SW
from Crop import randomCenterCrop as RC

bug = {0: '0', 1:'1', 2:'2', 3:'3',4:'4',5:'5',6:'6', 7:'7'}
sw = SW(rowcol = (4,6))

sw.inputImage('./test/images')
# sw.saveSubImage(-2, './train/splitimages', overlap=0.5, resize=None, new_box_iou_limit=0.35)
nums=0
for i in range(len(sw.dataSet)):
    sw.saveSubImage(i, './train/splittest', overlap=0.5, resize=None, new_box_iou_limit=0.35)
print(nums)
