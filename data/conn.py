from Connect import Connect, connectImage
bug = {0: '0', 1:'1', 2:'2', 3:'3',4:'4',5:'5',6:'6', 7:'7'}
connect = Connect(rowcol = (4,6), overlap=0.5, iou_thres=0.35, label_names=bug)
connect.saveConnectTxt('/root/2022IEEEUV-preliminary/data/test/images','/root/2022IEEEUV-preliminary/runs/test/exp12/labels','/root/2022IEEEUV-preliminary/data/train/conntest',show=False,coordinates='yolo')
