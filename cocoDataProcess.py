# coco数据集写入、计算mAP等
# 20190906 by：xz

# 将自己的数据写为标准的coco数据集
import cv2
import json
import sys
 
# process bar 状态条
def process_bar(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))
 
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
 
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
 
root_path = "data_with_box_example/"
images, categories, annotations = [], [], []
 
category_dict = {"people": 1}
 
for cat_n in category_dict:
    categories.append({"supercategory": "", "id": category_dict[cat_n], "name": cat_n}) # supercategory超类？
 
with open("label_example.txt", 'r') as f:
    img_id = 0
    anno_id_count = 0
    count = 1
    total = 100
    for line in f.readlines():
        process_bar(count, total)
        count += 1
        line = line.split(' ')
        img_name = line[0].replace('/', '_')
        bbox_num = int(line[1])
        img_cv2 = cv2.imread(root_path + img_name)
        [height, width, _] = img_cv2.shape
 
        # images info
        images.append({"file_name": img_name, "height": height, "width": width, "id": img_id})
 
        """
        annotation info:
        id : anno_id_count
        category_id : category_id
        bbox : bbox
        segmentation : [segment]
        area : area
        iscrowd : 0
        image_id : image_id
        """
        category_id = category_dict["people"]
        for i in range(0, bbox_num):
            x1 = float(line[i * 5 + 3])
            y1 = float(line[i * 5 + 4])
            x2 = float(line[i * 5 + 3]) + float(line[i * 5 + 5])
            y2 = float(line[i * 5 + 4]) + float(line[i * 5 + 6])
            width = float(line[i * 5 + 5])
            height = float(line[i * 5 + 6])
 
            bbox = [x1, y1, width, height]
            segment = [x1, y1, x2, y1, x2, y2, x1, y2]
            area = width * height
 
            anno_info = {'id': anno_id_count, 'category_id': category_id, 'bbox': bbox, 'segmentation': [segment],
                         'area': area, 'iscrowd': 0, 'image_id': img_id}
            annotations.append(anno_info)
            anno_id_count += 1
 
        img_id = img_id + 1
 
all_json = {"images": images, "annotations": annotations, "categories": categories}
 
with open("example.json", "w") as outfile:
    json.dump(all_json, outfile)

# 调用coco-api计算gt.json与det.json数据的mAP值
import matplotlib.pyplot as plt 
from pycocotools.coco import COCO 
from pycocotools.cocoeval import COCOeval 
import numpy as np 
import skimage.io as io 
import pylab,json 
pylab.rcParams['figure.figsize'] = (10.0, 8.0) 
def get_img_id(file_name): 
    ls = [] 
    myset = [] 
    annos = json.load(open(file_name, 'r')) 
    for anno in annos: 
      ls.append(anno['image_id']) 
    myset = {}.fromkeys(ls).keys() 
    return myset 

if __name__ == '__main__': 
    annType = ['segm', 'bbox', 'keypoints'] # 选择iou的类别，一般检测用bbox。
    annType = annType[1]
    cocoGt_file = '/home/ss/data/coco2014/annotations/instances_val2014.json'
    cocoGt = COCO(cocoGt_file)
    cocoDt_file = '/home/ss/darknet/results/coco_results.json'
    cocoDt = cocoGt.loadRes(cocoDt_file)

    imgIds = get_img_id(cocoDt_file)
    print len(imgIds)
    imgIds = sorted(imgIds) #按顺序排列coco标注集image_id
    imgIds = imgIds[0:5000] #标注集中的image数据
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds#参数设置，默认是所有[all] N img ids to use for evaluation
    cocoEval.evaluate()#评价
    cocoEval.accumulate()#积分
    cocoEval.summarize()#总

