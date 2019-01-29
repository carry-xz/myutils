#-*-coding:utf-8-*-
import os,shutil
from datetime import datetime
import re
import cv2 as cv
import xml.etree.ElementTree as ET

# 遍历图像文件，将文件中图像路径写入txt文件
# 文件内必须有iamges和labels文件夹
# iamges中存放图像，labels存放yolo需要的标签文件
# python 3.5测试

'''
主要函数
1.遍历文件夹找到指定的文件，复制到指定位置。
2.将文件夹内所有xml文件转化为txt文件。
3.遍历文件夹，将图片路径写入txt文档。
'''


def getTimeStr():
    return datetime.now().strftime('%Y%m%d-%H%M%S')

def checkJpegFile(image_path):
    try:
        f = cv.imread(image_path)
        return True
    except Exception as e:
        with open('err.txt', 'a') as f:
            f.write(image_path)
            f.write('\n')
        return False

def walkThroughAndCopy(walk_dir,tar_dir,file_type=('jpg','png')):
    # 遍历目标文件，将指定类型的文件存入另一目标文件夹
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)
    for root, dirs, files in os.walk(walk_dir, topdown=True):
        for name in files:
            filename = os.path.join(root,name)
            if name.endswith(file_type):
                shutil.copy(filename,os.path.join(tar_dir,name))
    print('copy finish!')

def walkThroughAndWriteTxt2Xml(walk_dir,tar_dir,file_type=('xml')):
    # 遍历目标文件，将txt转化为xml并存入另一目标文件夹
    classes=['car','person']
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)
    for root, dirs, files in os.walk(walk_dir, topdown=True):
        for name in files:
            filename = os.path.join(root,name)
            if name.endswith(file_type):
                xml2txt(filename,classes,tar_dir)
    print('write xml finish!')

def jpegFileWrite2Txt(walk_dir,out_dir,file_type=('jpg','png')):
    if not os.path.exists(out_dir):
        os.path.makedirs(out_dir)
    out_file = 'train_'+getTimeStr()+'.txt'
    out_file = os.path.join(out_dir,out_file)
    with open(out_file, 'a') as f:
        for root, dirs, files in os.walk(walk_dir, topdown=True):
            for name in files:
                filename = os.path.join(root,name)
                if name.endswith(file_type):
                    if checkJpegFile(filename):
                        f.write(filename)
                        f.write('\n')
    print('write txt finish!')

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def xml2txt(in_file,classes=[],out_dir='./'):
    if not os.path.exists(out_dir):
        os.path.makedirs(out_dir)
    out_file = os.path.split(in_file)[-1]
    out_file = out_file.split('.')[0]+'.txt'
    out_file = os.path.join(out_dir,out_file)
    out_file = open(out_file, 'w')
    in_file = open(in_file)

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def praperData4Yolo(walk_dir,tar_dir,classes=[]):
    # 遍历目标文件，将图片、xml存入指定文件夹，将xml文件转化为txt文件，适用于yolo的训练文件格式。
    image_type = ('jpg','png')
    label_type = ('xml')
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)
    if not os.path.exists(os.path.join(tar_dir,'images')):
        os.makedirs(os.path.join(tar_dir,'images'))
    if not os.path.exists(os.path.join(tar_dir,'anaotation')):
        os.makedirs(os.path.join(tar_dir,'anaotation'))
    if not os.path.exists(os.path.join(tar_dir,'labels')):
        os.makedirs(os.path.join(tar_dir,'labels'))
    for root, dirs, files in os.walk(walk_dir, topdown=True):
        for name in files:
            filename = os.path.join(root,name)
            if name.endswith(image_type):
                shutil.copy(filename,os.path.join(tar_dir,'images',name))
            if name.endswith(label_type):
                shutil.copy(filename,os.path.join(tar_dir,'anaotation',name))
                xmlfilename = os.path.join(tar_dir,'anaotation',name)
                txtfildir = os.path.join(tar_dir,'labels')
                xml2txt(xmlfilename,classes,out_dir=txtfildir)

    jpegFileWrite2Txt(tar_dir,tar_dir)
    print('praper yolo data finish!')




if __name__=="__main__":
    imgdir = 'D:\\2018\\testdata'
    tardir = 'D:\\2018\\'
    classes = ['car', 'person','house']
    praperData4Yolo(imgdir, tardir, classes)

