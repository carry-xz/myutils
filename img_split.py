
# -*-coding:utf-8-*-
import sys
import os
import json
import xml.etree.ElementTree as ET
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cv2, shutil
from tqdm import tqdm
import time

class VOC:
    """
    Handler Class for VOC PASCAL Format
    """

    def __indent(self, elem, level=0):
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.__indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def writeXmlAndImg(self, bboxs, obj_names, img, img_save_path, xml_save_pth, pix_limit=5):
        # bbox xywh
        hight, width, depth = img.shape
        img_name = os.path.split(img_save_path)[-1]
        xml_annotation = ET.Element("annotation")

        xml_folder = ET.Element("folder")
        xml_folder.text = ' '
        xml_filename = ET.Element("filename")
        xml_filename.text = img_name
        xml_path = ET.Element("path")
        xml_path.text = img_name
        xml_source = ET.Element("source")
        xml_database = ET.Element("database")
        xml_database.text = 'Unknown'
        xml_source.append(xml_database)
        xml_annotation.append(xml_folder)
        xml_annotation.append(xml_filename)
        xml_annotation.append(xml_path)
        xml_annotation.append(xml_source)

        xml_size = ET.Element("size")
        xml_width = ET.Element("width")
        xml_width.text = str(width)
        xml_size.append(xml_width)

        xml_height = ET.Element("height")
        xml_height.text = str(hight)
        xml_size.append(xml_height)

        xml_depth = ET.Element("depth")
        xml_depth.text = str(depth)
        xml_size.append(xml_depth)

        xml_annotation.append(xml_size)

        xml_segmented = ET.Element("segmented")
        xml_segmented.text = "0"

        xml_annotation.append(xml_segmented)

        if len(bboxs) < 1:
            print("number of Object less than 1")
            return False

        for i in range(0, len(bboxs)):
            xml_object = ET.Element("object")
            obj_name = ET.Element("name")

            x, y, w, h = bboxs[i]
            if w <= pix_limit or h <= pix_limit:
                continue
            if (x == 0 or y == 0 or x + w >= width - 1 or y + h >= hight - 1) and (w / h > 5 or h / w > 5):
                continue
            if isinstance(obj_names, list):
                obj_name.text = obj_names[i]
            else:
                obj_name.text = obj_names

            xml_object.append(obj_name)

            obj_pose = ET.Element("pose")
            obj_pose.text = "Unspecified"
            xml_object.append(obj_pose)

            obj_truncated = ET.Element("truncated")
            obj_truncated.text = "0"
            xml_object.append(obj_truncated)

            obj_difficult = ET.Element("difficult")
            obj_difficult.text = "0"
            xml_object.append(obj_difficult)

            xml_bndbox = ET.Element("bndbox")

            obj_xmin = ET.Element("xmin")
            obj_xmin.text = str(int(x))
            xml_bndbox.append(obj_xmin)

            obj_ymin = ET.Element("ymin")
            obj_ymin.text = str(int(y))
            xml_bndbox.append(obj_ymin)

            obj_xmax = ET.Element("xmax")
            obj_xmax.text = str(int(x + w))
            xml_bndbox.append(obj_xmax)

            obj_ymax = ET.Element("ymax")
            obj_ymax.text = str(int(y + h))
            xml_bndbox.append(obj_ymax)
            xml_object.append(xml_bndbox)
            xml_annotation.append(xml_object)

        self.__indent(xml_annotation)
        cv2.imwrite(img_save_path, img)
        time.sleep(0.002)
        self.save(xml_annotation, xml_save_pth)
        return True

    @staticmethod
    def save(xml_object, xml_path):
        xml_dir = os.path.split(xml_path)[0]
        os.makedirs(xml_dir, exist_ok=True)
        ET.ElementTree(xml_object).write(xml_path)

    @staticmethod
    def parse_(xml_path):
        xywhs, names = [], []

        dir_path, filename = os.path.split(xml_path)
        xml = open(os.path.join(dir_path, filename), "r")

        tree = ET.parse(xml)
        root = tree.getroot()

        xml_size = root.find("size")
        size = {
            "width": xml_size.find("width").text,
            "height": xml_size.find("height").text,
            "depth": xml_size.find("depth").text

        }

        objects = root.findall("object")
        if len(objects) == 0:
            return False, "number object zero"

        obj = {
            "num_obj": len(objects)
        }

        obj_index = 0
        for _object in objects:
            tmp = {
                "name": _object.find("name").text
            }

            xml_bndbox = _object.find("bndbox")
            xmin = float(xml_bndbox.find("xmin").text)
            ymin = float(xml_bndbox.find("ymin").text)
            xmax = float(xml_bndbox.find("xmax").text)
            ymax = float(xml_bndbox.find("ymax").text)
            bndbox = {
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            }
            xywhs.append([xmin, ymin, xmax - xmin, ymax - ymin])
            names.append(tmp['name'])
            tmp["bndbox"] = bndbox
            obj[str(obj_index)] = tmp
            obj_index += 1

        annotation = {
            "size": size,
            "objects": obj
        }

        return annotation, xywhs, names



class ImageSplit():
    def __init__(self, max_len, over_lap=0.0):
        self.over_lap = over_lap
        self.max_len = max_len
        self.voc = VOC()

    def _splitBoxsByLimits(self, xywhs, limits, box_names):
        res_boxs = []
        res_names = []
        xywhs = np.array(xywhs)
        names_ = np.array(box_names)
        xywhs_ = xywhs.copy()
        xywhs_[:, 2:4] = xywhs_[:, :2] + xywhs_[:, 2:4]
        for limit in limits:
            x1, x2, y1, y2 = limit
            mask_sel = (xywhs_[:, 0] >= x1) & (xywhs_[:, 0] <= x2) & (xywhs_[:, 1] >= y1) & (xywhs_[:, 1] <= y2)
            mask_sel = mask_sel | (
                    (xywhs_[:, 2] >= x1) & (xywhs_[:, 2] <= x2) & (xywhs_[:, 3] >= y1) & (xywhs_[:, 3] <= y2))
            xywhs_sel = xywhs_[mask_sel].copy()
            xywhs_sel[:, 0] = np.clip(xywhs_sel[:, 0], x1, x2) - x1
            xywhs_sel[:, 1] = np.clip(xywhs_sel[:, 1], y1, y2) - y1
            xywhs_sel[:, 2] = np.clip(xywhs_sel[:, 2], x1, x2) - x1 - xywhs_sel[:, 0]
            xywhs_sel[:, 3] = np.clip(xywhs_sel[:, 3], y1, y2) - y1 - xywhs_sel[:, 1]
            names_sel = names_[mask_sel]
            res_boxs.append(xywhs_sel)
            res_names.append(names_sel.tolist())
        return res_boxs, res_names

    def _splistImgMatrix(self, img, num_x, num_y):
        res = []
        limits = []
        height, width = img.shape[:2]
        img_w, img_h = width // num_x + 1, height // num_y + 1
        over_x, over_y = int(img_w * self.over_lap), int(img_h * self.over_lap)
        for i in range(0, num_y):
            for j in range(0, num_x):
                x1 = max(j * img_w - over_x, 0)
                x2 = min(width, (j + 1) * img_w + over_x)
                y1 = max(i * img_h - over_y, 0)
                y2 = min(height, (i + 1) * img_h + over_y)
                img_ = img[y1:y2, x1:x2, :]
                res.append(img_)
                limits.append([x1, x2, y1, y2])
        return res, limits

    def _splitImgAndBoxs(self, img, xywhs, box_names, part=None):
        height, width = img.shape[:2]
        if width < self.max_len * 2 and height < self.max_len * 2:
            return None
        if part is None:
            num_x = max(width // self.max_len, 1)
            num_y = max(height // self.max_len, 1)
        else:
            num_x = num_y = part
        if width >= self.max_len or height >= self.max_len:
            res_imgs, limt_ = self._splistImgMatrix(img, num_x, num_y)
            res_xywhs, res_names = self._splitBoxsByLimits(xywhs, limt_, box_names)
            return res_imgs, res_xywhs, res_names
        else:
            return None

    def splitImgAndXml(self, img_pth, save_dir=None, with_source=True):
        if save_dir is None:
            save_dir, img_name = os.path.split(img_pth)
        else:
            img_name = os.path.split(img_pth)[-1]
        os.makedirs(save_dir, exist_ok=True)
        end_fix = img_pth.split('.')[-1]
        xml_pth = img_pth.replace(end_fix, 'xml')
        xml_name = img_name.replace(end_fix, 'xml')
        _, xywhs, names = self.voc.parse_(xml_pth)
        img = cv2.imread(img_pth)
        data = self._splitImgAndBoxs(img, xywhs, names, None)
        if data is None:
            if with_source:
                shutil.copy(img_pth, os.path.join(save_dir, img_name))
                shutil.copy(xml_pth, os.path.join(save_dir, img_name.replace(end_fix, 'xml')))
            return 0

        res_imgs, res_xywhs, res_names = data
        for i, boxs in enumerate(res_xywhs):
            img_name_i = xml_name[:-4] + '_{}.jpg'.format(i)
            img_save_pth = os.path.join(save_dir, img_name_i)
            xml_save_pth = os.path.join(save_dir, xml_name[:-4] + '_{}.xml'.format(i))
            self.voc.writeXmlAndImg(boxs, res_names[i], res_imgs[i], img_save_pth, xml_save_pth)
        return 1

    def splitImgs(self, source_dir, tar_dir, with_source=True):
        img_list = os.listdir(source_dir)
        img_list = [os.path.join(source_dir, item) for item in img_list if item.endswith(('jpg', 'png', 'JPEG', 'PNG'))]
        count_ = 0
        for img_pth in tqdm(img_list):
            count = self.splitImgAndXml(img_pth, save_dir=tar_dir, with_source=with_source)
            count_ += count
        print('all split img number:{}/{}'.format(count_, len(img_list)))
