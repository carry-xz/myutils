# -*-coding:utf-8-*-

import sys
import os

import xml.etree.ElementTree as Et
from xml.etree.ElementTree import Element, ElementTree
from tqdm import tqdm
import numpy as np
import cv2

import json

# meta Data list format
"""
meta_data_list = [img_info, img_info1,....]
img_info:
{
    "filename" :  filename,
    "database":  database,   
    "path": path;
    "data":{                  
            "size" :
                        {
                            "width" : <string>
                            "height" : <string>
                            "depth" : <string>
                        }
        
            "objects" :
                        {
                            "<index>" :
                                        {
                                            "name" : <string>
                                            "bndbox" :
                                                        {
                                                            "xmin" : <float>
                                                            "ymin" : <float>
                                                            "xmax" : <float>
                                                            "ymax" : <float>
                                                        }
                                        }
                            ...
        
        
                        }
        }
"""

# XML Data format
"""
{
    "filename" : <XML Object>
    ...
}
"""


class DataTypeConvert:

    def __init__(self):
        self._info = {}
        self.img_num = 0
        self.bbox_num = 0
        self.class_num = 0
        self.meta_data_list = []

    def xml_indent(self, elem, level=0):
        i = "\n" + level * "\t"
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "\t"
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.xml_indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def _generateXmlElement(self, annotation_dic):
        filename = annotation_dic['filename']
        element = annotation_dic['data']
        xml_annotation = Element("annotation")

        xml_folder = Element("folder")
        xml_folder.text = ' '
        xml_filename = Element("filename")
        xml_filename.text = filename
        xml_path = Element("path")
        xml_path.text = filename
        xml_source = Element("source")
        xml_database = Element("database")
        xml_database.text = 'Unknown'
        xml_source.append(xml_database)
        xml_annotation.append(xml_folder)
        xml_annotation.append(xml_filename)
        xml_annotation.append(xml_path)
        xml_annotation.append(xml_source)

        xml_size = Element("size")
        xml_width = Element("width")
        xml_width.text = element["size"]["width"]
        xml_size.append(xml_width)

        xml_height = Element("height")
        xml_height.text = element["size"]["height"]
        xml_size.append(xml_height)

        xml_depth = Element("depth")
        xml_depth.text = element["size"]["depth"]
        xml_size.append(xml_depth)

        xml_annotation.append(xml_size)

        xml_segmented = Element("segmented")
        xml_segmented.text = "0"

        xml_annotation.append(xml_segmented)

        if int(element["objects"]["num_obj"]) < 1:
            return False, "number of Object less than 1"

        for i in range(0, int(element["objects"]["num_obj"])):
            xml_object = Element("object")
            obj_name = Element("name")
            obj_name.text = element["objects"][str(i)]["name"]
            xml_object.append(obj_name)

            obj_pose = Element("pose")
            obj_pose.text = "Unspecified"
            xml_object.append(obj_pose)

            obj_truncated = Element("truncated")
            obj_truncated.text = "0"
            xml_object.append(obj_truncated)

            obj_difficult = Element("difficult")
            obj_difficult.text = "0"
            xml_object.append(obj_difficult)

            xml_bndbox = Element("bndbox")

            obj_xmin = Element("xmin")
            obj_xmin.text = str(int(element["objects"][str(i)]["bndbox"]["xmin"]))
            xml_bndbox.append(obj_xmin)

            obj_ymin = Element("ymin")
            obj_ymin.text = str(int(element["objects"][str(i)]["bndbox"]["ymin"]))
            xml_bndbox.append(obj_ymin)

            obj_xmax = Element("xmax")
            obj_xmax.text = str(int(element["objects"][str(i)]["bndbox"]["xmax"]))
            xml_bndbox.append(obj_xmax)

            obj_ymax = Element("ymax")
            obj_ymax.text = str(int(element["objects"][str(i)]["bndbox"]["ymax"]))
            xml_bndbox.append(obj_ymax)
            xml_object.append(xml_bndbox)

            xml_annotation.append(xml_object)

        self.xml_indent(xml_annotation)

        return xml_annotation

    def generateXmlElements(self, meta_data_list):
        xml_list = []
        for data in meta_data_list:
            filename = data['filename']
            xml_annotation = self._generateXmlElement(data)
            xml_list.append((filename, xml_annotation))
        return xml_list

    def saveXmls(self, save_path, meta_data_list):

        path = os.path.abspath(save_path)
        os.makedirs(path, exist_ok=True)
        xmls_list = self.generateXmlElements(meta_data_list)

        for item in tqdm(xmls_list):
            filename = item[0]
            end_fix = filename.split('.')[-1]
            filename_xml = filename.replace(end_fix, 'xml')
            xml_element = item[1]
            filepath = os.path.join(path, filename_xml)
            Et.ElementTree(xml_element).write(filepath)
        print('finish save xml files')
        return True, None

    def _saveXml(self, element_xml, xml_save_pth):

        try:
            filepath = os.path.abspath(xml_save_pth)
            edata = ElementTree(element_xml)
            edata.write(filepath)

            return True, None

        except Exception as e:

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

            msg = "ERROR : {}, moreInfo : {}\t{}\t{}".format(e, exc_type, fname, exc_tb.tb_lineno)

            return False, msg

    def _parseXml(self, xml_pth):
        xml = open(xml_pth, "r")
        tree = Et.parse(xml)
        root = tree.getroot()

        filename = root.find('filename').text
        database = root.find('source').find('database').text
        path = root.find('path').text
        info = {}
        info['filename'] = filename
        info['database'] = database
        info['path'] = path

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
            bndbox = {
                "xmin": int(xml_bndbox.find("xmin").text),
                "ymin": int(xml_bndbox.find("ymin").text),
                "xmax": int(xml_bndbox.find("xmax").text),
                "ymax": int(xml_bndbox.find("ymax").text)
            }
            tmp["bndbox"] = bndbox
            obj[str(obj_index)] = tmp

            obj_index += 1

        annotation_dic = {
            "size": size,
            "objects": obj,
            "filename": filename,
        }

        return annotation_dic, info

    def parseXmls(self, img_dir, xml_dir=None):
        if xml_dir is None:
            xml_dir = img_dir

        (dir_path, dir_names, filenames) = next(os.walk(os.path.abspath(img_dir)))

        meta_data_list = []
        progress_cnt = 0
        for count, filename in enumerate(filenames):
            data_dic = {}
            info = {}
            if not filename.lower().endswith(('jpg', 'png', 'jpeg')):
                continue
            end_fix = filename.split('.')[-1]
            filename_xml = filename.replace(end_fix, 'xml')
            annotation, info = self._parseXml(os.path.join(xml_dir, filename_xml))

            data_dic['data'] = annotation
            data_dic['filename'] = filename
            data_dic['database'] = info['database']
            data_dic['path'] = info['path']

            meta_data_list.append(data_dic)

            progress_cnt += 1

        return meta_data_list

    def parseCoco(self, json_pth):
        meta_data_list = []
        with open(json_pth, 'r') as f:
            json_data = json.load(f)
            print(json_data.keys())
        images_ = json_data['images']
        annotations_ = json_data['annotations']
        type_ = json_data['type']
        categories_ = json_data['categories']

        img_num = len(images_)
        annotation_num = len(annotations_)
        ann_imgid_list = []
        for i in range(annotation_num):
            ann_imgid_list.append(annotations_[i]['image_id'])
        ann_imgid_list = np.array(ann_imgid_list)

        for i in range(img_num):
            img_info_dic = {}
            temp_info = images_[i]
            img_id = temp_info['id']
            filename = temp_info['file_name']
            img_width, img_height = temp_info["width"], temp_info["height"]
            if 'depth' in temp_info:
                depth = temp_info['depth']
            else:
                depth = 3
            size = {"width": img_width, "height": img_height, "depth": depth}
            ann_ind_selet = np.where(ann_imgid_list == img_id)[0]
            obj = {}
            obj_index = 0
            obj = {
                "num_obj": len(ann_ind_selet)
            }
            for j in ann_ind_selet:
                anno = annotations_[j]
                bndbox = {
                    "xmin": anno["bbox"][0],
                    "ymin": anno["bbox"][1],
                    "xmax": anno["bbox"][2] + anno["bbox"][0],
                    "ymax": anno["bbox"][3] + anno["bbox"][1]
                }
                cls_id = anno["category_id"]
                for category in categories_:
                    if category["id"] == cls_id:
                        cls = category["name"]
                tmp = {
                    "name": cls,
                    "bndbox": bndbox
                }
                obj[str(obj_index)] = tmp
                obj_index += 1

            annotation_dic = {
                "size": size,
                "objects": obj,
            }

            img_info_dic['data'] = annotation_dic
            img_info_dic['filename'] = filename
            img_info_dic['database'] = 'Unknown'
            img_info_dic['path'] = ' '

            meta_data_list.append(img_info_dic)

        return meta_data_list

    def saveJson(self, json_save_pth, meta_data_list, img_dir=None, limit=None):
        json_dir = os.path.split(json_save_pth)[0]
        os.makedirs(json_dir, exist_ok=True)

        json_dict = {"images": [], "type": "instances", "annotations": [],
                     "categories": []}
        categories = {}
        bnd_id = 0
        count = 0
        res_w_h = []
        for data_i in tqdm(meta_data_list):
            filename = data_i['filename']
            end_fix = filename.split('.')[-1]

            image_id = count
            count += 1
            data = data_i['data']
            width = int(data['size']['width'])
            height = int(data['size']['height'])
            if width == 0 or height == 0:
                if img_dir is None:
                    print('Warning {} width or height is 0'.format(filename))
                    print('Warning try to set img_dir parameter')
                    return 0
                img = cv2.imread(os.path.join(img_dir, filename))
                height, width = img.shape[:2]
            if (limit is not None) and (width < limit or height < limit):
                continue
            image = {'file_name': filename, 'height': height, 'width': width,
                     'id': image_id}
            json_dict['images'].append(image)

            if int(data["objects"]["num_obj"]) < 1:
                return False, "number of Object less than 1"

            for i in range(0, int(data["objects"]["num_obj"])):
                objs = data['objects']
                obj = objs[str(i)]
                category = obj['name']
                if category not in categories:
                    new_id = len(categories)
                    categories[category] = new_id
                category_id = categories[category]

                bndbox = obj['bndbox']
                xmin = int(bndbox['xmin'])
                ymin = int(bndbox['ymin'])
                xmax = int(bndbox['xmax'])
                ymax = int(bndbox['ymax'])
                assert (xmax > xmin)
                assert (ymax > ymin)
                o_width = abs(xmax - xmin)
                o_height = abs(ymax - ymin)
                ann = {'area': o_width * o_height, 'iscrowd': 0, 'image_id':
                    image_id, 'bbox': [xmin, ymin, o_width, o_height],
                       'category_id': category_id, 'id': bnd_id, 'ignore': 0,
                       'segmentation': []}
                json_dict['annotations'].append(ann)
                bnd_id = bnd_id + 1

        print('all img number:', count)
        for cate, cid in categories.items():
            cat = {'supercategory': 'none', 'id': cid, 'name': cate}
            json_dict['categories'].append(cat)
        with open(json_save_pth, 'w') as f:
            json_str = json.dumps(json_dict)
            f.write(json_str)
        print('finish save json:', json_save_pth)

    def parseYoloTxts(self, img_dir, txt_dir=None):
        if txt_dir is None:
            txt_dir = img_dir
        meta_data_list = []
        filenames = [item for item in os.listdir(img_dir) if item.lower().endswith(('jpg', 'png', 'jpeg'))]
        for filename in filenames:
            img_info_dic = {}
            filename_ = filename.split('.')[0]
            filename_txt = ''.join([filename_, ".txt"])
            txt_pth = os.path.join(txt_dir, filename_txt)
            if not os.path.exists(txt_pth):
                continue
            txt = open(txt_pth, "r")
            img = cv2.imread(os.path.join(img_dir, filename))
            img_height = str(img.shape[0])
            img_width = str(img.shape[1])
            img_depth = 3

            size = {
                "width": img_width,
                "height": img_height,
                "depth": img_depth
            }

            obj = {}
            obj_cnt = 0

            for line in txt:
                elements = line.strip().split(" ")
                name_id = elements[0]

                xminAddxmax = float(elements[1]) * (2.0 * float(img_width))
                yminAddymax = float(elements[2]) * (2.0 * float(img_height))

                w = float(elements[3]) * float(img_width)
                h = float(elements[4]) * float(img_height)

                xmin = (xminAddxmax - w) / 2
                ymin = (yminAddymax - h) / 2
                xmax = xmin + w
                ymax = ymin + h

                bndbox = {
                    "xmin": float(xmin),
                    "ymin": float(ymin),
                    "xmax": float(xmax),
                    "ymax": float(ymax)
                }

                obj_info = {
                    "name": name_id,
                    "bndbox": bndbox
                }

                obj[str(obj_cnt)] = obj_info
                obj_cnt += 1

            obj["num_obj"] = len(obj)

            annotation_dic = {
                "size": size,
                "objects": obj,
            }

            img_info_dic['data'] = annotation_dic
            img_info_dic['filename'] = filename
            img_info_dic['database'] = 'Unknown'
            img_info_dic['path'] = ' '

            meta_data_list.append(img_info_dic)
        return meta_data_list

    def xxyyCvt2YOLO(self, size, box):
        dw = 1. / size[0]
        dh = 1. / size[1]

        x = (box[0] + box[1]) / 2.0
        y = (box[2] + box[3]) / 2.0

        w = box[1] - box[0]
        h = box[3] - box[2]

        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (round(x, 3), round(y, 3), round(w, 3), round(h, 3))

    def saveYoloTxt(self, txt_save_pth,  meta_data_list, img_dir=None, limit=None):
        os.makedirs(txt_save_pth, exist_ok=True)
        count = 0
        categories = {}
        filename_list = []

        for data_img in tqdm(meta_data_list):
            filename = data_img['filename']
            filename_list.append(filename)
            filename_txt = filename.split('.')[0] + '.txt'
            count += 1
            data = data_img['data']
            width = int(data['size']['width'])
            height = int(data['size']['height'])
            if width == 0 or height == 0:
                if img_dir is None:
                    print('Warning {} width or height is 0'.format(filename))
                    print('Warning try to set img_dir parameter')
                    return 0
                img = cv2.imread(os.path.join(img_dir, filename))
                height, width = img.shape[:2]
            if (limit is not None) and (width < limit or height < limit):
                continue

            txt_pth = os.path.join(txt_save_pth, filename_txt)
            xywhcs = []

            if int(data["objects"]["num_obj"]) < 1:
                return False, "number of Object less than 1"

            for i in range(0, int(data["objects"]["num_obj"])):
                objs = data['objects']
                obj = objs[str(i)]
                category = obj['name']
                if category not in categories:
                    new_id = len(categories)
                    categories[category] = new_id
                category_id = categories[category]

                bndbox = obj['bndbox']
                xmin = int(bndbox['xmin'])
                ymin = int(bndbox['ymin'])
                xmax = int(bndbox['xmax'])
                ymax = int(bndbox['ymax'])
                assert (xmax > xmin)
                assert (ymax > ymin)
                xcycwh = self.xxyyCvt2YOLO([width, height], [xmin, xmax, ymin, ymax])
                xywhcs.append((xcycwh, category_id))

            with open(txt_pth, 'w') as f:
                for item in xywhcs:
                    f.write(' '.join([str(item[1]), ' '.join(map(str, item[0]))]) + '\n')

        with open(txt_save_pth + '_train.txt', 'w') as f:
            for item in filename_list:
                f.write(item + '\n')

        with open(txt_save_pth + '_name.txt', 'w') as f:
            name_list = sorted(categories, key=lambda x: x[1])
            for item in name_list:
                f.write(item + '\n')
        print('finish save yolo train,name,labels txt')


if __name__ == '__main__':
    con = DataTypeConvert()
    meta_data_list = con.parseXmls('/workspace/dataset/detection/temp')
    meta_data_list2 = con.parseCoco('/workspace/dataset/detection/temp/t.json')
    con.saveXmls('/workspace/dataset/detection/xml', meta_data_list)
    con.saveYoloTxt('/workspace/dataset/detection/yolo', meta_data_list)
    con.saveJson('/workspace/dataset/detection/temp/t.json', meta_data_list)
    meta_data_list3 = con.parseYoloTxts('/workspace/dataset/detection/temp', '/workspace/dataset/detection/yolo')

    print('finish')
