# -*-coding:utf-8-*-

import xml.dom.minidom
import cv2


def creatXML(filename, filepath, folder, width, height, trackResult):
    # 在内存中创建一个空的文档
    doc = xml.dom.minidom.Document()
    # 创建一个根节点Managers对象
    root = doc.createElement('annotation')
    # 设置根节点的属性
    # root.setAttribute('filename', filename)
    # root.setAttribute('path', path)
    # 将根节点添加到文档对象中
    doc.appendChild(root)

    trackResult = trackResult

    dir = doc.createElement('folder')
    dir.appendChild(doc.createTextNode(folder))

    file = doc.createElement('filename')
    file.appendChild(doc.createTextNode(filename))

    path = doc.createElement('path')
    path.appendChild(doc.createTextNode(filepath))

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode(str(0)))

    source = doc.createElement('source')
    nodedatabase = doc.createElement('database')
    nodedatabase.appendChild(doc.createTextNode(str('Unknown')))
    source.appendChild(nodedatabase)

    imageSize = doc.createElement('size')
    nodeWidth = doc.createElement('width')
    nodeWidth.appendChild(doc.createTextNode(str(width)))

    nodeHeight = doc.createElement('height')
    nodeHeight.appendChild(doc.createTextNode(str(height)))

    nodeDepth = doc.createElement('depth')
    nodeDepth.appendChild(doc.createTextNode(str(3)))

    imageSize.appendChild(nodeWidth)
    imageSize.appendChild(nodeHeight)
    imageSize.appendChild(nodeDepth)

    root.appendChild(dir)
    root.appendChild(file)
    root.appendChild(path)
    root.appendChild(source)
    root.appendChild(imageSize)
    root.appendChild(segmented)

    # for i in managerList:
    # ------------------------------
    for key, values in trackResult.items():

        # for index in range(len(values[-1])):
        if values[-1][-1] == filename.split(".")[0]:
            # values[-1]帧数列表
            x = values[1][-1][0]
            y = values[1][-1][1]
            w = x + values[1][-1][2]
            h = y + values[1][-1][3]
            fps = values[-1][-1]
            name = values[0][-1]
            # if name.split("_")[0] == filename.split(".")[0]:
            # if values[-1][-1] == filename.split(".")[0]:
            # ------------------------------
            # testPath = "/home/asus/mixGroup/facelabel_v2/readvideo/tmp/" + fps + ".jpg"
            # image = cv2.imread(testPath)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # cv2.imwrite("../readvideo/tmp/{}.png".format(filename.split(".")[0]), image)
            # ------------------------------

            nodeManager = doc.createElement('object')
            nodeName = doc.createElement('name')
            # 给叶子节点name设置一个文本节点，用于显示文本内容
            nodeName.appendChild(doc.createTextNode(str(name)))

            nodePose = doc.createElement('pose')
            nodePose.appendChild(doc.createTextNode(str('Unspecified')))

            nodetruncated = doc.createElement('truncated')
            nodetruncated.appendChild(doc.createTextNode(str(0)))

            nodedifficult = doc.createElement('difficult')
            nodedifficult.appendChild(doc.createTextNode(str(0)))
            ###########################
            bndbox = doc.createElement('bndbox')
            nodeX = doc.createElement('xmin')
            nodeX.appendChild(doc.createTextNode(str(x)))

            nodeY = doc.createElement("ymin")
            nodeY.appendChild(doc.createTextNode(str(y)))

            nodeW = doc.createElement("xmax")
            nodeW.appendChild(doc.createTextNode(str(w)))

            nodeH = doc.createElement("ymax")
            nodeH.appendChild(doc.createTextNode(str(h)))

            bndbox.appendChild(nodeX)
            bndbox.appendChild(nodeY)
            bndbox.appendChild(nodeW)
            bndbox.appendChild(nodeH)

            nodeManager.appendChild(nodeName)
            nodeManager.appendChild(nodePose)
            nodeManager.appendChild(nodetruncated)
            nodeManager.appendChild(nodedifficult)
            nodeManager.appendChild(bndbox)
            root.appendChild(nodeManager)
        else:
            continue
    # 开始写xml文档
    with open("../tmpImg/{}.xml".format(filename.split(".")[0]), 'a') as f:
        doc.writexml(f, indent='\t', addindent='\t', newl='\n', encoding="utf-8")


if __name__ == '__main__':
    filename = 'test.jpg'
    path = './images/'
    folder = '030'
    width = '1080'
    height = '1024'
    trackResult = ''
    creatXML(filename, path, folder, width, height, trackResult)
