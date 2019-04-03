# -*- coding=utf-8 -*-
# 20190403
# by:xz

# 将视频文件转化为jpg图片,每个视频一个文件夹。
# 运行环境python
# 依赖库：opencv-python

import cv2
import os,time,sys

def video2image(video_folder,save_folder=None,frame_skip=4):
    """
    input:str video_folder: 视频文件夹路径
          str save_folder:图片保存文件夹路径
          frame_skip:间隔多少帧保存图片
    """
    if save_folder is None:
        save_folder=video_folder
    video_list = os.listdir(video_folder)
    print(video_list)
    for video_name in video_list:
        video_file = os.path.join(video_folder,video_name)
        save_folder_name = os.path.join(save_folder,video_name.split('.')[0])
        try:
            cap = cv2.VideoCapture(video_file)
        except Exception as e:
            print('file is not video!')
            break
        if not os.path.exists(save_folder_name):
            os.makedirs(save_folder_name)
        isopened = cap.isOpened()
        ret, frame = cap.read()
        count = 0
        while isopened:
            img_name = video_name.split('.')[0]+'_'+str(count)+'.jpg'
            img_file_name = os.path.join(save_folder_name,img_name)
            count +=1
            for i in range(frame_skip):
                cap.read()
            ret, frame = cap.read()
            if frame is None:
                cap.release()
                break
            cv2.imwrite(img_file_name, frame)
            print('count:',count)
    
    print('convert video to image finish!')

if __name__ == "__main__":
    videoFolder = 'D:\\test'
    video2image(videoFolder,frame_skip=50)