import cv2
from src.MTCNN import MTCNN
import os
import  numpy as np
from src.FaceEncoding import get_all_face_encodings
from src.FaceEncoding import get_face_encoding

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建MTCNN对象
mtcnn = MTCNN(threshold=[0.5, 0.6, 0.7],  # 置信度阈值
              nms_threshold=[0.7, 0.7, 0.7],   # nms阈值
              weight_paths=['./weight_path/pnet.h5',   # 权重文件路径
                            './weight_path/rnet.h5',
                            './weight_path/onet.h5'],
              max_face=True,    # 是否检测最大人脸
              save_face=True,    # 是否保存检测到的人脸
              save_dirt="./output/3",  # 保存人脸的路径
              print_time=False,  # 是否打印时间信
              print_message=False,    # 是否打印辅助信息
              detect_hat=False,
              resize_type='without_loss',    # resize图片类型
              padding_type='mask'   # 填充图片类型
              )

# 读取图片
for root, dirs, files in os.walk('input/1'):
    for file in files:
        if file.endswith('.jpg'):
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            rects, landmarks, ret = mtcnn.detect_face(image)
            if ret == False:
                print("未检测到人脸")

# image = cv2.imread('test.jpg')
# rects,landmarks,ret=mtcnn.detect_face(image)
# if ret==False:
#     print("未检测到人脸")