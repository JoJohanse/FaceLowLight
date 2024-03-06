# import tensorflow as tf
import cv2
from src.MTCNN import MTCNN
import os
import  src.mytest as mytest
import  numpy as np
import math

def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建MTCNN对象
mtcnn = MTCNN(threshold=[0.5, 0.6, 0.7],  # 置信度阈值
              nms_threshold=[0.7, 0.7, 0.7],   # nms阈值
              weight_paths=['./weight_path/pnet.h5',   # 权重文件路径
                            './weight_path/rnet.h5',
                            './weight_path/onet.h5'],

              max_face=False,    # 是否检测最大人脸
              save_face=False,    # 是否保存检测到的人脸
              save_dirt="./output",  # 保存人脸的路径
              print_time=False,  # 是否打印时间信息
              print_message=False,    # 是否打印辅助信息
              detect_hat=False,  # 是否检测安全帽佩戴
              resize_type='without_loss',    # resize图片类型
              padding_type='mask'   # 填充图片类型
              )

# 读取图片
num = 0
a = 0
folder_path = "input"
for file in os.listdir(folder_path):
    s = '%06d' % (num+1)
    image = cv2.imread(folder_path +s + '.jpg')
    num +=1

    #直方图均衡化
    # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
    (b, g, r) = cv2.split(image)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    # # outputpath = "./output"
    # # name = '%06d' % num
    # # cv2.imwrite(outputpath + "/" + name + ".jpg", result)

    #gama矫正
    img_gray = cv2.imread(folder_path + s + '.jpg', 0)
    mean = np.mean(img_gray)
    gamma_val = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma
    image_gamma_correct = gamma_trans(image, gamma_val)  # gamma变换
    # outputpath = "./output"
    # name = '%06d' % num
    # cv2.imwrite(outputpath + "/" + name + ".jpg", image)

    # 传入网络
    rects, landmark, ret = mtcnn.detect_face(image_gamma_correct)

    print(np.array(landmark).shape)
    if np.array(landmark).shape == (5, 1, 2):

        landmark = np.array(landmark).reshape((5, 2))

        #仿射变换
        img_ = mytest.to_get_max_face(rects, image)
        wrap = mytest.norm_crop(img_, np.array(landmark), image_size=112, mode='arcface')
        #保存图片
        outputpath = "./output2"
        a += 1
        name = '%06d' % a
        cv2.imwrite(outputpath + "/" + name + ".jpg", img_)


# else:
#     for rect in rects:
#
#         img_ = cv2.rectangle(image.copy(),
#                              (int(rect[0]), int(rect[1])),
#                              (int(rect[2]), int(rect[3])),
#                              (0, 255, 0), 4)
#         cv2.imshow("img_", img_)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#