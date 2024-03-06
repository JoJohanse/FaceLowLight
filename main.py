import cv2
from src.MTCNN import MTCNN
import os
import  numpy as np
from src.FaceEncoding import get_all_face_encodings
from src.FaceEncoding import get_face_encoding
from src.FaceIdntify import FaceIdentify

# 创建MTCNN对象
mtcnn = MTCNN(threshold=[0.5, 0.6, 0.7],  # 置信度阈值
              nms_threshold=[0.7, 0.7, 0.7],   # nms阈值
              weight_paths=['./weight_path/pnet.h5',   # 权重文件路径
                            './weight_path/rnet.h5',
                            './weight_path/onet.h5'],
              max_face=True,    # 是否检测最大人脸
              save_face=False,    # 是否保存检测到的人脸
              save_dirt="./output",  # 保存人脸的路径
              print_time=False,  # 是否打印时间信息
              print_message=False,    # 是否打印辅助信息
              detect_hat=False,  # 是否检测安全帽佩戴
              resize_type='without_loss',    # resize图片类型
              padding_type='mask'   # 填充图片类型
              )

# 加载已知的人脸编码
known_encodings,labels= get_all_face_encodings('output')
print(known_encodings)
print(labels)

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取一帧
    ret, frame = cap.read()

    # 检测人脸
    rects, landmarks, ret = mtcnn.detect_face(frame)

    if ret == False:
        print("未检测到人脸")
    else:
        for i in range(len(rects)):
            rect = rects[i]
            landmark = landmarks[i]

            # 提取人脸
            face_image = frame[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])]
            # 获取人脸编码
            face_encoding= get_face_encoding(face_image)
            # 与已知的人脸编码进行比较
            result = FaceIdentify(face_encoding, known_encodings, labels)
            # 显示人脸识别结果
            cv2.putText(frame, str(result), (int(rect[0]), int(rect[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # 显示人脸框
            cv2.rectangle(frame, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (0, 255, 0), 2)

    # 显示结果
    cv2.imshow('Video', frame)

    # 按'q'退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()