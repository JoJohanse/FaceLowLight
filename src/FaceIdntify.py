import numpy as np
from sklearn.neighbors import NearestNeighbors

def FaceIdentify(face_encoding, known_encodings, labels, threshold=0.13):
    '''
    face_encoding: 输入人脸的编码
    known_encodings: 已知的人脸编码
    labels: 已知的人脸标签
    threshold: 距离阈值
    '''
    # 初始化最小距离和最小距离的索引
    min_distance = float('inf')
    min_distance_index = -1

    # 计算输入人脸编码与已知编码之间的距离
    for i, known_encoding in enumerate(known_encodings):
        distance = np.linalg.norm(known_encoding - face_encoding)
        if distance < min_distance:
            min_distance = distance
            min_distance_index = i

    # 如果最小距离小于阈值，则返回对应的标签，否则返回"Unknown"
    if min_distance < threshold:
        return labels[min_distance_index]
    else:
        return "Unknown"

# 使用欧式距离计算两个人脸编码之间的距离
# def is_known_face(face_encoding, known_encodings, threshold=1.0e-7):
#     '''
#     known_encodings: 已知的人脸编码列表,是一组编码
#     '''
#     # 计算新的人脸编码与已知编码之间的距离
#     distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
#     print(np.min(distances))
#     # 如果最小距离小于阈值，则返回True，否则返回False
#     return np.min(distances) < threshold