import os
import torch
import cv2
from facenet_pytorch import InceptionResnetV1

resnet=InceptionResnetV1(pretrained='casia-webface').eval()

def get_face_encoding(image):
    """
    获取人脸编码
    :param image: 人脸图片
    :return: 人脸编码 人脸标签
    """
    # 如果图片格式不为RGB，将图片转换为RGB格式
    if image.shape[2] == 4:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 1:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:  # 假设图像是BGR格式
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # 获取人脸编码
    face_encoding = resnet(torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0).float())
    face_encoding = face_encoding.detach().numpy()

    return face_encoding

def get_all_face_encodings(root_dir):
    """
    获取所有人脸编码
    :param: root_dir: 图片目录
    :return: 所有人脸的编码列表
    """
    all_encodings = []
    all_labels = []

    # 遍历根目录下的所有子目录
    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)

        # 如果这是一个目录
        if os.path.isdir(dir_path):
            # 遍历目录中的所有文件
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)

                # 如果这是一个文件
                if os.path.isfile(file_path):
                    # 读取图像文件
                    image = cv2.imread(file_path)
                    # 获取标签,标签为图片所在文件夹名
                    label = file_path.split(os.path.sep)[-2]
                    # 获取人脸编码
                    face_encoding = get_face_encoding(image)

                    # 将人脸编码添加到列表中
                    all_encodings.append(face_encoding)
                    # 将标签添加到列表中
                    all_labels.append(label)
                    
    return all_encodings, all_labels