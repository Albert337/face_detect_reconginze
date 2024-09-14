import torch
from yoloface_detect_align_module import yoloface
from get_face_feature import arcface_dnn
import numpy as np
from scipy import spatial


# 人脸检测→人脸对齐→人脸识别
def get_reconginzed_face(srcimg,faces_feature,names_list):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # arcface 用于提取人脸特征向量
    face_embdnet = arcface_dnn()
    # 人脸检测器
    detect_face = yoloface(device=device)

    if srcimg is None:
        exit('please give correct image')
    boxs, faces_img = detect_face.get_face(srcimg)
    if len(faces_img) == 0:
        exit('no detec face')
    threshold =0.65
    min_id = 0
    for i, face in enumerate(faces_img):
        tmp_embd = face_embdnet.get_feature(face)
        feature_out = np.reshape(tmp_embd[0],(1,-1))
        # print(feature_out.shape,faces_feature.shape)
        dist = spatial.distance.cdist(faces_feature, feature_out, metric='cosine').flatten()
        # [print("with {} the dist:{}".format(names_list[ii], dist[ii])) for ii in range(len(names_list))]   
        min_id = np.argmin(dist)
        pred_score = dist[min_id]
        pred_name = 'unknow'
        if dist[min_id] <= threshold:
            pred_name = names_list[min_id]

    return pred_name, min_id,feature_out


