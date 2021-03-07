import torch 
import torch.nn as nn 
import numpy as np 
from utils import * 
import os 
import pickle
import numpy as np 
import scipy
import torch.nn.functional as F
import pandas as pd
from utils import source_import

os.environ['CUDA_VISIBLE_DEVICES']='1'

def analyze_d_matrix():
    f_cos = os.path.join(os.getcwd(), 'logs/ImageNet_LT/stage1/ImageNet_LT_90_second_fc_M_from_1_normalize_cos_similarity_1.0_0.1_0.1_ldam_0.1.pkl')
    f_mse = os.path.join(os.getcwd(), 'logs/ImageNet_LT/stage1/ImageNet_LT_90_second_fc_M_from_1_normalize_cos_similarity_1.0_0.1_0.1_ldam_0.1_final.pkl')
    f_ori = os.path.join(os.getcwd(), 'logs/ImageNet_LT/stage1/ImageNet_LT_90_second_fc_M_from_1_normalize_cos_1.0_0.1_final.pkl')
    with open(f_cos, 'rb') as f_cos, open(f_mse, 'rb') as f_mse, open(f_ori, 'rb') as f_ori:
        data_cos = pickle.load(f_cos).cpu()
        data_mse = pickle.load(f_mse)['l2ncs']
        data_ori = pickle.load(f_ori)['l2ncs']
    # cal distance matrix
    d_cos = scipy.spatial.distance.cdist(data_cos, data_cos, metric='cosine')
    d_mse = scipy.spatial.distance.cdist(data_mse, data_mse, metric='cosine')
    d_ori = scipy.spatial.distance.cdist(data_ori, data_ori, metric='cosine')
    # lowest distance for each class
    nn_cos = np.sort(d_cos, axis=0)[1,:]
    nn_mse = np.sort(d_mse, axis=0)[1,:]
    nn_ori = np.sort(d_ori, axis=0)[1,:]
    class_dict = source_import("imagenet_dict.py").class_dict

    count_cos = 0
    count_mse = 0
    same_count=0
    for i in range(1000):
        idx_cos = int((np.where(d_cos[i]==nn_cos[i]))[0])
        idx_mse = int((np.where(d_mse[i]==nn_mse[i]))[0])
        idx_ori = int((np.where(d_ori[i]==nn_ori[i]))[0])
        if idx_ori == idx_cos:
            count_cos += 1
        if idx_ori == idx_mse:
            count_mse += 1
        if idx_cos ==idx_mse:
            same_count += 1
        # print("class: ",class_dict[i])
        # print("cos nearest: ", class_dict[idx_cos])
        # print("mse nearest: ", class_dict[idx_mse])
        # print("ori nearest: ", class_dict[idx_ori])
        # print()
        # if i == 20:
        #     break
    print(count_cos)
    print(count_mse)
    print(same_count)

def check_loss():
    feat = torch.rand(128, 512).view(128, -1)

    centroids = torch.rand(128, 512)
    loss = (feat - centroids).pow(2).sum()
    print(loss)

    # print(feat.shape)
    # batch_size_tensor = feat.new_empty(1).fill_(128)
    # print(batch_size_tensor)
    # print(batch_size_tensor.shape)


def cos_similarity(A, B):
    cos = nn.CosineSimilarity()
    a = A[1].unsqueeze(0).repeat(1000, 1)
    result = cos(a, B)
    print(result)
    print(torch.argmax(result))


    feat_dim = A.size(1)
    AB = torch.mm(A, B.t())
    AB = AB / feat_dim
    print(torch.argmax(AB, dim=1))
    print(AB)
    exit()
    return AB

def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    print(sim_mt)
    print(sim_mt.shape)
    exit()
    return sim_mt
if __name__ == "__main__":
    A = torch.rand(128, 512)
    B = torch.rand(1000, 512)
    # sim_matrix(A, B)
    analyze_d_matrix()
    # a = torch.tensor([[1,2.,3.]])
    # c = torch.tensor([[2.,4., 6.],[3., 4., 6.]])
    # print(l2_similarity(a, c))
    # print(torch.norm(a-c, 2, 1, keepdim=True).flatten())






