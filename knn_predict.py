import torch
import numpy as np 
import pickle
import torch.nn.functional as F
from tqdm import tqdm  

memory_path = './logs/ImageNet_LT/stage1/ImageNet_LT_90_coslr/ImageNet_LT_90_coslr_memorybank.pkl'
testfeature_path = './logs/ImageNet_LT/stage1/ImageNet_LT_90_coslr/ImageNet_LT_90_coslr_testfeatures.pkl'
testlabel_path = './logs/ImageNet_LT/stage1/ImageNet_LT_90_coslr/ImageNet_LT_90_coslr_testlabels.pkl'
with open(memory_path, 'rb') as f:
    data = pickle.load(f)
with open(testfeature_path, 'rb') as f:
    batch_x = pickle.load(f)
with open(testlabel_path, 'rb')  as f:
    gt_labels = torch.from_numpy(pickle.load(f))


classes = 1000
features = F.normalize(torch.from_numpy(data['all_features']),dim=1).t()
labels = torch.from_numpy(data['all_labels'])

batch_x = F.normalize(torch.from_numpy(batch_x),dim=1)

def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
    preds = []
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    for row in tqdm(sim_labels):
        if len(torch.unique(row)) == len(row):
            preds.append(row[0].item())
        else:
            preds.append(torch.mode(row)[0].item())
    preds = torch.tensor(preds)
    print((preds==gt_labels).sum()/50000)
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
  
    # weighted score ---> [B, C]
    # pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return (pred_labels[:,0]==gt_labels).sum()/50000

print(knn_predict(batch_x, features, labels, 1000, 100,  0.1))

