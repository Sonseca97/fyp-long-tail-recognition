import torch
from torch import nn
from torch.nn import functional as F

class AttentionLayer(nn.Module):

    def __init__(self, feat_dim=512):
        super(AttentionLayer, self).__init__()
        latent_dim = int(feat_dim / 4)
        self.query_linear = nn.Linear(feat_dim, latent_dim)
        self.key_linear = nn.Linear(feat_dim, latent_dim)
    
    def forward(self, features, centroids, logits):
        projected_features = self.query_linear(features)       
        projected_keys = self.key_linear(centroids)
        weight = F.softmax(projected_features.mm(projected_keys.t()), dim=1)

        return weight * logits
