import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

class Attentnion_auto(nn.Module):
    def __init__(self, num_logits=2, num_classes=100):
        super(Attentnion_auto, self).__init__()
        self.k_matrix = nn.Linear(num_logits, num_logits)
        self.q_matrix = nn.Linear(num_logits, num_logits)
        self.v_matrix = nn.Linear(num_logits, num_logits)
        

    def forward(self, x):
        # x (128, 2, 1000)
        


        feature_pow = torch.pow(x, 2)
        feature_map = torch.mean(feature_pow, dim=1).view(-1, 1, self.kernel_size, self.kernel_size)
        feature_map = self.conv(feature_map).view(-1, self.kernel_size ** 2)
        feature_weight = F.softmax(feature_map, dim=-1).view(-1, 1, self.kernel_size, self.kernel_size).expand_as(x)
        out_map = feature_weight * x
        output = torch.sum(torch.sum(out_map, dim=3), dim=2)
