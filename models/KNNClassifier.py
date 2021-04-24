"""Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import torch
import torch.nn as nn
import numpy as np
import pickle
import os 

class KNNClassifier(nn.Module):
    def __init__(self, feat_dim=512, num_classes=1000, feat_type='cl2n', dist_type='l2'):
        super(KNNClassifier, self).__init__()
        assert feat_type in ['uncs', 'l2ncs', 'cl2ncs'], "feat_type is wrong!!!"
        assert dist_type in ['l2', 'cos'], "dist_type is wrong!!!"
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.centroids = torch.randn(num_classes, feat_dim)
        self.feat_mean = torch.randn(feat_dim)
        self.feat_type = feat_type
        self.dist_type = dist_type
        self.initialized = True

        if self.feat_type in ['l2ncs', 'cl2ns']:
            self.norm_input = True
        else:
            self.norm_input = False
    
    def update(self, cfeats):
        mean = cfeats['mean']
        centroids = cfeats['{}cs'.format(self.feat_type)]
       

        mean = torch.from_numpy(mean)
        centroids = torch.from_numpy(centroids)
        self.feat_mean.copy_(mean)
        self.centroids.copy_(centroids)
        if torch.cuda.is_available():
            self.feat_mean = self.feat_mean.cuda()
            self.centroids = self.centroids.cuda()
        self.initialized = True
    
    def load_memory(self, log_dir, path):
        m_path = '{}.pkl'.format(path)
        fname = os.path.join(log_dir, m_path)
        if os.path.exists(fname):
            print('===> Loading features from %s' % fname)
            with open(fname, 'rb') as f:
                data = pickle.load(f).cpu()
            centroids = data
            # norm_c = torch.norm(centroids, 2, 1, keepdim=True)
            # centroids = centroids/norm_c
            self.centroids.copy_(centroids)
            if torch.cuda.is_available():
                self.centroids = self.centroids.cuda()
            self.initialized = True
      

    def forward(self, inputs, centroids, mean=None,*args):
        
        if self.feat_type == 'cl2ncs':
            assert mean is not None, "Mean cannot be None!!"
            inputs = inputs - mean 
        
        if self.norm_input:
            norm_x = torch.norm(inputs, 2, 1, keepdim=True)
            inputs = inputs / norm_x
            
        # Logit calculation
        if self.dist_type == 'l2':
            logit = self.l2_similarity(inputs, centroids)
    
        elif self.dist_type == 'cos':
            logit = self.cos_similarity(inputs, centroids)
         
        
        return logit

    def l2_similarity(self, A, B):
        # input A: [bs, fd] (batch_size x feat_dim)
        # input B: [nC, fd] (num_classes x feat_dim)
        feat_dim = A.size(1)

        AB = torch.mm(A, B.t())
        AA = (A**2).sum(dim=1, keepdim=True)
        BB = (B**2).sum(dim=1, keepdim=True)
        dist = AA + BB.t() - 2*AB

        return -dist
    
    def cos_similarity(self, a, b):
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / a_n
        b_norm = b / b_n
        cos_dist = torch.mm(a_norm, b_norm.transpose(0, 1))
        return cos_dist
    


def create_model(feat_dim, num_classes=1000, feat_type='l2n', dist_type='cos',
                 log_dir=None, test=False, path=None, eval_phase=None,  centroids=None, *args):
    print(feat_dim, num_classes, feat_type, dist_type)
    clf = KNNClassifier(feat_dim, num_classes, feat_type, dist_type)
    return clf 

    if eval_phase=='updated_centroids':
        print('Loading first KNN Classifier using updated centroids')
        clf = KNNClassifier(feat_dim, num_classes, feat_type, dist_type)
        clf.load_memory(log_dir, path)
    else:
        clf = KNNClassifier(feat_dim, num_classes, feat_type, dist_type)
        if centroids is not None:
            clf.update(centroids)
        else:
            if log_dir is not None:
                fname = os.path.join(log_dir, '{}_final.pkl'.format(path))
                if os.path.exists(fname):
                    print('===> Loading features from %s' % fname)
                    with open(fname, 'rb') as f:
                        data = pickle.load(f)
                    clf.update(data)
                    print("Loading second KNN Classifier using final centroids")
    # else:
    #     print('Random initialized KNN classifier weights.')
    
    return clf


if __name__ == "__main__":
    def l2_similarity(A, B):
        # input A: [bs, fd] (batch_size x feat_dim)
        # input B: [nC, fd] (num_classes x feat_dim)
        feat_dim = A.size(1)

        AB = torch.mm(A, B.t())
        AA = (A**2).sum(dim=1, keepdim=True)
        BB = (B**2).sum(dim=1, keepdim=True)
        dist = AA + BB.t() - 2*AB
        return dist
    a = torch.tensor([[5.0,1.0],[2.0,1.],[2.,2.]])
    b = torch.tensor([[1.,0.],[1.,1.],[1.,1.]])
    print(l2_similarity(a,b))
    print(torch.mean(nn.PairwiseDistance(p=2)(a,b)))
    print(nn.MSELoss()(a,b))