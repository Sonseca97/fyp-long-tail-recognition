import torch.nn as nn
from utils import *
import torch.nn.functional as F

class DotProduct_Classifier(nn.Module):
    
    def __init__(self, num_classes=1000, feat_dim=2048, *args):
        super(DotProduct_Classifier, self).__init__()
        print(args)
        
        if args[-1] == True:
            # Normalize weight
            self.weight_norm = True
            self.weight = nn.Parameter(torch.Tensor(feat_dim, num_classes))  # (input,output)
            nn.init.xavier_uniform_(self.weight)
            self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
            self.s = 20
        else:
            self.weight_norm = False
            self.fc = nn.Linear(feat_dim, num_classes)
    
        
    def forward(self, x, *args):
        
        if self.weight_norm:
            x = x.mm(F.normalize(self.weight, dim=0))
        else:
            x = self.fc(x) # 128 x 1000
        return x
    
def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None, test=False, *args):
    print('Loading Dot Product Classifier.')
    clf = DotProduct_Classifier(num_classes, feat_dim, *args)
    
    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 Classifier Weights.' % dataset)
            clf.fc = init_weights(model=clf.fc,
                                  weights_path='./logs/%s/stage1/30_baseline.pth' % dataset,
                                  classifier=True)
        else:
            print('Random initialized classifier weights.')

    return clf