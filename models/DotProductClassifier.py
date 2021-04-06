import torch.nn as nn
from utils import *
import torch.nn.functional as F

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class DotProduct_Classifier(nn.Module):
    
    def __init__(self, num_classes=1000, feat_dim=2048, use_norm=False, *args):
        super(DotProduct_Classifier, self).__init__()
        if use_norm:
            self.fc = NormedLinear(feat_dim, num_classes)
            self.apply(_weights_init)
        else:
            self.fc = nn.Linear(feat_dim, num_classes)
    

    def forward(self, x, *args):
        x = self.fc(x)
        return x
    
def create_model(feat_dim, num_classes=1000, stage1_weights=False, dataset=None, use_norm=False, test=False, *args):
    print('Loading Dot Product Classifier.')
    clf = DotProduct_Classifier(num_classes, feat_dim, use_norm=use_norm, *args)
    
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