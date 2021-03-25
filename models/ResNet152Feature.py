from models.ResNetFeature import *
from utils import *
from torchvision import models
        
def create_model(use_modulatedatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, caffe=False, test=False, *args):
    
    print('Loading Scratch ResNet 152 Feature Model.')
    resnet152 = ResNet(Bottleneck, [3, 8, 36, 3], use_modulatedatt=use_modulatedatt, use_fc=use_fc, dropout=None)
    
   
    
    if not test:
        pretrained = models.resnet152(pretrained=True)

        weights = {k: pretrained.state_dict()[k] if k in pretrained.state_dict().keys() else resnet152.state_dict()[k] for k in resnet152.state_dict() }
        resnet152.load_state_dict(weights)

    return resnet152
