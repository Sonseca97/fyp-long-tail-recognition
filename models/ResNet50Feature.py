from models.ResNetFeature import *
from utils import *
import torch 

def create_model(use_modulatedatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, test=False, *args):
    
    print('Loading Scratch ResNet 50 Feature Model.')
    resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], use_modulatedatt=use_modulatedatt, use_fc=use_fc, dropout=None)

    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 ResNet 50 Weights.' % dataset)
            resnet50 = init_weights(model=resnet50,
                                    weights_path='./logs/%s/stage1/final_model_checkpoint.pth' % dataset)
        else:
            print('No Pretrained Weights For Feature Model.')

    return resnet50


if __name__ == '__main__':
    model = create_model()
    
    x = torch.rand(64, 3, 224, 224)
    model(x)