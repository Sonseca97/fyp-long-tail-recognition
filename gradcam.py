import os
import argparse
import pprint
from data import dataloader
from run_network_gc import model
import warnings
import cv2
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from utils import source_import
import random
from torchsummary import summary
'''
    TO SOLVE ISSUE with output feature size at GRADCAM
'''
MixUp = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
baseline_path = './logs/ImageNet_LT/stage1/baseline.pth'
mixup_path = './logs/ImageNet_LT/stage1/mixup_alpha1_randomerasing_model_checkpoint.pth'
mixup_only_path = './logs/ImageNet_LT/stage1/mixup_alpha1_model_checkpoint.pth'
data_root = {'ImageNet': '/mnt/lizhaochen', #change this
             'Places': '/home/public/dataset/Places365'}

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/ImageNet_LT/stage_1.py', type=str)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
args = parser.parse_args()

test_mode = args.test
test_open = args.test_open
if test_open:
    test_mode = True
output_logits = args.output_logits

config = source_import(args.config).config
training_opt = config['training_opt']
# change
relatin_opt = config['memory']
dataset = training_opt['dataset']

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])

# print('Loading dataset from: %s' % data_root[dataset.rstrip('_LT')])
# pprint.pprint(config)

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_forshow = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224)
])

class Flatten(nn.Module):
    """One layer module that flattens its input."""
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)
'''
    GradCam implementation from https://github.com/eclique/pytorch-gradcam/blob/master/gradcam.ipynb
'''
def GradCAM(img, c, features_fn, classifier_fn, cf=False):
    feats = features_fn(img.cuda())
    _, N, H, W = feats.size()

    out = classifier_fn(feats)
    c_score = out[0, c] #logit
    grads = torch.autograd.grad(c_score, feats)
    if cf:
        neg_grads = grads[0][0] * (-1)
        w = neg_grads.mean(-1).mean(-1)
    else:
        w = grads[0][0].mean(-1).mean(-1)
    sal = torch.matmul(w, feats.view(N, H*W))
    sal = sal.view(H, W).cpu().detach().numpy() # 8,8
    sal = np.maximum(sal, 0)
    return sal

def test(features_fn, classifier_fn, cls):
    image = Image.open('/mnt/lizhaochen/train/n01580077/n01580077_9092.JPEG')
    input = data_transform(image).unsqueeze(0).requires_grad_(True)
    sal = GradCAM(input, 18, features_fn, classifier_fn)
    sal = Image.fromarray(sal)
    sal = sal.resize((224, 224), resample=Image.LINEAR)
    plt.imshow(transform_forshow(image))
    plt.imshow(np.array(sal), alpha=0.4, cmap='jet')
    plt.savefig('test1.jpg')

if not test_mode:
    sampler_defs = training_opt['sampler']
    if sampler_defs:
        sampler_dic = {'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                       'num_samples_cls': sampler_defs['num_samples_cls']}
    else:
        sampler_dic = None

    data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')], dataset=dataset, phase=x,
                                    batch_size=128,
                                    sampler_dic=sampler_dic,
                                    num_workers=training_opt['num_workers'],
                                    shuffle=False) for x in  ['train', 'val']}

    baseline_model = model(config, data, test=False)
    baseline_model.load_model(baseline_path)
    feature_b = nn.Sequential(*list(baseline_model.networks['feat_model'].module.children())[:-1])
    avgpool = [list(baseline_model.networks['feat_model'].module.children())[-1],Flatten()]
    classifier_b = nn.Sequential(*(avgpool+list(baseline_model.networks['classifier'].module.children())))

    mixup_model = model(config, data, test=False)
    mixup_model.load_model(mixup_only_path)
    feature_m = nn.Sequential(*list(mixup_model.networks['feat_model'].module.children())[:-1])
    avgpool_m = [list(mixup_model.networks['feat_model'].module.children())[-1],Flatten()]
    classifier_m = nn.Sequential(*(avgpool_m+list(mixup_model.networks['classifier'].module.children())))

    mixup_ran_model = model(config, data, test=False)
    mixup_ran_model.load_model(mixup_path)
    feature_m_r = nn.Sequential(*list(mixup_ran_model.networks['feat_model'].module.children())[:-1])
    avgpool_m_r = [list(mixup_ran_model.networks['feat_model'].module.children())[-1],Flatten()]
    classifier_m_r = nn.Sequential(*(avgpool_m_r+list(mixup_ran_model.networks['classifier'].module.children())))

# test(feature_b, classifier_b)
random.seed(30)
count = 0
lines = open('allcorrect.txt').readlines()
random.shuffle(lines)

for line in lines:
    count += 1
    l = line.split()
    path, gt = l[0], l[1]
    image = Image.open(path)
    input = data_transform(image).unsqueeze(0).requires_grad_(True)
    sal = GradCAM(input, int(gt), feature_m_r, classifier_m_r)
    sal = Image.fromarray(sal)
    sal = sal.resize((224, 224), resample=Image.LINEAR)
    plt.imshow(transform_forshow(image))
    plt.imshow(np.array(sal), alpha=0.4, cmap='jet')
    plt.savefig('./allcorrect/mixup_ran_'+path.split('/')[-1].split('.')[0]+'_'+gt+'.jpg')
    if count == 10:
        break
exit()
with open('allcorrect.txt', 'r') as f:
    data = f.readlines()

    for line in data:
        count += 1
        l = line.split()
        path, gt = l[0], l[1]
        image = Image.open(path)
        input = data_transform(image).unsqueeze(0).requires_grad_(True)
        sal = GradCAM(input, int(gt), feature_m_r, classifier_m_r)
        sal = Image.fromarray(sal)
        sal = sal.resize((224, 224), resample=Image.LINEAR)
        plt.imshow(transform_forshow(image))
        plt.imshow(np.array(sal), alpha=0.4, cmap='jet')
        plt.savefig('./allcorrect/mixup_ran_'+path.split('/')[-1].split('.')[0]+'_'+gt+'.jpg')
        if count == 10:
            break
    # preds, gts, paths, low_class_list = baseline_model.eval()
    # # print("preds:", preds.shape)
    # # print("ground truth:", gts.shape)
    # # print("paths: ",len(paths))
    # print("low class list: ", low_class_list)
    #
    # mixup_model = model(config, data, test=False)
    # mixup_model.load_model(mixup_path)
    # preds_m, gts_m, paths_m, _ = mixup_model.eval()
    # preds_m_r, gets_m_r, paths_m_r, _= mixup_ran_model.eval()

    # image_path_b0m1 = []
    # image_path_b0m0 = []
    # image_path_b1m0 = []
    #
    # all_correct = []
    # for i in range(20000):
    #     if gts[i] in low_class_list:
    #         if preds[i] == gts[i] and preds[i] == preds_m[i] and preds_m[i] == preds_m_r[i]:
    #             all_correct.append(paths[i]+' '+str(gts[i]))
    #         # elif preds[i] != gts[i] and preds_m[i] != gts[i]:
    #         #     image_path_b0m0.append(paths[i]+' '+str(preds[i])+' '+str(preds_m[i])+' '+str(gts[i]))
    #         # elif preds[i] == gts[i] and preds_m[i] != gts[i]:
    #         #     image_path_b1m0.append(paths[i]+' '+str(preds_m[i])+' '+str(gts[i]))
    #
    # with open('allcorrect.txt', 'w') as f:
    #     for i in all_correct:
    #         f.write('%s\n' % i)
    # # with open('b1m0.txt', 'w') as f:
    # #     for i in image_path_b1m0:
    # #         f.write('%s\n' % i)
    # # with open('b0m0.txt', 'w') as f:
    # #     for i in image_path_b0m0:
    # #         f.write('%s\n' % i)



print('ALL COMPLETED.')
