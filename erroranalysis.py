import os
import argparse
import pprint
import torch
from data import dataloader
from run_networks import model
import warnings
from PIL import Image
from models.ResNet10Feature import create_model
from utils import source_import
import pandas as pd
import torchvision.models as models

MixUp = True

baseline_path = './logs/ImageNet_LT/stage1/nosampler_baseline.pth'
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
import time
import numpy as np
import warnings
import pdb
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES']='0'
class model ():

    def __init__(self, config, data, test=False):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.training_opt = self.config['training_opt']
        self.memory = self.config['memory']
        self.data = data
        self.test_mode = test

        # Initialize model
        self.init_models()

        # Set up log file
        self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
        if os.path.isfile(self.log_file):
            os.remove(self.log_file)

    def init_models(self, optimizer=True):

        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_list = []

        print("Using", torch.cuda.device_count(), "GPUs.")

        for key, val in networks_defs.items():

            # Networks
            def_file = val['def_file']
            model_args = list(val['params'].values())
            model_args.append(self.test_mode)

            self.networks[key] = source_import(def_file).create_model(*model_args)
            self.networks[key] = nn.DataParallel(self.networks[key]).to(self.device)

            if 'fix' in val and val['fix']:
                print('Freezing feature weights except for modulated attention weights (if exist).')
                for param_name, param in self.networks[key].named_parameters():
                    # Freeze all parameters except self attention parameters
                    if 'modulatedatt' not in param_name and 'fc' not in param_name:
                        param.requires_grad = False

            # Optimizer list
            optim_params = val['optim_params']
            self.model_optim_params_list.append({'params': self.networks[key].parameters(),
                                                 'lr': optim_params['lr'],
                                                 'momentum': optim_params['momentum'],
                                                 'weight_decay': optim_params['weight_decay']})
    def load_model(self):

        print('Validation on the best model.')
        print('Loading model from %s' % (baseline_path))

        checkpoint = torch.load(baseline_path)
        model_state = checkpoint['state_dict_best']

        for key, model in self.networks.items():

            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            # model.load_state_dict(model_state[key])
            model.load_state_dict(weights)
# ================
# LOAD CONFIGURATIONS
# python main.py --config ./config/ImageNet_LT/stage_1.py
data_root = {'ImageNet': '/mnt/lizhaochen', #change this
             'Places': '/home/public/dataset/Places365'}

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/ImageNet_LT/stage_1.py', type=str)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
parser.add_argument('--col', type=str)
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
data_loader = dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')], dataset=dataset, phase='val',
                                batch_size=1,
                                num_workers=training_opt['num_workers'])
baseline_model = model(config, data_loader, test=False)
baseline_model.load_model()

for model in baseline_model.networks.values():
    model.eval()

cls_weight = baseline_model.networks['classifier'].module.fc.weight
baseline_model.networks['classifier'].module.fc.bias


cls_weight = cls_weight.norm(dim=1).tolist()
df = pd.read_csv("./analysis/classifier_weight_norm.csv")
df[args.col] = cls_weight 
df.to_csv("./analysis/classifier_weight_norm.csv",index=False)
