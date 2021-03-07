# test remote connection
# http://www.image-net.org/download/synset?wnid=[wnid]&username=[username]&accesskey=[accesskey]
# 423c575d00858640ad48c3b77729db64d1ee722f
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle
from tqdm import tqdm
from models import DotProductClassifier, DistClassifier, KNNClassifier, AssignmentModule, LogitsWeight
from loss.ContrastiveLoss import ContraLoss
from loss.LDAMLoss import LDAMLoss
from utils import *
from mixuputils import *
import time
import numpy as np
import pandas as pd
import warnings
import pdb
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.metrics import (
    precision_recall_curve, 
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)
from data.dataloader import ASM_Dataset
from data.ClassAwareSampler import get_sampler
from thop import clever_format, profile


deterministic = False
if deterministic:
    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

class model ():

    def __init__(self, args, config, data, test=False):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.config = config
        self.memory = self.config['memory']
        self.training_opt = self.config['training_opt']
        self.data = data
        self.test_mode = test
        self.args = args
        self.epoch = None

        # initialize initial centroids
        self.centers = None
        self.logspace = np.geomspace(0.01, 1, num=89)
        # add label info tail, median, head
        if self.config['label_info'] is not None:
            self.tail = config['label_info'][0]
            self.median = config['label_info'][1]
            self.head = config['label_info'][2]
            self.label_counts = config['label_counts']
            self.label_dict = {k:v for k, v in enumerate(self.label_counts)}
            self.df = pd.Series(self.label_dict).to_frame().rename(columns={0: 'label_count'})

            self.class_prior = torch.tensor(list(self.label_dict.values())) / sum(self.label_counts)
            self.inverse_freq = 1 / torch.tensor(list(self.label_dict.values()))
            target_range = torch.linspace(0, 1, steps=1000)
            self.scaled_inverse_freq = scaling(self.inverse_freq.unsqueeze(0), target_range.unsqueeze(0))
            self.scaled_class_prior = scaling(self.class_prior.unsqueeze(0), target_range.unsqueeze(0))

        self.mseloss = nn.MSELoss()
        self.cosloss = nn.CosineEmbeddingLoss()
        self.contraloss = ContraLoss()
        self.kl_div_loss = nn.KLDivLoss(reduction='batchmean')
        # self.ldamloss = LDAMLoss(cls_count = self.label_counts)
        self.dloss_weight = None
        if self.args.second_fc:
            self.distclassifier = DistClassifier.create_model(feat_dim=self.training_opt['feature_dim'])
        if self.args.mixup:
            print("Using MixUp")

        expname_list = [
            self.args.dataset, 
            str(self.training_opt['num_epochs']), 
            self.args.mixup_type if self.args.mixup else None, 
            'manifold_mixup' if self.args.manifold_mixup else None,
            'resample' if self.training_opt['sampler'] is not None else None,
            'second_fc' if self.args.second_fc else None,
            'M_from_{}'.format(str(self.args.m_from)) if self.args.second_fc else None,
            'normalize' if self.args.feat_norm else None,
            self.args.dloss if self.args.second_fc else None,
            str(self.args.lam) if self.args.second_fc else None,
            str(self.args.alpha) if self.args.second_fc else None,
            str(self.args.secondlr) if self.args.second_fc else None,
            str(self.args.description) if self.args.description is not None else None,
            'merge_logits' if self.args.merge_logits else None,
            str(self.args.mixup_alpha) if self.args.mixup else None
        ]
        self.args.expname = '_'.join(elem for elem in expname_list if elem is not None)
        if self.args.path is not None:
            self.args.expname = self.args.path
     
        # mk log dir
        dir_name = os.path.join(self.training_opt['log_dir'], self.args.expname)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        self.training_opt['log_dir'] = dir_name
            
        if self.args.tensorboard:
            self.writer = SummaryWriter(log_dir=os.path.join('runs',\
                self.args.expname+'_{}'.format(self.args.asm_description) if self.args.asm_description \
                else self.args.expname))
        
        # initialize memory bank
        self.normalized_features_centroids = None

        print('Using steps for training.')
        self.training_data_num = len(self.data['train'].dataset)
        self.epoch_steps = int(self.training_data_num  \
                                / self.training_opt['batch_size'])
        
        if self.args.assignment_module:
            self.asm_total_linear_logits = torch.empty((0, self.training_opt['num_classes']))
            self.asm_total_knn_logits = torch.empty((0, self.training_opt['num_classes']))
            self.asm_total_features = torch.empty((0, self.training_opt['feature_dim']))
            self.asm_total_labels = torch.empty(0, dtype=torch.long)
        # Initialize model
        self.init_models()
        # Under training mode, initialize training steps, optimizers, schedulers, criterions, and centroids
        if not self.test_mode:

            # If using steps for training, we need to calculate training steps
            # for each epoch based on actual number of training data instead of
            # oversampled data number
            print('Using steps for training.')
            self.training_data_num = len(self.data['train'].dataset)
            self.epoch_steps = int(self.training_data_num  \
                                   / self.training_opt['batch_size'])

            # Initialize model optimizer and scheduler
            print('Initializing model optimizer.')
            self.scheduler_params = self.training_opt['scheduler_params']
            self.model_optimizer, \
            self.model_optimizer_scheduler = self.init_optimizers(self.model_optim_params_list)
            self.init_criterions()
            if self.memory['init_centroids']:
                self.criterions['FeatureLoss'].centroids.data = \
                    self.centroids_cal(self.data['train_plain'])

            # Set up log file
            if self.args.assignment_module:
                self.log_file = os.path.join(self.training_opt['log_dir'], 'log_assignment_module.txt')
            else:
                self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
        
        self.per_cls_attention_weight = [0] * 1000
        self.mixup_function_bank = {
            'mixup_head': mixup_head,
            'mixup_tail': mixup_tail,
            'mixup_original': mixup,
            'mixup_tail_enrich': mixup_tail_enrich,
            'mixup_head_enrich': mixup_head_enrich,
            'mixup_rank': mixup_rank
        }
        if self.args.mixup:
            self.mixup_function = self.mixup_function_bank[self.args.mixup_type]
        

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
            model_args.append(self.args.expname)
            model_args.append(self.args.weight_norm)
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
        if self.args.trainable_logits_weight:
            self.load_model()
            for key, model in self.networks.items():
                for params in model.parameters():
                    params.requires_grad = False
            fname_final = os.path.join(self.training_opt['log_dir'], '{}_final.pkl'.format(self.args.expname))

            assert os.path.isfile(fname_final), "cannot find final centroids!"
            with open(fname_final, 'rb') as f:
                data = pickle.load(f)
            self.centers =  torch.from_numpy(data['l2ncs']).cuda()

            self.networks['w1'] = nn.DataParallel(LogitsWeight.create_model(num_classes=1000)).to(self.device)
            self.networks['w2'] = nn.DataParallel(LogitsWeight.create_model(num_classes=1000)).to(self.device)
            self.model_optim_params_list.append({
                'params': self.networks['w1'].parameters(),
                'lr': optim_params['lr'],
                'momentum': optim_params['momentum'],
                'weight_decay': optim_params['weight_decay']
            })
            self.model_optim_params_list.append({
                'params': self.networks['w2'].parameters(),
                'lr': optim_params['lr'],
                'momentum': optim_params['momentum'],
                'weight_decay': optim_params['weight_decay']
            })

        if self.args.second_fc:
            self.networks['second_fc'] = nn.DataParallel(self.distclassifier).to(self.device)
            self.model_optim_params_list.append({
                'params': self.networks['second_fc'].parameters(),
                'lr': self.args.secondlr,
                'momentum': 0.9,
                'weight_decay': 0.0005
            })
        
        # non-parametric knn classifier
        self.knnclassifier = KNNClassifier.create_model(self.training_opt['feature_dim'], feat_type=self.args.feat_type, dist_type=self.args.dist_type,
                                                                log_dir=self.training_opt['log_dir'], test=True, 
                                                                path=self.args.expname, norm_input=True)

        if self.args.assignment_module:
            self.load_model()
            for key, model in self.networks.items():
                for params in model.parameters():
                    params.requires_grad = False
            self.networks['asm'] = nn.DataParallel(AssignmentModule.create_model()).to(self.device)
            # self.asm = AssignmentModule.create_model().to(self.device)
            self.model_optim_params_list.append({
                'params': self.networks['asm'].parameters(),
                'lr': 0.1,
                'momentum': 0.9,
                'weight_decay': 0.0005
            })
            # self.asm_optimizer = optim.SGD(self.asm_model_params)
            self.args.feat_norm = True
            fname_final = os.path.join(self.training_opt['log_dir'], '{}_final.pkl'.format(self.args.expname))
            assert os.path.isfile(fname_final), "cannot find final centroids!"
            with open(fname_final, 'rb') as f:
                data = pickle.load(f)
            self.centers =  torch.from_numpy(data['l2ncs']).cuda()
            self.construct_asm_loader()
        
        if self.args.second_dotproduct:
            self.networks['second_dot_product'] = nn.DataParallel(DotProductClassifier.create_model(*model_args)).to(self.device)
            self.model_optim_params_list.append({
                'params': self.networks['second_dot_product'].parameters(),
                'lr': 0.1,
                'momentum': 0.9,
                'weight_decay': 0.0005
            })
        
        if self.args.memory_bank:
            fname = os.path.join(self.training_opt['log_dir'], '{}_memorybank.pkl'.format(self.args.expname))
            if os.path.isfile(fname):
                print("Loading memory bank from {}".format(fname))
                with open(fname, 'rb') as f:
                    self.memorybank  = pickle.load(f)
    
            else:
                self.memorybank = self.get_memory_bank()
                print('===> Saving memory_bank to %s' %
                        os.path.join(self.training_opt['log_dir'], '{}_memorybank.pkl'.format(self.args.expname)))
                with open(os.path.join(self.training_opt['log_dir'], '{}_memorybank.pkl'.format(self.args.expname)), 'wb') as f:
                    pickle.dump(self.memorybank, f)
                print('Done')
        

        ## --------count FLOP-------
        # for key, model in self.networks.items():
        #     x = torch.rand(1,3,224,224).to(self.device)
        #     print(profile(model, inputs=(x,)))
        #     exit()



    def init_criterions(self):

        criterion_defs = self.config['criterions']
        self.criterions = {}
        self.criterion_weights = {}

        for key, val in criterion_defs.items():
            # print(val, val['optim_params'])
            def_file = val['def_file']
            loss_args = val['loss_params'].values()
            # print(def_file)
            self.criterions[key] = source_import(def_file).create_loss(*loss_args).to(self.device)
            self.criterion_weights[key] = val['weight']

            if val['optim_params']:
                print('Initializing criterion optimizer.')
                optim_params = val['optim_params']
                optim_params = [{'params': self.criterions[key].parameters(),
                                'lr': optim_params['lr'],
                                'momentum': optim_params['momentum'],
                                'weight_decay': optim_params['weight_decay']}]
                # Initialize criterion optimizer and scheduler
                self.criterion_optimizer, \
                self.criterion_optimizer_scheduler = self.init_optimizers(optim_params)
            else:
                self.criterion_optimizer = None
    
    def init_optimizers(self, optim_params):
        optimizer = optim.SGD(optim_params)
        if self.args.lr_scheduler == 'cos':
            print("Using Cosine Decay Learning Rate")
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, self.training_opt['num_epochs'], eta_min=0.0)
        else:
            print("Using Step Decay Learning Rate")
            scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=self.scheduler_params['step_size'],
                                                  gamma=self.scheduler_params['gamma'])
        return optimizer, scheduler

    def construct_asm_loader(self):
        for model in self.networks.values():
            model.eval()

        torch.cuda.empty_cache()

        for step, (inputs, labels, _) in tqdm(enumerate(self.data['train'])):
            # Break when step equal to epoch step
            if step == self.epoch_steps: #or image_t==self.training_data_num:
                break
        
            inputs, labels = inputs.to(self.device), labels.to(self.device) # 128, 3, 224, 224
            self.asm_total_labels = torch.cat([self.asm_total_labels, labels.cpu()])
            # If on training phase, enable gradients
            with torch.set_grad_enabled(True):
                # If training, forward with loss, and no top 5 accuracy calculation
                self.features, self.feature_maps, self.attention_weight = self.networks['feat_model'](inputs)
                self.normalized_features = matrix_norm(self.features)
                self.asm_total_features = torch.cat([self.asm_total_features, self.normalized_features.cpu()])
                self.logits = self.networks['classifier'](self.normalized_features*20)
                self.asm_total_linear_logits = torch.cat([self.asm_total_linear_logits, self.logits.cpu()])
                self.logits_dist = self.knnclassifier(self.fc_features if self.args.second_fc else self.features, self.centers)
                # topk, _ = torch.topk(self.logits_dist, k=self.top_k, dim=1)
                if self.args.scaling_logits:
                    self.logits_dist = scaling(self.logits_dist, self.logits)
                self.asm_total_knn_logits = torch.cat([self.asm_total_knn_logits, self.logits_dist.cpu()])

        _, linear_preds = F.softmax(self.asm_total_linear_logits.detach(), dim=1).max(dim=1)
        _, knn_preds = F.softmax(self.asm_total_knn_logits.detach(), dim=1).max(dim=1)
        
        correctness_linear = linear_preds == self.asm_total_labels
        correctness_knn = knn_preds == self.asm_total_labels

        targets = (~correctness_linear)&correctness_knn

        set_ = ASM_Dataset(self.asm_total_features, self.asm_total_linear_logits, self.asm_total_knn_logits, targets.cpu())
       
        sampler = get_sampler()(set_, 128)
        
        self.asm_dataloader = DataLoader(dataset=set_,
                                         batch_size=256,
                                        sampler=sampler, 
                                        num_workers=4)
  
    def batch_forward (self, inputs, labels=None, centroids=False, feature_ext=False, phase='train', eval_phase=None):
        '''
        This is a general single batch running function.
        '''
        self.inputs_augment = None
        # Calculate Features
        self.features, self.feature_maps, self.attention_weight = self.networks['feat_model'](inputs)
        
        # for i in range(len(self.attention_weight)):
        #     self.per_cls_attention_weight[labels[i].item()] += self.attention_weight[i].item()
        # using manifold mixup
        if self.args.manifold_mixup:
            self.mixed_feature, self.targets_a, self.targets_b,\
            self.lam, self.count, _ = mixup(self.features, labels, self.tail, self.count)
        
        # norm feature
        if self.args.feat_norm:
            self.normalized_features = matrix_norm(self.features)

        # using second head
        if self.args.second_fc:
            self.fc_features = self.networks['second_fc'](self.normalized_features if self.args.feat_norm else self.features)
        
        # If not just extracting features, calculate logits
        # Almost always false except get_knn centroids
        if not feature_ext:
            # calculate logits
            if self.args.manifold_mixup:
                self.logits = self.networks['classifier'](self.mixed_feature)

            else:
                self.logits = self.networks['classifier'](self.normalized_features*20 if self.args.feat_norm else self.features)

            self.correctness_linear = self.logits.argmax(dim=1) == labels
            self.linear_output = F.log_softmax(self.logits / self.args.temperature, dim=1) 
            
            if self.args.second_dotproduct:
                self.second_logits = self.networks['second_dot_product'](self.features.detach())
                self.second_linear_output = F.log_softmax(self.second_logits/self.args.temperature, dim=1)
            


            if self.centers is not None:
                self.logits_dist = self.knnclassifier(self.fc_features if self.args.second_fc else self.features, self.centers)
                eta = torch.max(self.logits, dim=1)[0] / torch.max(self.logits_dist, dim=1)[0]
                # topk, _ = torch.topk(self.logits_dist, k=self.top_k, dim=1)
                if self.args.scaling_logits:
                    self.logits_dist = scaling(self.logits_dist, self.logits)
                self.correctness_knn = self.logits_dist.argmax(dim=1) == labels
                self.target_one_hot = self.correctness_knn
                # print(self.target_one_hot.unsqueeze(1).shape)
                # knn_output all 0.001
                self.knn_output = F.softmax(self.logits_dist / self.args.temperature, dim=1)
          
            if self.args.assignment_module:
                self.asm_output, self.asm_target = self.networks['asm'](self.normalized_features, self.logits, self.logits_dist, labels)
                # print(self.asm_target.sum())
                # self.pos_weight = (torch.nonzero(self.asm_target==False).shape[0]) / (torch.nonzero(self.asm_target==True).shape[0]) \
                #                     if torch.nonzero(self.asm_target==True).shape[0] != 0 else 100
                # self.pos_weight = 112095 / 3751

                if phase != 'train':
                    self.knn_correctness = torch.sigmoid(self.asm_output) > 0.5
         

            # if self.args.klloss:
            #     self.prob_gt = F.softmax(self.knnclassifier.l2_similarity(self.features, self.centers).detach(), dim=1)
            
            if self.args.merge_logits and self.centers is not None and phase != 'test':
                if self.args.log_w:
                    self.args.w1 = 1 - self.logspace[self.epoch - 2]
                    self.args.w2 = self.logspace[self.epoch - 2]
                if self.args.trainable_logits_weight:
                    self.logits = self.networks['w1'](self.logits) + self.networks['w2'](self.logits_dist.detach())
                   
                # self.logits = self.args.w1 * self.logits + self.args.w2 * self.logits_dist.detach()


            if phase != 'train':
                if eval_phase == 'softmax':
                    pass 
                elif eval_phase == 'second dot product':
                    self.logits = self.second_logits
                elif eval_phase == 'merge_logits':
                    if self.args.log_w:
                        self.args.w1, self.args.w2 = 0.5, 0.5
                    # self.args.w2 = self.scaled_inverse_freq.squeeze(0)[labels].unsqueeze(1).cuda() * 40
                    '''
                        used for trainable logits weight
                    '''
                    if self.args.trainable_logits_weight:
                        self.logits = self.networks['w1'](self.logits) + self.networks['w2'](self.logits_dist.detach())
                    else:
                        # Fusion Methods
                        self.args.w2 = 1 - self.args.w1
                        # self.logits = self.args.w1 * F.softmax(self.logits, dim=1) + self.args.w2 * F.softmax(self.logits_dist.detach() * eta.unsqueeze(1), dim=1)
                        # self.logits = self.args.w1 * F.softmax(self.logits, dim=1) + self.args.w2 * F.softmax(self.logits_dist.detach(), dim=1)
                        # self.logits = self.args.w1 * self.logits + self.args.w2 * self.logits_dist.detach()
                        # self.logits = self.args.w1 * self.logits + self.args.w2 * self.logits_dist.detach() * eta.unsqueeze(1)
                        self.logits_dist = self.logits_dist.detach() * eta.unsqueeze(1)
                        for idx in range(self.logits.shape[0]):
                            self.logits[idx, :] = torch.stack([self.logits[idx, :], self.logits_dist[idx, :]]).max(dim=0)[0]

                # ncm classifier
                else:
                    self.logits = self.logits_dist

 
    def batch_backward(self):
        # Zero out optimizer gradients
        self.model_optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()
        # Back-propagation from loss outputs
        self.loss.backward()
    
        # Step optimizers
        self.model_optimizer.step()
        if self.criterion_optimizer:
            self.criterion_optimizer.step()
        

    def batch_loss(self, labels, phase='train'):

        if self.args.assignment_module:
        
            self.loss = F.binary_cross_entropy_with_logits(self.asm_output, labels.float().cuda())#, pos_weight=torch.tensor([20.]).cuda())
            self.loss_perf = self.loss
            return
        # print(self.logits)
        # self.loss_perf = nn.CrossEntropyLoss(reduction='none')(self.logits, labels) \
        #             * self.criterion_weights['PerformanceLoss']
        self.loss_perf = self.criterions['PerformanceLoss'](self.logits, labels) \
                    * self.criterion_weights['PerformanceLoss']

        # calculate center loss
        if self.centers is not None and self.args.center_loss:
            if self.args.second_fc and self.args.ce_dist==False:
                # self.center_loss = self.contraloss(self.fc_features, self.centers[labels]) * self.args.lam
                self.center_loss = compute_center_loss(self.fc_features, self.centers, labels, self.args.dloss, self.device) * self.args.lam
            elif self.args.second_fc and self.args.ce_dist:
                if self.args.ldam:
                    self.center_loss = self.ldamloss(self.logits_dist, labels) * self.args.lam
                else:
                    self.center_loss = self.criterions['PerformanceLoss'](self.logits_dist, labels) * self.args.lam
            else:
                self.center_loss = compute_center_loss(self.features, self.centers, labels, self.args.dloss, self.device) * self.args.lam

            if self.args.klloss:
                self.klloss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(self.logits_dist.detach(), dim=1), self.prob_gt)
            else:
                self.klloss = 0
        
        else:
            self.center_loss = 0
            self.klloss = 0

        '''
            Add KL Loss for training
        '''
        if self.args.second_dotproduct and self.centers is not None and phase=='train':
            self.second_head_loss = 0.2 * self.criterions['PerformanceLoss'](self.second_logits, labels) + 0.8 * self.kl_div_loss(self.second_linear_output, self.knn_output) \
                        * self.args.temperature * self.args.temperature
           
        else:
            self.second_head_loss = 0
        if self.args.klloss and self.centers is not None:
            # self.klloss = self.kl_div_loss(self.linear_output, self.knn_output) * self.args.temperature * self.args.temperature
            # print((self.scaled_inverse_freq.t().cuda() * 10)[labels].shape)
            # self.klloss = ((self.scaled_inverse_freq.t().cuda().log())[labels] \
            #             * nn.KLDivLoss(reductiselfon='none')(self.linear_output, self.knn_output)) \
            #             .sum(axis=1).mean() * .args.temperature * self.args.temperature
            # self.klloss = ((self.scaled_class_prior.t().cuda())[labels] * nn.KLDivLoss(reduction='none')(self.linear_output, self.knn_output)).sum(axis=1).mean() * self.args.temperature * self.args.temperature
            # self.klloss = (5 * self.target_one_hot.unsqueeze(1).cuda() * nn.KLDivLoss(reduction='none')(self.linear_output, self.knn_output)).sum(axis=1).mean() * self.args.temperature * self.args.temperature

            self.klloss = (self.target_one_hot.unsqueeze(1).cuda() * nn.KLDivLoss(reduction='none')(self.linear_output, self.knn_output)).sum()/(self.target_one_hot.sum()+1e-5) * self.args.temperature * self.args.temperature
           
        a = 1
        b = 5
        self.loss = self.loss_perf + b * self.klloss + self.center_loss + self.second_head_loss
        # print(self.loss)
    def batch_loss_mixup(self, targets_a, targets_b, lam):
        # mixup criterion
        if 'enrich' in self.args.mixup_type:
            self.loss_perf = mixup_criterion(self.criterions['PerformanceLoss'], self.logits[:32], targets_a[:32], targets_b[:32], lam) \
                            + self.criterions['PerformanceLoss'](self.logits[32:], targets_a[32:])
            
        else:
            self.loss_perf = mixup_criterion(self.criterions['PerformanceLoss'], self.logits, targets_a, targets_b, lam)
   
        self.loss = self.loss_perf

    def train(self):
        # When training the network
        print_str = ['Phase: train']
        # randomly initialize centers
        if self.args.m_from==0:
            print("initializing centroids")
            self.centers = torch.rand(self.training_opt['num_classes'], self.training_opt['feature_dim']).to(self.device) - 0.5

        print_write(print_str, self.log_file)
        time.sleep(0.25)

        # Initialize best model
        best_model_weights = {}
        for key in self.networks.keys():
            best_model_weights[key] = copy.deepcopy(self.networks[key].state_dict())

        best_acc = 0.0
        best_epoch = 0
        best_f1 = 0.0

        end_epoch = self.training_opt['num_epochs']

        # Loop over epochs
        self.count = [0]*1000
        step_total = 0

        if self.args.assignment_module:
            end_epoch = 10
            self.training_opt['display_step'] = 100
        for epoch in range(1, end_epoch + 1):

            self.ce_losses_stats = AverageMeter('ce_loss', ":4e")
            if self.args.second_fc:
                self.kl_loss_stats = AverageMeter('klloss', ":4e")
                self.d_losses_stats = AverageMeter('d_loss', ":4e")
            self.acc_stats = AverageMeter('accuracy', ':6.2f')

            self.epoch=epoch
            '''
                Set Deferred Mixup
            '''
            # if epoch > 70:
            #     self.args.mixup = True
            
            for model in self.networks.values():
                if self.args.assignment_module or self.args.trainable_logits_weight:
                    for module in model.modules():
                        if isinstance(module,  nn.BatchNorm2d):
                            module.eval()
                        else:
                            module.train()
                else:
                    model.train()
                    
            torch.cuda.empty_cache()
            
            # Iterate over dataset
            image_t = 0
            start = 0
            for step, (inputs, labels, _) in enumerate(self.data['train']):
                # Break when step equal to epoch step
                if step == self.epoch_steps: #or image_t==self.training_data_num:
                    break
              
            
                inputs, labels = inputs.to(self.device), labels.to(self.device) # 128, 3, 224, 224
                ori_labels = labels
                # if self.inputs_augment is not None:
                #     inputs = torch.cat((inputs, self.inputs_augment), dim=0)
                #     labels = torch.cat((labels, self.labels_augment), dim=0)
                
                if self.args.mixup:
                    inputs, self.targets_a, self.targets_b, self.lam, self.count, labels = self.mixup_function(inputs, labels, self.label_dict, self.count, alpha=self.args.mixup_alpha)
                    inputs, self.targets_a, self.targets_b = map(Variable, (inputs, self.targets_a, self.targets_b))
                
                image_t += len(labels)
                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):
                    # If training, forward with loss, and no top 5 accuracy calculation
                    self.batch_forward(inputs, labels, phase='train')
                   
                
                    if self.args.mixup or self.args.manifold_mixup and self.lam!=-1:
                        self.batch_loss_mixup(self.targets_a, self.targets_b, self.lam)
                    # elif 'enrich' in self.args.mixup_type and inputs.shape[0]
                    else:
                        self.batch_loss(labels)
                  
                    self.batch_backward()
                    
                    # udpate centroids
                    if self.centers is not None \
                            and self.args.m_freeze==False \
                            and self.args.assignment_module==False \
                            and self.args.trainable_logits_weight==False:

                        centers = self.centers
                        center_deltas = get_center_delta(
                            self.normalized_features.data if self.args.feat_norm else matrix_norm(self.features).data,
                            self.centers, labels, self.args.alpha, self.device)
                        self.centers = centers - center_deltas

                    _, preds = torch.max(self.logits, 1)
                    self.minibatch_acc = mic_acc_cal(preds, labels)
                        
                    # Output minibatch training results
                    if step % self.training_opt['display_step'] == 0:
                        step_total += step
                        minibatch_loss_perf = self.loss_perf.item()
                        if self.centers is not None and self.args.center_loss:
                            center_loss = self.center_loss.item()
                        else:
                            center_loss = 0
                        minibatch_dloss = self.dist_loss.item() if self.normalized_features_centroids is not None else None

                        print_str = ['Epoch: [%d/%d]'
                                     % (epoch, end_epoch),
                                     'Step: %5d'
                                     % (step),
                                     'Minibatch_loss_performance: %.3f'
                                     % (minibatch_loss_perf),
                                     'Minibatch_distance_loss: %.3f'
                                     % (center_loss) if (self.centers is not None and self.args.center_loss) else '',
                                     'Minibatch kl loss: %.3f'
                                     % (self.klloss.item()) if self.centers is not None and self.args.klloss else '',
                                     'Minibatch_accuracy_micro: %.3f'
                                      % (self.minibatch_acc)]
                        print_write(print_str, self.log_file)

                # update stats every iteration
                self.ce_losses_stats.update(self.loss_perf.item(), inputs.size(0))
                if self.args.second_fc and self.centers is not None and self.args.center_loss:
                    self.d_losses_stats.update(self.center_loss.item(), inputs.size(0))
                if self.args.second_fc and self.centers is not None and self.args.klloss:
                    self.kl_loss_stats.update(self.klloss.item(), inputs.size(0))
                self.acc_stats.update(self.minibatch_acc, inputs.size(0))

            # update tensorboard every epoch
            if self.args.tensorboard:
                self.writer.add_scalar('Loss/ce_train', self.ce_losses_stats.avg, epoch)
                if self.args.second_fc:
                    self.writer.add_scalar('Loss/dloss_train', self.d_losses_stats.avg, epoch)
                    if self.args.klloss:
                        self.writer.add_scalar('Loss/kl_loss_train', self.kl_loss_stats.avg, epoch)
                self.writer.add_scalar('Accuracy/Train', self.acc_stats.avg, epoch)

            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train()
            if self.args.assignment_module==False:
                self.model_optimizer_scheduler.step()
                if self.criterion_optimizer:
                    self.criterion_optimizer_scheduler.step()

           
            # After every epoch, validation
            self.eval(phase='val',epoch=epoch)
            # Under validation, the best model need to be updated
            if self.args.assignment_module == False:
                if self.eval_acc_mic_top1 > best_acc:
                    best_epoch = copy.deepcopy(epoch)
                    best_acc = copy.deepcopy(self.eval_acc_mic_top1)
                    best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                    best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
                    if self.args.second_fc:
                        best_model_weights['second_fc'] = copy.deepcopy(self.networks['second_fc'].state_dict())
                    if self.args.trainable_logits_weight:
                        best_model_weights['w1'] = copy.deepcopy(self.networks['w1'].state_dict())
                        best_model_weights['w2'] = copy.deepcopy(self.networks['w2'].state_dict())
                    if self.args.assignment_module:
                        best_model_weights['asm'] = copy.deepcopy(self.networks['asm'].state_dict())
                    if self.args.second_dotproduct:
                        best_model_weights['second_dot_product'] = copy.deepcopy(self.networks['second_dot_product'].state_dict())
            else:
                if self.f1 > best_f1:
                    best_epoch = copy.deepcopy(epoch)
                    best_f1 = copy.deepcopy(self.f1)
                    best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                    best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
                    if self.args.second_dotproduct:
                        best_model_weights['second_dot_product'] = copy.deepcopy(self.networks['second_dot_product'].state_dict())
                    if self.args.second_fc:
                        best_model_weights['second_fc'] = copy.deepcopy(self.networks['second_fc'].state_dict())
                    if self.args.trainable_logits_weight:
                        best_model_weights['w1'] = copy.deepcopy(self.networks['w1'].state_dict())
                        best_model_weights['w2'] = copy.deepcopy(self.networks['w2'].state_dict())
                    if self.args.assignment_module:
                        best_model_weights['asm'] = copy.deepcopy(self.networks['asm'].state_dict())
            self.best_model_weights =copy.deepcopy(best_model_weights)

            # if epoch is m_from: calculate centroids
            # if finetuning, don't calculate
            if epoch==self.args.m_from and self.args.assignment_module==False and self.args.trainable_logits_weight==False:
                self.feat_dict = self.get_knncentroids()
                if self.args.feat_norm:
                    self.centers = torch.from_numpy(self.feat_dict['l2ncs']).to(self.device)
                else:
                    self.centers = torch.from_numpy(self.feat_dict['l2ncs']).to(self.device)
                # for key, model in self.networks.items():
                #     if key != 'second_fc':
                #         for params in model.parameters():
                #             params.requires_grad = False
                # self.model_optim_params_list = [self.model_optim_params_list[-1]]
                # self.model_optimizer = optim.SGD(self.model_optim_params_list)
               



        print()
        print('Training Complete.')
        if self.args.tensorboard:
            self.writer.close()

        # save updated centroids
        if self.centers is not None and self.args.assignment_module==False and self.args.trainable_logits_weight==False:
            print("saving M into pkl file..")
            with open(os.path.join(self.training_opt['log_dir'], '{}.pkl'.format(self.args.expname)), 'wb') as f:
                pickle.dump(self.centers, f)

        if self.args.assignment_module == False:
            print_str = ['Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch)]
            self.save_model(epoch, best_epoch, best_model_weights, best_acc)
        else:
            print_str = ['Best binary classification f1 is %.3f at epoch %d' % (best_f1, best_epoch)]
            self.save_model(epoch, best_epoch, best_model_weights, best_f1)

        print_write(print_str, self.log_file)

        
        # Save the best model and best centroids if calculated
        if self.args.assignment_module==False:
            
            if self.args.trainable_logits_weight == False:
                cfeats = self.get_knncentroids()
                print('===> Saving final features to %s' %
                        os.path.join(self.training_opt['log_dir'], '{}_final.pkl'.format(self.args.expname)))
                with open(os.path.join(self.training_opt['log_dir'], '{}_final.pkl'.format(self.args.expname)), 'wb') as f:
                    pickle.dump(cfeats, f)
                print('Done')
        # bot.send_message("Training Complete, Evaluation Ongoing..")

    def eval(self, phase='val', openset=False, epoch=0):
        print_str = ['Phase: %s' % (phase)]
        if phase != 'test':
            print_write(print_str, self.log_file)
        time.sleep(0.25)

        if self.test_mode:
            self.load_model()

        elif phase=='test':
            for key, model in self.networks.items():
                weights = self.best_model_weights[key]
                weights = {k: weights[k] for k in weights if k in model.state_dict()}
                model.load_state_dict(weights)
        
        if openset:
            print('Under openset test mode. Open threshold is %.1f'
                  % self.training_opt['open_threshold'])

        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()
        
        eval_list = ['softmax']
        eval_dict = {}

        if self.centers is not None and self.args.cal_knn_val and phase=='val':
            eval_list.append('updated_centroids')
        
        if phase=='test' and self.args.assignment_module==False:
            fname_update = os.path.join(self.training_opt['log_dir'], '{}.pkl'.format(self.args.expname))
            if os.path.isfile(fname_update):
                eval_list.append('updated_centroids')
                with open(fname_update, 'rb') as f:
                    data = pickle.load(f)
                eval_dict['updated_centroids'] = data
            fname_final = os.path.join(self.training_opt['log_dir'], '{}_final.pkl'.format(self.args.expname))
            if os.path.isfile(fname_final):
                eval_list.append('final_centroids')
                with open(fname_final, 'rb') as f:
                    data = pickle.load(f)
                eval_dict['final_centroids'] = torch.from_numpy(data['l2ncs']).cuda()
            else:
                print("not found final_centroids") 
                eval_dict['final_centroids'] = torch.from_numpy(self.get_knncentroids()['l2ncs']).cuda()
                eval_list.append('final_centroids')
            if 'merge_logits' in self.args.expname or self.args.merge_logits:
                try:
                    eval_dict['merge_logits'] = eval_dict['final_centroids']
                    eval_list.append("merge_logits")
                except:
                    pass
            if self.test_mode == False:
                with open(os.path.join(self.training_opt['log_dir'],'result.txt'), 'a') as f:
                        f.write(self.args.expname+'\n')
                        f.close()

        if self.args.trainable_logits_weight:
            eval_list = ['merge_logits']
        if self.args.assignment_module:
            eval_list = ['softmax']
        if self.args.second_dotproduct:
            eval_list.append("second dot product")
        for eval_phase in eval_list:
            self.val_loss = AverageMeter('val_loss', ":4e")
            print(eval_phase)
            self.total_logits = torch.empty((0, self.training_opt['num_classes'])).to(self.device)
            self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)
            self.total_logits_dist = torch.empty((0, self.training_opt['num_classes'])).to(self.device)
            self.assignment_pred = torch.empty(0).to(self.device)
            self.total_paths = np.empty(0)
            self.total_features = torch.empty((0, self.training_opt['feature_dim']))

            self.total_targets = torch.empty(0)
            
            if eval_phase not in ['softmax', 'second dot product'] and phase == 'test':
                self.centers = eval_dict[eval_phase]

            # Iterate over dataset
            for inputs, labels, paths in tqdm(self.data[phase]):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # If on training phase, enable gradients
                with torch.set_grad_enabled(False):

                    # In validation or testing
                    self.batch_forward(inputs, labels, phase=phase, eval_phase=eval_phase)
                    
                    if self.test_mode==False:
                       
                        self.batch_loss(labels, phase=phase)
                    
                        self.val_loss.update(self.loss.item(), inputs.size(0))
                        
                    # Uncomment this line to construct memory bank
                    if self.args.memory_bank:
                        self.total_features = torch.cat((self.total_features, self.features.cpu()))
                    self.total_logits = torch.cat((self.total_logits, self.logits if eval_phase in ['softmax', 'merge_logits', 'second dot product'] else self.logits_dist))
                    self.total_labels = torch.cat((self.total_labels, labels))
                    self.total_paths = np.concatenate((self.total_paths, paths))
                    if self.args.assignment_module:
                        self.total_logits_dist = torch.cat((self.total_logits_dist, self.logits_dist))
                        self.assignment_pred = torch.cat((self.assignment_pred, self.asm_output))
                        self.total_targets = torch.cat((self.total_targets, self.asm_target.cpu()))
         
            # df = pd.DataFrame(self.total_logits.detach().tolist(), columns=[i for i in range(1000)])
            # df.to_csv('./analysis/knn_logits.csv')
            # df = pd.DataFrame(F.softmax(self.total_logits.detach(), dim=1).tolist(), columns=[i for i in range(1000)])
            # df.to_csv('./analysis/knn_prob.csv')

            # df = pd.DataFrame(self.total_logits_dot.detach().tolist(), columns=[i for i in range(1000)])
            # df.to_csv('./analysis/dp_logits.csv')
            # df = pd.DataFrame(F.softmax(self.total_logits_dot.detach(), dim=1).tolist(), columns=[i for i in range(1000)])
            # df.to_csv('./analysis/dp_prob.csv')
            # sum of two logits
            # self.sum_logits = self.total_logits_dot + self.total_logits
            
            if eval_phase in ['softmax', 'final_centroids']:
                print("saving all {} logits into pkl file..".format(eval_phase))
                with open(os.path.join(self.training_opt['log_dir'], '{}_logits.pkl'.format(eval_phase)), 'wb') as f:
                    pickle.dump(self.total_logits.cpu().numpy(), f)

            probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)
            _, preds_topk = F.softmax(self.total_logits.detach(), dim=1).topk(k=self.args.k, dim=1, largest=True, sorted=True)
            if self.args.memory_bank:
                with open(os.path.join(self.training_opt['log_dir'], '{}_testlabels.pkl'.format(self.args.expname)), 'wb') as f:
                    pickle.dump(self.total_labels.cpu().numpy(), f)
                print('Done')
                exit()
                pred_score = self.knn_predict(F.normalize(self.total_features, dim=1).cpu(), \
                                        F.normalize(torch.from_numpy(self.memorybank['all_features']), dim=1).t(), \
                                        torch.from_numpy(self.memorybank['all_labels']), \
                                        self.training_opt['num_classes'], \
                                        5, 1)
                preds = pred_score[:,0].to(self.device)
        


            if self.args.assignment_module:
                probs_knn, preds_knn = F.softmax(self.total_logits_dist.detach(), dim=1).max(dim=1)
                _, preds_topk_knn = F.softmax(self.total_logits_dist.detach(), dim=1).topk(k=self.args.k, dim=1, largest=True, sorted=True)
                
                yhat = torch.sigmoid(self.assignment_pred).cpu().numpy()
                ytest = self.total_targets.numpy()
            
                precision, recall, threshold = precision_recall_curve(ytest, yhat)
                fpr, tpr, thresholds = roc_curve(ytest, yhat)
    
                pr_name = str(self.epoch) if phase!='test' else 'test'
                plt.plot(recall ,precision, linestyle='--', label=pr_name)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.legend()
                plt.savefig('PR_curve_{}.png'.format(pr_name))
                plt.clf()

                roc_name = str(self.epoch) if phase!='test' else 'test'
                plt.plot(fpr, tpr, linestyle='--', label=roc_name)
                plt.xlabel('fpr')
                plt.ylabel('tpr')
                plt.legend()
                plt.savefig("ROC_curve_{}.png".format(roc_name))
                plt.clf()

                # fscore = (2 * precision * recall) / (precision + recall)
                # ix = np.argmax(fscore)
                # thresh = threshold[ix]
                # print(thresh)
                self.assignment_pred = torch.sigmoid(self.assignment_pred)> 0.5
                precision_s = precision_score(ytest, self.assignment_pred.cpu())
                recall_s = recall_score(ytest, self.assignment_pred.cpu())
                f1 = f1_score(ytest, self.assignment_pred.cpu())
                self.f1 = f1
                print_write(["precision: {}, recall: {}, f1: {}".format(precision_s, recall_s, f1)], self.log_file)
                knn_idx = torch.nonzero(self.assignment_pred==True).view((-1,))
                print("accuracy: {}".format((self.assignment_pred.cpu()==self.total_targets).sum()/len(self.total_targets)))
                assert len(self.total_targets) == len(self.assignment_pred), "wrong!!"
                idx_pos = torch.nonzero(self.total_targets==True).view((-1,))
                idx_neg = torch.nonzero(self.total_targets==False).view((-1,))
                print("length of 1: ", len(idx_pos))
                print("length of 0: ", len(idx_neg))
                print(self.assignment_pred.cpu()[idx_pos].sum())
                print(len(idx_neg) - self.assignment_pred.cpu()[idx_neg].sum())
                if self.test_mode == False:
                    print_write([get_cls_report(self.total_targets, self.assignment_pred)],self.log_file)
                # print(self.total_targets.sum())
                preds[knn_idx] = preds_knn[knn_idx]
                # print((previous_pred == preds).sum())
                
            
            # if self.knnclassifier is not None:
            #     scaled_knn = F.softmax(self.total_logits.detach(), dim=1)
                
            #     scaled_linear = F.softmax(self.linear_total_logits.detach(), dim=1)
            #     # scaled_knn = scaling(scaled_knn, scaled_linear)
               
            #     probs, preds = scaled_knn.max(dim=1)
            #     mask_tail = [index for index, value in enumerate(self.total_labels) if value.item() in self.tail]
            #     mask_median = [index for index, value in enumerate(self.total_labels) if value.item() in self.median]
            #     mask_head = [index for index, value in enumerate(self.total_labels) if value.item() in self.head]
            #     knn_probs = probs[mask_tail]
            #     # print(probs[mask_tail].min(), probs[mask_median].min(), probs[mask_head].min())
            #     # print(probs[mask_tail].max(), probs[mask_median].max(), probs[mask_head].max())
            #     # exit()
            #     # knn_preds = preds[mask_tail]
            #     # probs, preds = scaled_linear.max(dim=1)
            #     # for i, prob in enumerate(probs):
            #     #     if prob < probs[mask_tail].mean():
            #     #         scaled_linear[i] = scaled_knn[i]
            #     # dp_probs = probs[mask]
            #     # dp_preds = preds[mask]
            #     # gt = self.total_labels[mask]
            #     # print("probs dot product wrong: ")
            #     # print(dp_probs[dp_preds!=gt])
            #     # print(len(dp_probs[dp_preds==gt]))
            #     # exit()
            #     # print("dot product tail ")
            #     # print(scaled_linear[self.tail][:2])
            #     # print("knn tail")
            #     # print(scaled_knn[self.tail][:2])
            #     # exit()

            #     # w_inver = (1 / torch.tensor(self.label_counts)).repeat(50000, 1)
            #     # # print(w_inver)
            #     # target_range = torch.linspace(0, 1, steps=1000).repeat(50000, 1)
            #     # w_inver = scaling(w_inver, target_range) 
            #     # # print(w_inver)
            #     # # exit()
            #     # w_inver = w_inver.to(self.device)
                

            #     sum_logits = self.args.w1 * scaled_knn + self.args.w2 * scaled_linear
            #     # print(F.softmax(sum_logits.detach(), dim=1))
            #     # exit()
            #     probs, preds = F.softmax(sum_logits.detach(), dim=1).max(dim=1)
            #     _, preds_topk = F.softmax(sum_logits.detach(), dim=1).topk(k=self.args.k, dim=1, largest=True, sorted=True)

            # calculate validation cross-entropy loss
            print(self.val_loss.avg)
            if self.args.tensorboard:
                self.writer.add_scalar('Loss/ce_val', self.val_loss.avg, self.epoch)

            if openset:
                preds[probs < self.training_opt['open_threshold']] = -1
                self.openset_acc = mic_acc_cal(preds[self.total_labels == -1],
                                                self.total_labels[self.total_labels == -1])
                print('\n\nOpenset Accuracy: %.3f' % self.openset_acc)

            # Calculate the overall accuracy and F measurement
            self.eval_acc_mic_top1 = mic_acc_cal(preds[self.total_labels != -1],
                                                self.total_labels[self.total_labels != -1])
            # Calculate top k
            self.eval_acc_mic_topk = mic_acc_cal_topk(preds_topk[self.total_labels != -1],
                                                self.total_labels[self.total_labels != -1], self.args.k)

            self.eval_f_measure = F_measure(preds, self.total_labels, openset=openset,
                                            theta=self.training_opt['open_threshold'])
            self.many_acc_top1, \
            self.median_acc_top1, \
            self.low_acc_top1, pred, gt, \
            self.many_acc_topk, \
            self.median_acc_topk, \
            self.low_acc_topk = shot_acc(preds[self.total_labels != -1], preds_topk[self.total_labels != -1],
                                        self.total_labels[self.total_labels != -1],
                                        self.data['train'])
            category_dict = {
                "many": self.many_acc_top1,
                "median": self.median_acc_top1,
                "low": self.low_acc_top1
            }
            if phase=='val' and self.args.tensorboard:
                self.writer.add_scalar("Accuracy/Val",self.eval_acc_mic_top1, epoch)
                self.writer.add_scalars("Accuracy/Val_category", category_dict, epoch)

            # Top-1 accuracy and additional string
            if self.args.acc and self.test_mode:
                data_tuples = list(zip(pred,gt))
                df = pd.DataFrame(data_tuples,columns=['pred','gt'])
                df['acc'] = df['pred']/df['gt']
                final_df = pd.read_csv("./analysis/accuracy.csv")
                final_df[self.args.acc_csv] = df['acc']
                final_df.to_csv("./analysis/accuracy.csv",index=False)

            print_str = ['\n',
                        'Phase: %s'
                        % (phase),
                        '\n',
                        'Evaluation_accuracy_micro_top1: %.3f'
                        % (self.eval_acc_mic_top1),
                        '\n',
                        'Evaluation_accuracy_micro_topk: %.3f'
                        % (self.eval_acc_mic_topk),
                        '\n',
                        'Averaged F-measure: %.3f'
                        % (self.eval_f_measure),
                        '\n',
                        'Many_shot_accuracy_top1: %.3f(%.3f)'
                        % (self.many_acc_top1, self.many_acc_topk),
                        'Median_shot_accuracy_top1: %.3f(%.3f)'
                        % (self.median_acc_top1, self.median_acc_topk),
                        'Low_shot_accuracy_top1: %.3f(%.3f)'
                        % (self.low_acc_top1, self.low_acc_topk),
                        '\n']
            

            if phase == 'test' and self.args.hypersearch:
                with open('hyper.txt', 'a') as f:
                    f.write(str(self.config['dloss_weight']))
                    f.write('\n')
                    f.write(str(print_str))
                    f.write('\n')
                f.close()

            if phase == 'test':
                txt_name = 'result_assignment_module.txt' if self.args.assignment_module else 'result.txt'
                with open(os.path.join(self.training_opt['log_dir'],txt_name), 'a') as f:
                    txt = '---w1 {}, w2 {} ----\n'.format(self.args.w1, self.args.w2)
                    txt += 'dot product softmax' if eval_phase=='softmax' else 'knn dist_type-{} {}'.format(self.args.dist_type, eval_phase)
                    txt += ' '.join(print_str) + '\n'
                    f.write(txt)
                f.close()

            if phase == 'val':
                print_write(print_str, self.log_file)
            else:
                print(*print_str)
            
            # self.per_cls_attention_weight = np.array(self.per_cls_attention_weight)/50
            # df = pd.Series(self.label_dict).to_frame("freq")
            # df['w'] = self.per_cls_attention_weight
            # df.to_csv("sigmoid_weight.csv")
            # tail, median, many = [], [], []
            # for i in range(len(self.per_cls_attention_weight)):
            #     if i in self.tail:
            #         tail.append(self.per_cls_attention_weight[i])
            #     elif i in self.median:
            #         median.append(self.per_cls_attention_weight[i])
            #     else:
            #         many.append(self.per_cls_attention_weight[i])
            # print(np.mean(tail), np.mean(median), np.mean(many))
            # exit()

    def centroids_cal(self, data):

        centroids = torch.zeros(self.training_opt['num_classes'],
                                   self.training_opt['feature_dim']).cuda()

        print('Calculating centroids.')

        for model in self.networks.values():
            model.eval()

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):

            for inputs, labels, _ in tqdm(data):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # Calculate Features of each training data
                self.batch_forward(inputs, feature_ext=True)
                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    centroids[label] += self.features[i]

        # Average summed features with class count
        centroids /= torch.tensor(class_count(data)).float().unsqueeze(1).cuda()

        return centroids

    def load_model(self):
        if self.args.path is None:
            model_dir = os.path.join(self.training_opt['log_dir'], "{}.pth".format(self.args.expname))
        elif self.test_mode:
            if self.args.trainable_logits_weight:
                suffix = '_with_weight'
            elif self.args.assignment_module:
                suffix = '_with_asm'
            else:
                suffix = ''
            model_dir = os.path.join(self.training_opt['log_dir'], "{}{}.pth".format(self.args.path, suffix))
        else:
            model_dir = os.path.join(self.training_opt['log_dir'], "{}.pth".format(self.args.path))
        print('Validation on the best model.')
        print('Loading model from %s' % (model_dir))

        checkpoint = torch.load(model_dir)
        model_state = checkpoint['state_dict_best']
        print(model_state.keys())
        
        self.centroids = checkpoint['centroids'] if 'centroids' in checkpoint else None
        for key, model in self.networks.items():

            print("Loading ", key)
            
            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            # model.load_state_dict(model_state[key])

            # two lines added for knn below
            # x = model.state_dict()
            # x.update(weights)
            model.load_state_dict(weights)
    

    def get_knncentroids(self, second_fc=False):
        datakey = 'train_plain'
        assert datakey in self.data

        print('===> Calculating KNN centroids.')

        torch.cuda.empty_cache()
        for model in self.networks.values():
            model.eval()
            

        feats_all, labels_all = [], []

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            for inputs, labels, idxs in tqdm(self.data[datakey]):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Calculate Features of each training data
                self.batch_forward(inputs, feature_ext=True)
                if second_fc:
                    feats_all.append(self.fc_features.cpu().numpy())
                else:
                    feats_all.append(self.features.cpu().numpy())
                labels_all.append(labels.cpu().numpy())

        feats = np.concatenate(feats_all) 
        print("feats shape: ", feats.shape) # 115846 * 512 
        labels = np.concatenate(labels_all)

        featmean = feats.mean(axis=0)
        print("featmean shape: ", featmean.shape) # 512, 
        def get_centroids(feats_, labels_):
            centroids = []
            for i in np.unique(labels_):
                # centroid of feature for each class
                centroids.append(np.mean(feats_[labels_==i], axis=0))
            return np.stack(centroids)
        # Get unnormalized centorids
        un_centers = get_centroids(feats, labels)
        print("uncenters shape: ", un_centers.shape) # 1000 * 512
        # Get l2n centorids
        l2n_feats = torch.Tensor(feats.copy())
        norm_l2n = torch.norm(l2n_feats, 2, 1, keepdim=True)
        l2n_feats = l2n_feats / norm_l2n
        l2n_centers = get_centroids(l2n_feats.numpy(), labels)
        print("l2n_centers shape: ", l2n_centers.shape) # 1000 * 512
        # Get cl2n centorids
        cl2n_feats = torch.Tensor(feats.copy())
        cl2n_feats = cl2n_feats - torch.Tensor(featmean)
        norm_cl2n = torch.norm(cl2n_feats, 2, 1, keepdim=True) # calculate matrix norm 
        cl2n_feats = cl2n_feats / norm_cl2n
        cl2n_centers = get_centroids(cl2n_feats.numpy(), labels)
        print("cl2n_centers shape: ", cl2n_centers.shape) # 1000 * 512
        return {'mean': featmean,
                'uncs': un_centers,
                'l2ncs': l2n_centers,
                'cl2ncs': cl2n_centers}
    
    # create memory bank containing all features
    def get_memory_bank(self, second_fc=False):
        datakey = 'train'

        print('===> Calculating memory bank.')

        torch.cuda.empty_cache()
        for model in self.networks.values():
            model.eval()
            

        feats_all, labels_all = [], []

        # Calculate initial centroids only on training data.
        with torch.set_grad_enabled(False):
            for inputs, labels, idxs in tqdm(self.data[datakey]):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Calculate Features of each training data
                self.batch_forward(inputs, feature_ext=True)
                if second_fc:
                    feats_all.append(self.fc_features.cpu().numpy())
                else:
                    feats_all.append(self.features.cpu().numpy())
                labels_all.append(labels.cpu().numpy())

        feats = np.concatenate(feats_all) 
        print("feats shape: ", feats.shape) # 115846 * 512 
        labels = np.concatenate(labels_all)
        return {
            'all_features': feats,
            'all_labels': labels
        }

    def save_model(self, epoch, best_epoch, best_model_weights, best_acc, centroids=None):

        model_states = {'epoch': epoch,
                'best_epoch': best_epoch,
                'state_dict_best': best_model_weights,
                'best_acc': best_acc,
                'centroids': centroids}
        if self.args.expname is not None:
            if self.args.trainable_logits_weight:
                suffix = '_with_weight'
            elif self.args.assignment_module:
                suffix = '_with_asm'
            else:
                suffix = ''
            model_dir = os.path.join(self.training_opt['log_dir'],
                                    '{}{}.pth'.format(self.args.expname, suffix))
        else:
            model_dir = os.path.join(self.training_opt['log_dir'],
                                    'final_model_checkpoint.pth')

        torch.save(model_states, model_dir)

    def output_logits(self, openset=False):
        filename = os.path.join(self.training_opt['log_dir'],
                                'logits_%s'%('open' if openset else 'close'))
        print("Saving total logits to: %s.npz" % filename)
        np.savez(filename,
                 logits=self.total_logits.detach().cpu().numpy(),
                 labels=self.total_labels.detach().cpu().numpy(),
                 paths=self.total_paths)
    
    def knn_predict(self, feature, feature_bank, feature_labels, classes, knn_k, knn_t):
        # compute cos similarity between each feature vector and feature bank ---> [B, N]

        sim_matrix = torch.mm(feature, feature_bank)
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(feature.size(0) * knn_k, classes)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels
    
        
    def train_asm(self):
        # When training the network
        print_str = ['Phase: train']
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        # Initialize best model
        best_model_weights = {}
        for key in self.networks.keys():
            best_model_weights[key] = copy.deepcopy(self.networks[key].state_dict())

        best_acc = 0.0
        best_epoch = 0
        best_f1 = 0.0

        end_epoch = self.training_opt['num_epochs']

        # Loop over epochs
        self.count = [0]*1000
        step_total = 0

        end_epoch = 10
        self.training_opt['display_step'] = 500
        for epoch in range(1, end_epoch + 1):

            self.ce_losses_stats = AverageMeter('ce_loss', ":4e")
            if self.args.second_fc:
                self.kl_loss_stats = AverageMeter('klloss', ":4e")
                self.d_losses_stats = AverageMeter('d_loss', ":4e")
            self.acc_stats = AverageMeter('accuracy', ':6.2f')

            self.epoch=epoch

            for model in self.networks.values():
                if self.args.assignment_module or self.args.trainable_logits_weight:
                    for module in model.modules():
                        if isinstance(module,  nn.BatchNorm2d):
                            module.eval()
                        else:
                            module.train()
                else:
                    model.train()
                    
            torch.cuda.empty_cache()

            for step, (normalized_features, logits, knn_logits, labels) in enumerate(self.asm_dataloader):
                normalized_features = normalized_features.to(self.device)
                logits = logits.to(self.device)
                knn_logits = knn_logits.to(self.device)
                labels = labels.to(self.device)
                # Break when step equal to epoch step
                if step == self.epoch_steps: #or image_t==self.training_data_num:
                    break
     
                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):
    
                    self.asm_output, _ = self.networks['asm'](normalized_features, logits, knn_logits, labels)

                    self.batch_loss(labels)
                  
                    self.batch_backward()
                    
                    preds = torch.sigmoid(self.asm_output) > 0.5
                    self.minibatch_acc = mic_acc_cal(preds, labels)
                        
                    # Output minibatch training results
                    if step % self.training_opt['display_step'] == 0:
                        step_total += step
                        minibatch_loss_perf = self.loss_perf.item()
                        if self.centers is not None and self.args.center_loss:
                            center_loss = self.center_loss.item()
                        else:
                            center_loss = 0
                        minibatch_dloss = self.dist_loss.item() if self.normalized_features_centroids is not None else None

                        print_str = ['Epoch: [%d/%d]'
                                     % (epoch, end_epoch),
                                     'Step: %5d'
                                     % (step),
                                     'Minibatch_loss_performance: %.3f'
                                     % (minibatch_loss_perf),
                                     'Minibatch_distance_loss: %.3f'
                                     % (center_loss) if (self.centers is not None and self.args.center_loss) else '',
                                     'Minibatch kl loss: %.3f'
                                     % (self.klloss.item()) if self.centers is not None and self.args.klloss else '',
                                     'Minibatch_accuracy_micro: %.3f'
                                      % (self.minibatch_acc)]
                        print_write(print_str, self.log_file)

                # update stats every iteration
                self.ce_losses_stats.update(self.loss_perf.item(), normalized_features.size(0))
                if self.args.second_fc and self.centers is not None and self.args.center_loss:
                    self.d_losses_stats.update(self.center_loss.item(), normalized_features.size(0))
                if self.args.second_fc and self.centers is not None and self.args.klloss:
                    self.kl_loss_stats.update(self.klloss.item(), normalized_features.size(0))
                self.acc_stats.update(self.minibatch_acc, normalized_features.size(0))

            # update tensorboard every epoch
            if self.args.tensorboard:
                self.writer.add_scalar('Loss/ce_train', self.ce_losses_stats.avg, epoch)
                if self.args.second_fc:
                    self.writer.add_scalar('Loss/dloss_train', self.d_losses_stats.avg, epoch)
                    if self.args.klloss:
                        self.writer.add_scalar('Loss/kl_loss_train', self.kl_loss_stats.avg, epoch)
                self.writer.add_scalar('Accuracy/Train', self.acc_stats.avg, epoch)

            # Set model modes and set scheduler
            # In training, step optimizer scheduler and set model to train()
            if self.args.assignment_module==False:
                self.model_optimizer_scheduler.step()
                if self.criterion_optimizer:
                    self.criterion_optimizer_scheduler.step()
           
            # After every epoch, validation
            self.eval(phase='val',epoch=epoch)
            # Under validation, the best model need to be updated
            if self.args.assignment_module == False:
                if self.eval_acc_mic_top1 > best_acc:
                    best_epoch = copy.deepcopy(epoch)
                    best_acc = copy.deepcopy(self.eval_acc_mic_top1)
                    best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                    best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
                    if self.args.second_fc:
                        best_model_weights['second_fc'] = copy.deepcopy(self.networks['second_fc'].state_dict())
                    if self.args.trainable_logits_weight:
                        best_model_weights['w1'] = copy.deepcopy(self.networks['w1'].state_dict())
                        best_model_weights['w2'] = copy.deepcopy(self.networks['w2'].state_dict())
                    if self.args.assignment_module:
                        best_model_weights['asm'] = copy.deepcopy(self.networks['asm'].state_dict())
            else:
                if self.f1 > best_f1:
                    best_epoch = copy.deepcopy(epoch)
                    best_f1 = copy.deepcopy(self.f1)
                    best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                    best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
                    if self.args.second_fc:
                        best_model_weights['second_fc'] = copy.deepcopy(self.networks['second_fc'].state_dict())
                    if self.args.trainable_logits_weight:
                        best_model_weights['w1'] = copy.deepcopy(self.networks['w1'].state_dict())
                        best_model_weights['w2'] = copy.deepcopy(self.networks['w2'].state_dict())
                    if self.args.assignment_module:
                        best_model_weights['asm'] = copy.deepcopy(self.networks['asm'].state_dict())
            self.best_model_weights =copy.deepcopy(best_model_weights)

            # if epoch is m_from: calculate centroids
            if epoch==self.args.m_from and self.args.assignment_module==False and self.args.trainable_logits_weight==False:
                self.feat_dict = self.get_knncentroids()
                if self.args.feat_norm:
                    self.centers = torch.from_numpy(self.feat_dict['l2ncs']).to(self.device)
                else:
                    self.centers = torch.from_numpy(self.feat_dict['l2ncs']).to(self.device)
       
        print()
        print('Training Complete.')
        if self.args.tensorboard:
            self.writer.close()

        # save updated centroids
        if self.centers is not None and self.args.assignment_module==False and self.args.trainable_logits_weight==False:
            print("saving M into pkl file..")
            with open(os.path.join(self.training_opt['log_dir'], '{}.pkl'.format(self.args.expname)), 'wb') as f:
                pickle.dump(self.centers, f)

        if self.args.assignment_module == False:
            print_str = ['Best validation accuracy is %.3f at epoch %d' % (best_acc, best_epoch)]
            self.save_model(epoch, best_epoch, best_model_weights, best_acc)
        else:
            print_str = ['Best binary classification f1 is %.3f at epoch %d' % (best_f1, best_epoch)]
            self.save_model(epoch, best_epoch, best_model_weights, best_f1)

        print_write(print_str, self.log_file)

        
        # Save the best model and best centroids if calculated
        if self.args.assignment_module==False:
            if self.args.trainable_logits_weight == False:
                cfeats = self.get_knncentroids()
                print('===> Saving final features to %s' %
                        os.path.join(self.training_opt['log_dir'], '{}_final.pkl'.format(self.args.expname)))
                with open(os.path.join(self.training_opt['log_dir'], '{}_final.pkl'.format(self.args.expname)), 'wb') as f:
                    pickle.dump(cfeats, f)
                print('Done')