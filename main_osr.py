import os
import argparse
import pprint
from data import dataloader_osr
from run_networks import model
import warnings
import pandas as pd
from utils import source_import
import pause 
from datetime import datetime
# pause.until(datetime(2021, 1, 6, 23, 59, 59))
# print("finished pausing")
# python main.py --config ./config/ImageNet_LT/stage_1.py --test
# ================
# LOAD CONFIGURATIONS
# python main.py --config ./config/ImageNet_LT/stage_1.py
data_root = {'ImageNet': '/mnt/lizhaochen', #change this
             'Places': '/home/public/dataset/Places365',
             'iNaturalist18': '/mnt/lizhaochen/iNaturalist18'}
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./config/ImageNet_LT/stage_1.py')
parser.add_argument('--dataset', default='ImageNet_LT', type=str)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
parser.add_argument('--tensorboard', default=False, action='store_true')

# ---------Mixup Parameters-------
parser.add_argument('--mixup', default=False, action='store_true')
parser.add_argument('--mixup_type', default='mixup_original', type=str)
parser.add_argument('--mixup_alpha', default=0.1, type=float)


parser.add_argument('--knn', default=False, action='store_true')
parser.add_argument('--feat_type', type=str, default='un')
parser.add_argument('--dist_type', type=str, default='cos')
parser.add_argument('--count_csv', type=str)
parser.add_argument('--acc_csv',type=str)
parser.add_argument('--acc', default=False, action='store_true')
parser.add_argument('--balms', default=False, action='store_true', help='using balanced meta softmax')
parser.add_argument('--rrloss', default=False, action='store_true')
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--memory', default=False, action='store_true', help='construct memory block')
parser.add_argument('--dloss', type=str, help='either cosine loss or mse loss')
parser.add_argument('--dloss_weight', type=float, default=0.1, help='store distance loss weight')
parser.add_argument('--hypersearch', default=False, action='store_true', help='search mse weight and store in hyper.txt')
parser.add_argument('--second_fc', default=False, action='store_true', help='using second fc for features')
parser.add_argument('--merge_logits', default=False, action='store_true', help='merge two logits into one final logit')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha controls the centroids update factor')
parser.add_argument('--lam', type=float, default=1.0, help='lambda controls weight of center loss')
parser.add_argument('--secondlr', type=float, default=0.1)
parser.add_argument('--feat_norm', default=False, action='store_true')
parser.add_argument('--m_from', default=1, type=int)
parser.add_argument('--path', type=str)
parser.add_argument('--ce_dist', default=False, action='store_true')
parser.add_argument('--klloss', default=False, action='store_true')
parser.add_argument('--description', type=str)
parser.add_argument('--m_freeze', default=False, action='store_true')
parser.add_argument('--k', default=5, type=int)
parser.add_argument('--ldam', default=False, action='store_true')
parser.add_argument('--manifold_mixup',default=False, action='store_true')
parser.add_argument('--logit_weight', default=1.0, type=float)
parser.add_argument('--w1', default=1, type=float)
parser.add_argument('--w2', default=1, type=float)
parser.add_argument('--resample', default=False, action='store_true')
parser.add_argument('--scaling_logits', default=False, action='store_true')
parser.add_argument('--center_loss', default=False, action='store_true')
parser.add_argument('--cal_knn_val', default=False, action='store_true')
parser.add_argument('--log_w', default=False, action='store_true')
parser.add_argument('--temperature', default=1.0, type=float)
parser.add_argument('--alpha_loss', default=0.7, type=float)
parser.add_argument('--assignment_module', default=False, action='store_true')
parser.add_argument('--trainable_logits_weight', default=False, action='store_true')
parser.add_argument('--second_dotproduct', default=False, action='store_true')
parser.add_argument('--asm_description', type=str)
parser.add_argument('--weight_norm', default=False, action='store_true', help='norm weight of fc layer')
parser.add_argument('--memory_bank', default=False, action='store_true', help='memory bank of all features')
parser.add_argument('--lr_scheduler', type=str, default='cos')
parser.add_argument('--second_head_alpha', type=float, default=0.1, help='trade-off loss hyper-parameters of of student model')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

def update(config, args):
    # Change parameters
    config = source_import(args.config).config
    # Using BALMS
    if args.balms:
        perf_loss_param = {'freq_path': './cls_freq/ImageNet_LT.json'}
        performance_loss = {'def_file': './loss/BalancedSoftmaxLoss.py', 'loss_params': perf_loss_param,
                                'optim_params': None, 'weight': 1.0}
        config['criterions']['Performanceloss'] = performance_loss

    return config

test_mode = args.test
test_open = args.test_open
if test_open:
    test_mode = True
output_logits = args.output_logits

config = source_import(args.config).config
# config = update(config, args)

training_opt = config['training_opt']
if args.resample:
    training_opt['sampler'] = {'def_file': './data/ClassAwareSampler.py', 'num_samples_cls': 4, 'type': 'ClassAwareSampler'}
# change
relatin_opt = config['memory']
dataset = training_opt['dataset']

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])

print('Loading dataset from: %s' % data_root[dataset.rstrip('_LT')])
# pprint.pprint(config)
config['label_info'] = None
config['dloss_weight'] = args.dloss_weight
config['memory_block'] = args.memory
config['merge_logits'] = args.merge_logits


if not test_mode: # test mode is false

    sampler_defs = training_opt['sampler']
    if sampler_defs:
        if sampler_defs['type'] == 'ClassAwareSampler':
            sampler_dic = {
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {'num_samples_cls': sampler_defs['num_samples_cls']}
            }
        elif sampler_defs['type'] in ['MixedPrioritizedSampler',
                                      'ClassPrioritySampler']:
            sampler_dic = {
                'sampler': source_import(sampler_defs['def_file']).get_sampler(),
                'params': {k: v for k, v in sampler_defs.items() \
                           if k not in ['type', 'def_file']}
            }
    else:
        sampler_dic = None

    if dataset == 'iNaturalist18':
        phase_bank = ['train', 'val', 'train_plain']
    else:
        phase_bank = ['train', 'val', 'train_plain', 'test']

    data = {x: dataloader_osr.load_data(data_root=data_root[dataset.rstrip('_LT')], dataset=dataset, phase=x,
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=sampler_dic if x!='train_plain' else None,
                                    num_workers=training_opt['num_workers'])
            for x in (phase_bank)}# if relatin_opt['init_centroids'] else ['train', 'val'])}

    lbs = data['train'].dataset.labels
    counts = []
    for i in range(1000):
        counts.append(lbs.count(i))
    config['label_counts'] = counts
    counts = pd.DataFrame(counts)
    #tail classes
    tail = counts[counts[0]<=20].index.tolist()
    median = counts[(counts[0]>20)&(counts[0]<100)].index.tolist()
    head = counts[counts[0]>100].index.tolist()
    config['label_info'] = [tail, median, head]

    training_model = model(args, config, data, test=False)
    print("entering train function in run_networks")
    if args.assignment_module:
        training_model.train_asm()
    else:
        training_model.train()
    
    if dataset != 'iNaturalist18':
        training_model.eval(phase='test', openset=test_open)

else:

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    print('Under testing phase, we load training data simply to calculate training data number for each class.')
    splits = ['train', 'test', 'train_plain']
    if args.knn:
        splits.append('train_plain')
    data = {x: dataloader_osr.load_data(data_root=data_root[dataset.rstrip('_LT')], dataset=dataset, phase=x,
                                    batch_size=training_opt['batch_size'],
                                    sampler_dic=None,
                                    test_open=test_open,
                                    num_workers=training_opt['num_workers'],
                                    shuffle=False)
            for x in splits}

    lbs = data['train'].dataset.labels
    counts = []
    for i in range(1000):
        counts.append(lbs.count(i))
    config['label_counts'] = counts
    counts = pd.DataFrame(counts)
    #tail classes
    tail = counts[counts[0]<=20].index.tolist()
    median = counts[(counts[0]>20)&(counts[0]<100)].index.tolist()
    head = counts[counts[0]>100].index.tolist()
    config['label_info'] = [tail, median, head]
    
    training_model = model(args, config, data, test=True)
    # training_model.load_model()
    training_model.eval(phase='test', openset=test_open)

    if output_logits:
        training_model.output_logits(openset=test_open)

print('ALL COMPLETED.')
